package unit_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"testing"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// gray builds a one-row image of grey levels.
func gray(levels ...uint8) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, len(levels), 1))
	for x, v := range levels {
		img.Set(x, 0, color.RGBA{v, v, v, 255})
	}
	return img
}

// row reads the red channel of the first row.
func row(img *image.RGBA) []uint8 {
	out := make([]uint8, img.Rect.Dx())
	for x := range out {
		out[x] = img.RGBAAt(img.Rect.Min.X+x, img.Rect.Min.Y).R
	}
	return out
}

func equalBytes(a, b []uint8) bool { return bytes.Equal(a, b) }

func TestResizeIsBilinearWithAlignedPixelCentres(t *testing.T) {
	t.Parallel()

	// Up: destination centres at -0.25, 0.25, 0.75 and 1.25 source pixels.
	up, err := onnx.Resize(gray(0, 255), 4, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := row(up), []uint8{0, 64, 191, 255}; !equalBytes(got, want) {
		t.Errorf("upscaled row = %v, want %v", got, want)
	}

	// Down: destination centres at 0.5 and 2.5 source pixels.
	down, err := onnx.Resize(gray(0, 100, 200, 250), 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := row(down), []uint8{50, 225}; !equalBytes(got, want) {
		t.Errorf("downscaled row = %v, want %v", got, want)
	}

	same, err := onnx.Resize(gray(3, 7, 11), 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := row(same), []uint8{3, 7, 11}; !equalBytes(got, want) {
		t.Errorf("same-size row = %v, want %v", got, want)
	}

	if _, err := onnx.Resize(gray(1), 0, 1); err == nil {
		t.Error("Resize accepted a width of 0")
	}
}

func TestLetterboxKeepsTheAspectRatioAndCentres(t *testing.T) {
	t.Parallel()

	white := image.NewRGBA(image.Rect(0, 0, 4, 2))
	for i := range white.Pix {
		white.Pix[i] = 255
	}
	fill := color.RGBA{114, 114, 114, 255}

	canvas, info, err := onnx.Letterbox(white, 4, 4, fill)
	if err != nil {
		t.Fatal(err)
	}
	want := onnx.LetterboxInfo{Scale: 1, PadX: 0, PadY: 1, Width: 4, Height: 2}
	if info != want {
		t.Fatalf("info = %+v, want %+v", info, want)
	}
	for y, level := range []uint8{114, 255, 255, 114} {
		for x := 0; x < 4; x++ {
			if got := canvas.RGBAAt(x, y); got.R != level || got.A != 255 {
				t.Fatalf("canvas (%d,%d) = %v, want level %d", x, y, got, level)
			}
		}
	}
	if x, y := info.ToSource(2, 1.5); x != 2 || y != 0.5 {
		t.Errorf("ToSource(2, 1.5) = (%v, %v), want (2, 0.5)", x, y)
	}

	tall := image.NewRGBA(image.Rect(0, 0, 2, 4))
	_, info, err = onnx.Letterbox(tall, 8, 8, fill)
	if err != nil {
		t.Fatal(err)
	}
	want = onnx.LetterboxInfo{Scale: 2, PadX: 2, PadY: 0, Width: 4, Height: 8}
	if info != want {
		t.Fatalf("info = %+v, want %+v", info, want)
	}
	if x, y := info.ToSource(6, 8); x != 2 || y != 4 {
		t.Errorf("ToSource(6, 8) = (%v, %v), want the far corner (2, 4)", x, y)
	}
}

func TestCropCopiesTheClippedRegion(t *testing.T) {
	t.Parallel()

	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			img.Set(x, y, color.RGBA{uint8(x + 4*y), 0, 0, 255})
		}
	}

	crop, err := onnx.Crop(img, image.Rect(1, 1, 3, 3))
	if err != nil {
		t.Fatal(err)
	}
	if crop.Rect != image.Rect(0, 0, 2, 2) {
		t.Fatalf("crop bounds = %v, want a 2x2 image at the origin", crop.Rect)
	}
	got := []uint8{crop.RGBAAt(0, 0).R, crop.RGBAAt(1, 0).R, crop.RGBAAt(0, 1).R, crop.RGBAAt(1, 1).R}
	if want := []uint8{5, 6, 9, 10}; !equalBytes(got, want) {
		t.Errorf("crop = %v, want %v", got, want)
	}

	edge, err := onnx.Crop(img, image.Rect(-1, -1, 1, 1))
	if err != nil {
		t.Fatal(err)
	}
	if edge.Rect.Dx() != 1 || edge.Rect.Dy() != 1 || edge.RGBAAt(0, 0).R != 0 {
		t.Errorf("a crop over the edge = %v %v, want the one pixel inside", edge.Rect, edge.RGBAAt(0, 0))
	}

	if _, err := onnx.Crop(img, image.Rect(10, 10, 12, 12)); err == nil {
		t.Error("Crop accepted a rectangle outside the image")
	}
}

func TestNCHWNormalisesEachPlane(t *testing.T) {
	t.Parallel()

	img := image.NewRGBA(image.Rect(0, 0, 2, 1))
	img.Set(0, 0, color.RGBA{255, 0, 0, 255})
	img.Set(1, 0, color.RGBA{0, 128, 255, 255})
	half := [3]float32{0.5, 0.5, 0.5}
	g := (128.0/255.0 - 0.5) / 0.5

	rgb, err := onnx.NCHW(img, onnx.Normalization{Mean: half, Std: half})
	if err != nil {
		t.Fatal(err)
	}
	if got := rgb.Shape; len(got) != 4 || got[0] != 1 || got[1] != 3 || got[2] != 1 || got[3] != 2 {
		t.Fatalf("shape = %v, want [1 3 1 2]", got)
	}
	assertClose(t, "rgb", rgb.Float32, 1, -1, -1, float32(g), -1, 1)

	bgr, err := onnx.NCHW(img, onnx.Normalization{Mean: half, Std: half, BGR: true})
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "bgr", bgr.Float32, -1, 1, -1, float32(g), 1, -1)

	imagenet := onnx.Normalization{Mean: [3]float32{0.485, 0.456, 0.406}, Std: [3]float32{0.229, 0.224, 0.225}}
	in, err := onnx.NCHW(img, imagenet)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "imagenet", in.Float32,
		(1-0.485)/0.229, (0-0.485)/0.229,
		(0-0.456)/0.224, (128.0/255-0.456)/0.224,
		(0-0.406)/0.225, (1-0.406)/0.225)

	if _, err := onnx.NCHW(img, onnx.Normalization{}); err == nil {
		t.Error("NCHW accepted a standard deviation of 0")
	}
}

func assertClose(t *testing.T, name string, got []float32, want ...float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s = %v, want %v", name, got, want)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("%s = %v, want %v", name, got, want)
		}
	}
}

// quadrants is a 32x16 image: red top-left, blue top-right, green
// bottom-left and white bottom-right, so every orientation puts a different
// pair of colours in the top-left and bottom-right corners.
func quadrants() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, 32, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 32; x++ {
			var c color.RGBA
			switch {
			case x < 16 && y < 8:
				c = color.RGBA{230, 10, 10, 255}
			case y < 8:
				c = color.RGBA{10, 10, 230, 255}
			case x < 16:
				c = color.RGBA{10, 230, 10, 255}
			default:
				c = color.RGBA{240, 240, 240, 255}
			}
			img.Set(x, y, c)
		}
	}
	return img
}

// byteOrder is a TIFF byte order that can both read and append.
type byteOrder interface {
	binary.ByteOrder
	binary.AppendByteOrder
}

// withOrientation encodes an image as a JPEG carrying an EXIF block whose
// first directory holds the orientation, in the given TIFF byte order.
func withOrientation(t *testing.T, img image.Image, orientation uint16, order byteOrder) []byte {
	t.Helper()
	var encoded bytes.Buffer
	if err := jpeg.Encode(&encoded, img, &jpeg.Options{Quality: 95}); err != nil {
		t.Fatal(err)
	}

	tiff := []byte("MM")
	if order == binary.LittleEndian {
		tiff = []byte("II")
	}
	tiff = order.AppendUint16(tiff, 42)
	tiff = order.AppendUint32(tiff, 8)
	tiff = order.AppendUint16(tiff, 1)
	tiff = order.AppendUint16(tiff, 0x0112)
	tiff = order.AppendUint16(tiff, 3)
	tiff = order.AppendUint32(tiff, 1)
	tiff = order.AppendUint16(tiff, orientation)
	tiff = append(tiff, 0, 0)
	tiff = order.AppendUint32(tiff, 0)

	segment := append([]byte("Exif\x00\x00"), tiff...)
	out := []byte{0xFF, 0xD8, 0xFF, 0xE1}
	out = binary.BigEndian.AppendUint16(out, uint16(2+len(segment)))
	out = append(out, segment...)
	return append(out, encoded.Bytes()[2:]...)
}

// The colours of the quadrants, with room for the JPEG's loss.
func isRed(c color.RGBA) bool   { return c.R > 150 && c.G < 100 && c.B < 100 }
func isBlue(c color.RGBA) bool  { return c.B > 150 && c.R < 100 && c.G < 100 }
func isGreen(c color.RGBA) bool { return c.G > 150 && c.R < 100 && c.B < 100 }
func isWhite(c color.RGBA) bool { return c.R > 150 && c.G > 150 && c.B > 150 }

func TestDecodeImageTurnsAJPEGUprightByItsEXIFOrientation(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		orientation   uint16
		order         byteOrder
		width, height int
		// first and last are the pixels at the top-left corner and the
		// bottom-right corner.
		first, last func(color.RGBA) bool
	}{
		{1, binary.BigEndian, 32, 16, isRed, isWhite},
		{2, binary.BigEndian, 32, 16, isBlue, isGreen},
		{3, binary.LittleEndian, 32, 16, isWhite, isRed},
		{4, binary.BigEndian, 32, 16, isGreen, isBlue},
		{5, binary.BigEndian, 16, 32, isRed, isWhite},
		{6, binary.LittleEndian, 16, 32, isGreen, isBlue},
		{7, binary.BigEndian, 16, 32, isWhite, isRed},
		{8, binary.BigEndian, 16, 32, isBlue, isGreen},
	} {
		img, err := onnx.DecodeImage(bytes.NewReader(withOrientation(t, quadrants(), tc.orientation, tc.order)))
		if err != nil {
			t.Fatalf("orientation %d: %v", tc.orientation, err)
		}
		if img.Rect != image.Rect(0, 0, tc.width, tc.height) {
			t.Errorf("orientation %d: bounds = %v, want %dx%d", tc.orientation, img.Rect, tc.width, tc.height)
			continue
		}
		if first := img.RGBAAt(2, 2); !tc.first(first) {
			t.Errorf("orientation %d: the top-left pixel is %v", tc.orientation, first)
		}
		if last := img.RGBAAt(tc.width-3, tc.height-3); !tc.last(last) {
			t.Errorf("orientation %d: the bottom-right pixel is %v", tc.orientation, last)
		}
	}
}

func TestDecodeImageReadsAPNGExactly(t *testing.T) {
	t.Parallel()

	src := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	src.Set(0, 0, color.NRGBA{10, 20, 30, 255})
	src.Set(1, 0, color.NRGBA{200, 100, 50, 255})
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, src); err != nil {
		t.Fatal(err)
	}

	img, err := onnx.DecodeImage(&encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got := img.RGBAAt(0, 0); got != (color.RGBA{10, 20, 30, 255}) {
		t.Errorf("pixel 0 = %v", got)
	}
	if got := img.RGBAAt(1, 0); got != (color.RGBA{200, 100, 50, 255}) {
		t.Errorf("pixel 1 = %v", got)
	}
}

func TestDecodeImageRefusesWhatItCannotOrShouldNotRead(t *testing.T) {
	t.Parallel()

	if _, err := onnx.DecodeImage(bytes.NewReader([]byte("not an image"))); err == nil {
		t.Error("DecodeImage accepted text")
	}

	// A one-pixel PNG whose header is rewritten to claim 100000 x 100000
	// pixels, with the header's checksum recomputed so it reads as valid.
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewGray(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	data := encoded.Bytes()
	ihdr := data[12 : 12+4+13] // chunk type and payload
	binary.BigEndian.PutUint32(ihdr[4:], 100000)
	binary.BigEndian.PutUint32(ihdr[8:], 100000)
	binary.BigEndian.PutUint32(data[12+4+13:], crc32.ChecksumIEEE(ihdr))

	if _, err := onnx.DecodeImage(bytes.NewReader(data)); !errors.Is(err, onnx.ErrImageTooLarge) {
		t.Fatalf("DecodeImage = %v, want ErrImageTooLarge", err)
	}
}
