package onnx

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg" // registers the JPEG decoder DecodeImage reads
	_ "image/png"  // registers the PNG decoder DecodeImage reads
	"io"
	"math"
)

// MaxImagePixels is the largest image DecodeImage accepts, in pixels. It is
// checked from the header, before the pixels are decoded, so an upload that
// declares a huge canvas costs a header read and not the memory it asks for.
const MaxImagePixels = 50_000_000

// ErrImageTooLarge is an image whose header declares more than MaxImagePixels.
var ErrImageTooLarge = errors.New("onnx: image too large")

// DecodeImage decodes a JPEG or PNG image and returns it upright.
//
// A JPEG carrying an EXIF orientation is turned the way the camera meant it,
// so a portrait photographed sideways reaches the model the way a person sees
// it. The result is always an *image.RGBA with its origin at (0, 0).
func DecodeImage(r io.Reader) (*image.RGBA, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("onnx: reading image: %w", err)
	}
	config, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("onnx: decoding image header: %w", err)
	}
	if int64(config.Width)*int64(config.Height) > MaxImagePixels {
		return nil, fmt.Errorf("%w: %dx%d is more than %d pixels", ErrImageTooLarge, config.Width, config.Height, MaxImagePixels)
	}
	decoded, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("onnx: decoding image: %w", err)
	}
	return orient(toRGBA(decoded), exifOrientation(data)), nil
}

// toRGBA copies an image into an *image.RGBA whose origin is (0, 0). An
// image with transparency is composited over black, which is what its
// premultiplied colour is.
func toRGBA(src image.Image) *image.RGBA {
	b := src.Bounds()
	if rgba, ok := src.(*image.RGBA); ok && b.Min == (image.Point{}) {
		return rgba
	}
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), src, b.Min, draw.Src)
	return dst
}

// Resize scales an image to width by height with bilinear interpolation.
//
// Pixel centres are aligned the way OpenCV's INTER_LINEAR aligns them, so a
// preprocessing step ported from a Python reference produces the same values
// to within rounding. There is no antialiasing: shrinking by more than half
// skips pixels, as that reference does.
func Resize(src image.Image, width, height int) (*image.RGBA, error) {
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("onnx: resize to %dx%d: the size has to be positive", width, height)
	}
	in := toRGBA(src)
	sw, sh := in.Rect.Dx(), in.Rect.Dy()
	if sw == 0 || sh == 0 {
		return nil, errors.New("onnx: resize of an empty image")
	}
	out := image.NewRGBA(image.Rect(0, 0, width, height))

	xs := make([]sample, width)
	for x := range xs {
		xs[x] = sampleAt(x, sw, width)
	}
	for y := 0; y < height; y++ {
		sy := sampleAt(y, sh, height)
		row0 := in.Pix[sy.lo*in.Stride:]
		row1 := in.Pix[sy.hi*in.Stride:]
		dst := out.Pix[y*out.Stride:]
		for x, sx := range xs {
			for c := 0; c < 4; c++ {
				top := lerp(float64(row0[sx.lo*4+c]), float64(row0[sx.hi*4+c]), sx.frac)
				bottom := lerp(float64(row1[sx.lo*4+c]), float64(row1[sx.hi*4+c]), sx.frac)
				dst[x*4+c] = clampByte(lerp(top, bottom, sy.frac))
			}
		}
	}
	return out, nil
}

// sample is where one destination pixel reads from along one axis: the two
// source pixels either side of it and the weight of the second.
type sample struct {
	lo, hi int
	frac   float64
}

// sampleAt maps destination index d of n onto a source axis of size s.
func sampleAt(d, s, n int) sample {
	pos := (float64(d)+0.5)*float64(s)/float64(n) - 0.5
	if pos <= 0 {
		return sample{lo: 0, hi: 0}
	}
	if pos >= float64(s-1) {
		return sample{lo: s - 1, hi: s - 1}
	}
	lo := int(math.Floor(pos))
	return sample{lo: lo, hi: lo + 1, frac: pos - float64(lo)}
}

func lerp(a, b, t float64) float64 { return a + (b-a)*t }

func clampByte(v float64) uint8 {
	v = math.Round(v)
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// LetterboxInfo is where Letterbox placed the source inside its canvas.
type LetterboxInfo struct {
	// Scale is the factor the source was resized by.
	Scale float64
	// PadX and PadY are the canvas columns left of the image and the rows
	// above it.
	PadX, PadY int
	// Width and Height are the size the source was resized to.
	Width, Height int
}

// ToSource maps a point on the canvas back onto the source image, which is
// how a box a model found on the canvas is drawn on the original.
func (l LetterboxInfo) ToSource(x, y float64) (float64, float64) {
	return (x - float64(l.PadX)) / l.Scale, (y - float64(l.PadY)) / l.Scale
}

// Letterbox resizes an image to fit width by height without changing its
// aspect ratio, centres it, and fills the rest of the canvas with fill.
func Letterbox(src image.Image, width, height int, fill color.RGBA) (*image.RGBA, LetterboxInfo, error) {
	if width <= 0 || height <= 0 {
		return nil, LetterboxInfo{}, fmt.Errorf("onnx: letterbox to %dx%d: the size has to be positive", width, height)
	}
	b := src.Bounds()
	if b.Empty() {
		return nil, LetterboxInfo{}, errors.New("onnx: letterbox of an empty image")
	}
	scale := math.Min(float64(width)/float64(b.Dx()), float64(height)/float64(b.Dy()))
	info := LetterboxInfo{
		Scale:  scale,
		Width:  max(1, min(width, int(math.Round(float64(b.Dx())*scale)))),
		Height: max(1, min(height, int(math.Round(float64(b.Dy())*scale)))),
	}
	info.PadX = (width - info.Width) / 2
	info.PadY = (height - info.Height) / 2

	resized, err := Resize(src, info.Width, info.Height)
	if err != nil {
		return nil, LetterboxInfo{}, err
	}
	canvas := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(canvas, canvas.Bounds(), image.NewUniform(fill), image.Point{}, draw.Src)
	draw.Draw(canvas, image.Rect(info.PadX, info.PadY, info.PadX+info.Width, info.PadY+info.Height), resized, image.Point{}, draw.Src)
	return canvas, info, nil
}

// Crop copies the part of an image inside r, clipped to the image's bounds.
// The result's origin is (0, 0). A rectangle that misses the image entirely
// is refused.
func Crop(src image.Image, r image.Rectangle) (*image.RGBA, error) {
	clipped := r.Canon().Intersect(src.Bounds())
	if clipped.Empty() {
		return nil, fmt.Errorf("onnx: crop %v lies outside the image %v", r, src.Bounds())
	}
	dst := image.NewRGBA(image.Rect(0, 0, clipped.Dx(), clipped.Dy()))
	draw.Draw(dst, dst.Bounds(), src, clipped.Min, draw.Src)
	return dst, nil
}

// Normalization is how NCHW turns 8-bit channels into a model's floats.
type Normalization struct {
	// Mean and Std are per channel, in the output's channel order, on the
	// 0..1 scale: a value is (pixel/255 - Mean) / Std. Mean {0.5, 0.5, 0.5}
	// with Std {0.5, 0.5, 0.5} maps 0..255 onto -1..1; Mean {0, 0, 0} with Std
	// {1, 1, 1} maps it onto 0..1.
	Mean, Std [3]float32
	// BGR writes the channels blue, green, red instead of red, green, blue.
	BGR bool
}

// NCHW lays an image out as a float32 tensor of shape [1, 3, height, width],
// one plane per channel, normalised as n says. Alpha is dropped.
func NCHW(src image.Image, n Normalization) (Tensor, error) {
	for c := 0; c < 3; c++ {
		if n.Std[c] == 0 {
			return Tensor{}, fmt.Errorf("onnx: normalization channel %d has a standard deviation of 0", c)
		}
	}
	in := toRGBA(src)
	w, h := in.Rect.Dx(), in.Rect.Dy()
	if w == 0 || h == 0 {
		return Tensor{}, errors.New("onnx: NCHW of an empty image")
	}
	order := [3]int{0, 1, 2}
	if n.BGR {
		order = [3]int{2, 1, 0}
	}
	plane := w * h
	data := make([]float32, 3*plane)
	for y := 0; y < h; y++ {
		row := in.Pix[y*in.Stride:]
		for x := 0; x < w; x++ {
			for c, source := range order {
				v := float32(row[x*4+source]) / 255
				data[c*plane+y*w+x] = (v - n.Mean[c]) / n.Std[c]
			}
		}
	}
	return Tensor{Shape: []int64{1, 3, int64(h), int64(w)}, Float32: data}, nil
}
