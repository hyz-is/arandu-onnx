package onnx

import (
	"encoding/binary"
	"image"
)

// exifOrientation reads the EXIF orientation of a JPEG: 1 to 8, or 1 when the
// data is not a JPEG, carries no EXIF block, or carries one this reader cannot
// follow. Only the first image file directory is read, which is where the
// orientation tag lives.
func exifOrientation(data []byte) int {
	if len(data) < 4 || data[0] != 0xFF || data[1] != 0xD8 {
		return 1
	}
	for at := 2; at+4 <= len(data); {
		if data[at] != 0xFF {
			return 1
		}
		marker := data[at+1]
		if marker == 0xFF { // fill byte
			at++
			continue
		}
		if marker == 0xDA || marker == 0xD9 { // start of scan, end of image
			return 1
		}
		size := int(binary.BigEndian.Uint16(data[at+2:]))
		if size < 2 || at+2+size > len(data) {
			return 1
		}
		segment := data[at+4 : at+2+size]
		if marker == 0xE1 && len(segment) > 6 && string(segment[:6]) == "Exif\x00\x00" {
			return tiffOrientation(segment[6:])
		}
		at += 2 + size
	}
	return 1
}

// tiffOrientation finds tag 0x0112 in the first directory of a TIFF header.
func tiffOrientation(tiff []byte) int {
	if len(tiff) < 8 {
		return 1
	}
	var order binary.ByteOrder
	switch string(tiff[:2]) {
	case "II":
		order = binary.LittleEndian
	case "MM":
		order = binary.BigEndian
	default:
		return 1
	}
	if order.Uint16(tiff[2:]) != 42 {
		return 1
	}
	dir := int(order.Uint32(tiff[4:]))
	if dir < 8 || dir+2 > len(tiff) {
		return 1
	}
	count := int(order.Uint16(tiff[dir:]))
	for i := 0; i < count; i++ {
		entry := dir + 2 + 12*i
		if entry+12 > len(tiff) {
			return 1
		}
		// Tag 0x0112 is the orientation, of type SHORT (3) and count 1, so
		// its value sits in the first two bytes of the value field.
		if order.Uint16(tiff[entry:]) == 0x0112 && order.Uint16(tiff[entry+2:]) == 3 {
			if v := int(order.Uint16(tiff[entry+8:])); v >= 1 && v <= 8 {
				return v
			}
			return 1
		}
	}
	return 1
}

// orient turns an image the way an EXIF orientation says the camera held it.
func orient(src *image.RGBA, orientation int) *image.RGBA {
	if orientation <= 1 || orientation > 8 {
		return src
	}
	w, h := src.Rect.Dx(), src.Rect.Dy()
	dw, dh := w, h
	if orientation >= 5 {
		dw, dh = h, w
	}
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			var sx, sy int
			switch orientation {
			case 2: // mirrored horizontally
				sx, sy = w-1-x, y
			case 3: // rotated 180
				sx, sy = w-1-x, h-1-y
			case 4: // mirrored vertically
				sx, sy = x, h-1-y
			case 5: // transposed
				sx, sy = y, x
			case 6: // rotated 90 clockwise to display
				sx, sy = y, h-1-x
			case 7: // transversed
				sx, sy = w-1-y, h-1-x
			case 8: // rotated 90 counter-clockwise to display
				sx, sy = w-1-y, x
			}
			copy(dst.Pix[y*dst.Stride+x*4:y*dst.Stride+x*4+4], src.Pix[sy*src.Stride+sx*4:sy*src.Stride+sx*4+4])
		}
	}
	return dst
}
