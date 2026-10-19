package graph

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
)

// The protobuf wire types.
const (
	wireVarint  = 0
	wireFixed64 = 1
	wireBytes   = 2
	wireFixed32 = 5
)

// field is one encoded field of a message: its number, its wire type, the
// whole encoding (tag included) and the payload. The payload of a varint or a
// fixed-width field is its raw bytes; that of a length-delimited field is its
// content.
type field struct {
	num, wire  int
	raw, value []byte
}

// rawField is a field this package does not interpret, kept as it was read so
// that writing the message again writes it again.
type rawField struct {
	num int
	raw []byte
}

// each calls fn with every field of a message, in the order it is encoded.
func each(data []byte, fn func(f field) error) error {
	for offset := 0; offset < len(data); {
		start := offset
		tag, n := binary.Uvarint(data[offset:])
		if n <= 0 {
			return fmt.Errorf("%w: unreadable tag at byte %d", ErrMalformed, offset)
		}
		offset += n
		num, wire := int(tag>>3), int(tag&7)
		if num <= 0 || tag>>3 > math.MaxInt32 {
			return fmt.Errorf("%w: field number %d at byte %d", ErrMalformed, tag>>3, start)
		}
		var value []byte
		switch wire {
		case wireVarint:
			_, n := binary.Uvarint(data[offset:])
			if n <= 0 {
				return fmt.Errorf("%w: unreadable varint at byte %d", ErrMalformed, offset)
			}
			value = data[offset : offset+n]
			offset += n
		case wireFixed64:
			if len(data)-offset < 8 {
				return fmt.Errorf("%w: truncated fixed64 at byte %d", ErrMalformed, offset)
			}
			value = data[offset : offset+8]
			offset += 8
		case wireFixed32:
			if len(data)-offset < 4 {
				return fmt.Errorf("%w: truncated fixed32 at byte %d", ErrMalformed, offset)
			}
			value = data[offset : offset+4]
			offset += 4
		case wireBytes:
			length, n := binary.Uvarint(data[offset:])
			if n <= 0 {
				return fmt.Errorf("%w: unreadable length at byte %d", ErrMalformed, offset)
			}
			offset += n
			if length > uint64(len(data)-offset) {
				return fmt.Errorf("%w: a field of %d bytes at byte %d runs past the end", ErrMalformed, length, offset)
			}
			value = data[offset : offset+int(length)]
			offset += int(length)
		default:
			return fmt.Errorf("%w: wire type %d at byte %d is not one an ONNX model uses", ErrMalformed, wire, start)
		}
		if err := fn(field{num: num, wire: wire, raw: data[start:offset], value: value}); err != nil {
			return err
		}
	}
	return nil
}

// keep is the rawField of f.
func (f field) keep() rawField { return rawField{num: f.num, raw: f.raw} }

// uint reads a varint field.
func (f field) uint() (uint64, error) {
	if f.wire != wireVarint {
		return 0, fmt.Errorf("%w: field %d has wire type %d where a varint is expected", ErrMalformed, f.num, f.wire)
	}
	v, _ := binary.Uvarint(f.value)
	return v, nil
}

// int reads a varint field as the two's complement int64 protobuf writes.
func (f field) int() (int64, error) {
	v, err := f.uint()
	return int64(v), err
}

// bytes reads a length-delimited field.
func (f field) bytes() ([]byte, error) {
	if f.wire != wireBytes {
		return nil, fmt.Errorf("%w: field %d has wire type %d where a length-delimited value is expected", ErrMalformed, f.num, f.wire)
	}
	return f.value, nil
}

// str reads a string field.
func (f field) str() (string, error) {
	b, err := f.bytes()
	return string(b), err
}

// float reads a float field.
func (f field) float() (float32, error) {
	if f.wire != wireFixed32 {
		return 0, fmt.Errorf("%w: field %d has wire type %d where a float is expected", ErrMalformed, f.num, f.wire)
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(f.value)), nil
}

// appendInts reads a repeated varint field, packed or not.
func (f field) appendInts(dst []int64) ([]int64, error) {
	switch f.wire {
	case wireVarint:
		v, _ := binary.Uvarint(f.value)
		return append(dst, int64(v)), nil
	case wireBytes:
		for b := f.value; len(b) > 0; {
			v, n := binary.Uvarint(b)
			if n <= 0 {
				return nil, fmt.Errorf("%w: unreadable packed varint in field %d", ErrMalformed, f.num)
			}
			dst = append(dst, int64(v))
			b = b[n:]
		}
		return dst, nil
	default:
		return nil, fmt.Errorf("%w: field %d has wire type %d where integers are expected", ErrMalformed, f.num, f.wire)
	}
}

// appendFloats reads a repeated float field, packed or not.
func (f field) appendFloats(dst []float32) ([]float32, error) {
	switch f.wire {
	case wireFixed32:
		return append(dst, math.Float32frombits(binary.LittleEndian.Uint32(f.value))), nil
	case wireBytes:
		if len(f.value)%4 != 0 {
			return nil, fmt.Errorf("%w: packed floats in field %d are %d bytes", ErrMalformed, f.num, len(f.value))
		}
		for i := 0; i < len(f.value); i += 4 {
			dst = append(dst, math.Float32frombits(binary.LittleEndian.Uint32(f.value[i:])))
		}
		return dst, nil
	default:
		return nil, fmt.Errorf("%w: field %d has wire type %d where floats are expected", ErrMalformed, f.num, f.wire)
	}
}

// encoder writes a message as the C++ protobuf library does: fields in
// ascending field number, the values of a repeated field in their order, and
// the fields this package keeps unread in the order they were read.
type encoder struct {
	entries []entry
}

type entry struct {
	num  int
	data []byte
}

func tag(num, wire int) []byte {
	return binary.AppendUvarint(nil, uint64(num)<<3|uint64(wire))
}

func (e *encoder) add(num int, data []byte) { e.entries = append(e.entries, entry{num, data}) }

// varint writes a varint field.
func (e *encoder) varint(num int, v uint64) {
	e.add(num, binary.AppendUvarint(tag(num, wireVarint), v))
}

// bytes writes a length-delimited field.
func (e *encoder) bytes(num int, b []byte) {
	out := binary.AppendUvarint(tag(num, wireBytes), uint64(len(b)))
	e.add(num, append(out, b...))
}

// str writes a string field.
func (e *encoder) str(num int, s string) { e.bytes(num, []byte(s)) }

// float writes a float field.
func (e *encoder) float(num int, v float32) {
	e.add(num, binary.LittleEndian.AppendUint32(tag(num, wireFixed32), math.Float32bits(v)))
}

// ints writes a repeated varint field unpacked, which is what proto2 does
// for a field not marked packed.
func (e *encoder) ints(num int, vs []int64) {
	for _, v := range vs {
		e.varint(num, uint64(v))
	}
}

// floats writes a repeated float field unpacked.
func (e *encoder) floats(num int, vs []float32) {
	for _, v := range vs {
		e.float(num, v)
	}
}

// packedInts writes a repeated varint field marked packed, or nothing when it
// is empty.
func (e *encoder) packedInts(num int, vs []int64) {
	if len(vs) == 0 {
		return
	}
	var b []byte
	for _, v := range vs {
		b = binary.AppendUvarint(b, uint64(v))
	}
	e.bytes(num, b)
}

// packedFloats writes a repeated float field marked packed, or nothing when
// it is empty.
func (e *encoder) packedFloats(num int, vs []float32) {
	if len(vs) == 0 {
		return
	}
	b := make([]byte, 0, 4*len(vs))
	for _, v := range vs {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}
	e.bytes(num, b)
}

// keep writes the fields kept unread.
func (e *encoder) keep(fields []rawField) {
	for _, f := range fields {
		e.add(f.num, f.raw)
	}
}

// encode returns the message.
func (e *encoder) encode() []byte {
	sort.SliceStable(e.entries, func(i, j int) bool { return e.entries[i].num < e.entries[j].num })
	size := 0
	for _, en := range e.entries {
		size += len(en.data)
	}
	out := make([]byte, 0, size)
	for _, en := range e.entries {
		out = append(out, en.data...)
	}
	return out
}
