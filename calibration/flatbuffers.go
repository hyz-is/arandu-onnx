package calibration

import (
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"
)

// The FlatBuffers encoding of ONNX Runtime's CalTableFlatBuffers.TrtTable,
// written and read by hand.
//
// The schema is two tables. KeyValue has field 0 key:string (required) at
// vtable offset 4 and field 1 value:string at offset 6; TrtTable has field 0
// dict:[KeyValue] at offset 4. ONNX Runtime reads the file with GetRoot and
// no Verifier, so a malformed table reaches its native code unchecked; this
// reader checks every offset against the buffer before anything is handed
// over, which is why Runtime.Open parses a table before it lets TensorRT read
// it.
//
// The encoding rules used: little-endian throughout; the buffer starts with a
// uoffset (uint32) to the root table; a table starts with an soffset (int32)
// that is the table's position minus its vtable's; a vtable is uint16 vtable
// size, uint16 table size, then one uint16 offset per field, 0 for absent; a
// uoffset field points forward from its own position; a vector and a string
// are a uint32 length then the elements, and a string ends with a NUL that
// the length does not count.

// maxTableBytes bounds the file this package reads: a table of a large model
// is tens of kilobytes, and a file of hundreds of megabytes is not a table.
const maxTableBytes = 64 << 20

// MarshalONNXRuntime encodes the table as ONNX Runtime's FlatBuffers TrtTable,
// with the keys in byte order and each range written as the shortest decimal
// text that reads back as the same float32.
//
// The layout is front to back, which FlatBuffers allows because every offset
// is relative: the root offset, TrtTable's vtable and table, the dict vector,
// one vtable shared by every KeyValue, the KeyValue tables, then the strings.
func (t Table) MarshalONNXRuntime() ([]byte, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	names := t.Names()
	n := len(names)

	const (
		rootVTable   = 4                  // after the root uoffset
		rootTable    = rootVTable + 6 + 2 // vtable of 6 bytes, padded to 4
		dictVector   = rootTable + 8      // soffset + dict uoffset
		entryVTSize  = 8                  // size, table size, key, value
		entryTblSize = 12                 // soffset + key + value
	)
	entryVTable := dictVector + 4 + 4*n
	firstEntry := entryVTable + entryVTSize
	firstString := firstEntry + entryTblSize*n

	// Each string's position, so the tables can point at them.
	type pair struct{ key, value string }
	pairs := make([]pair, n)
	stringsSize := 0
	for i, name := range names {
		pairs[i] = pair{key: name, value: strconv.FormatFloat(float64(t[name]), 'g', -1, 32)}
		stringsSize += paddedString(len(pairs[i].key)) + paddedString(len(pairs[i].value))
	}
	out := make([]byte, firstString+stringsSize)
	le := binary.LittleEndian

	le.PutUint32(out[0:], uint32(rootTable))
	le.PutUint16(out[rootVTable:], 6)
	le.PutUint16(out[rootVTable+2:], 8)
	le.PutUint16(out[rootVTable+4:], 4)
	le.PutUint32(out[rootTable:], uint32(int32(rootTable-rootVTable)))
	le.PutUint32(out[rootTable+4:], uint32(dictVector-(rootTable+4)))

	le.PutUint32(out[dictVector:], uint32(n))
	le.PutUint16(out[entryVTable:], entryVTSize)
	le.PutUint16(out[entryVTable+2:], entryTblSize)
	le.PutUint16(out[entryVTable+4:], 4)
	le.PutUint16(out[entryVTable+6:], 8)

	at := firstString
	for i, p := range pairs {
		element := dictVector + 4 + 4*i
		entry := firstEntry + entryTblSize*i
		le.PutUint32(out[element:], uint32(entry-element))
		le.PutUint32(out[entry:], uint32(int32(entry-entryVTable)))

		le.PutUint32(out[entry+4:], uint32(at-(entry+4)))
		at = putString(out, at, p.key)
		le.PutUint32(out[entry+8:], uint32(at-(entry+8)))
		at = putString(out, at, p.value)
	}
	return out, nil
}

// paddedString is the bytes a string takes: its length, its bytes, the NUL,
// and padding to a multiple of four.
func paddedString(length int) int { return (4 + length + 1 + 3) &^ 3 }

// putString writes a string at a position and returns the position after it.
func putString(out []byte, at int, s string) int {
	binary.LittleEndian.PutUint32(out[at:], uint32(len(s)))
	copy(out[at+4:], s)
	return at + paddedString(len(s))
}

// ParseONNXRuntime decodes ONNX Runtime's FlatBuffers TrtTable, checking every
// offset, and reads each value the way ONNX Runtime does, as a decimal float.
func ParseONNXRuntime(data []byte) (Table, error) {
	if len(data) > maxTableBytes {
		return nil, fmt.Errorf("%w: %d bytes is larger than any calibration table", ErrInvalidTable, len(data))
	}
	buf := flatBuffer(data)
	root, err := buf.uoffset(0)
	if err != nil {
		return nil, err
	}
	dictField, ok, err := buf.field(root, 0)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("%w: the TrtTable has no dict", ErrInvalidTable)
	}
	dict, err := buf.uoffset(dictField)
	if err != nil {
		return nil, err
	}
	count, err := buf.u32(dict)
	if err != nil {
		return nil, err
	}
	if uint64(count)*4 > uint64(len(data)) {
		return nil, fmt.Errorf("%w: the dict claims %d entries in %d bytes", ErrInvalidTable, count, len(data))
	}
	t := make(Table, count)
	for i := range int(count) {
		element := dict + 4 + 4*i
		entry, err := buf.uoffset(element)
		if err != nil {
			return nil, err
		}
		key, err := buf.stringField(entry, 0)
		if err != nil {
			return nil, fmt.Errorf("%w (entry %d key)", err, i)
		}
		value, err := buf.stringField(entry, 1)
		if err != nil {
			return nil, fmt.Errorf("%w (entry %d value)", err, i)
		}
		// std::stof reads a leading float and ignores what follows; a value
		// that is not wholly a number is refused here instead.
		r, err := strconv.ParseFloat(strings.TrimSpace(value), 32)
		if err != nil {
			return nil, fmt.Errorf("%w: tensor %q has range %q, which is not a float32 number", ErrInvalidTable, key, value)
		}
		if _, dup := t[key]; dup {
			return nil, fmt.Errorf("%w: tensor %q appears twice", ErrInvalidTable, key)
		}
		t[key] = float32(r)
	}
	return t, nil
}

// flatBuffer is a FlatBuffers buffer read with bounds checks.
type flatBuffer []byte

func (b flatBuffer) fail(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalidTable, fmt.Sprintf(format, args...))
}

func (b flatBuffer) u16(at int) (int, error) {
	if at < 0 || at+2 > len(b) {
		return 0, b.fail("a uint16 at byte %d is outside the %d-byte table", at, len(b))
	}
	return int(binary.LittleEndian.Uint16(b[at:])), nil
}

func (b flatBuffer) u32(at int) (uint32, error) {
	if at < 0 || at+4 > len(b) {
		return 0, b.fail("a uint32 at byte %d is outside the %d-byte table", at, len(b))
	}
	return binary.LittleEndian.Uint32(b[at:]), nil
}

// uoffset follows the forward offset stored at a position.
func (b flatBuffer) uoffset(at int) (int, error) {
	off, err := b.u32(at)
	if err != nil {
		return 0, err
	}
	target := uint64(at) + uint64(off)
	if target >= uint64(len(b)) {
		return 0, b.fail("the offset at byte %d points to byte %d, outside the %d-byte table", at, target, len(b))
	}
	return int(target), nil
}

// field returns the position of a table's field, and whether it is present.
func (b flatBuffer) field(table, index int) (int, bool, error) {
	soffset, err := b.u32(table)
	if err != nil {
		return 0, false, err
	}
	vtable := int64(table) - int64(int32(soffset))
	if vtable < 0 || vtable+4 > int64(len(b)) {
		return 0, false, b.fail("the table at byte %d has its vtable outside the buffer", table)
	}
	vtSize, err := b.u16(int(vtable))
	if err != nil {
		return 0, false, err
	}
	tableSize, err := b.u16(int(vtable) + 2)
	if err != nil {
		return 0, false, err
	}
	if vtSize < 4 || vtSize%2 != 0 || int(vtable)+vtSize > len(b) || table+tableSize > len(b) {
		return 0, false, b.fail("the vtable at byte %d does not fit the buffer", vtable)
	}
	slot := 4 + 2*index
	if slot+2 > vtSize {
		return 0, false, nil
	}
	off, err := b.u16(int(vtable) + slot)
	if err != nil {
		return 0, false, err
	}
	if off == 0 {
		return 0, false, nil
	}
	if off+4 > tableSize {
		return 0, false, b.fail("field %d of the table at byte %d lies outside the table", index, table)
	}
	return table + off, true, nil
}

// stringField reads a required string field of a table.
func (b flatBuffer) stringField(table, index int) (string, error) {
	at, ok, err := b.field(table, index)
	if err != nil {
		return "", err
	}
	if !ok {
		return "", b.fail("the table at byte %d has no field %d", table, index)
	}
	start, err := b.uoffset(at)
	if err != nil {
		return "", err
	}
	length, err := b.u32(start)
	if err != nil {
		return "", err
	}
	end := uint64(start) + 4 + uint64(length)
	if end >= uint64(len(b)) || b[end] != 0 {
		return "", b.fail("the string at byte %d is not NUL-terminated inside the buffer", start)
	}
	return string(b[start+4 : end]), nil
}
