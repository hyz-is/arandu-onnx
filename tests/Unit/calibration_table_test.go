package unit_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/hyz-is/arandu-onnx/calibration"
)

// The formats are the ones ONNX Runtime's TensorRT provider reads
// (core/providers/tensorrt/tensorrt_execution_provider_utils.h,
// ReadDynamicRange, the same in 1.27.0 and 1.30.0). The golden bytes below
// are written out by hand from the FlatBuffers encoding rules, not by running
// the encoder.

func TestATableRoundTripsThroughBothFormats(t *testing.T) {
	t.Parallel()

	table := calibration.Table{
		"images":                1,
		"/model.0/conv/Conv:0":  12.5,
		"/model.1/act/Mul":      3.1415927,
		"a tensor with spaces":  0.0078125,
		"/backbone/stage4/Relu": 6.0221e3,
	}
	encoded, err := table.MarshalONNXRuntime()
	if err != nil {
		t.Fatalf("MarshalONNXRuntime: %v", err)
	}
	if got := calibration.Detect(encoded); got != calibration.FormatONNXRuntime {
		t.Errorf("Detect(FlatBuffers) = %q", got)
	}
	back, err := calibration.Parse(encoded, calibration.FormatONNXRuntime)
	if err != nil {
		t.Fatalf("Parse(onnxruntime): %v", err)
	}
	assertTablesEqual(t, back, table, 0)

	// The ':' in "/model.0/conv/Conv:0" is refused by TensorRT's format, so
	// the round trip there runs on the names it can carry.
	plain := calibration.Table{}
	for name, r := range table {
		if !strings.Contains(name, ":") {
			plain[name] = r
		}
	}
	text, err := plain.MarshalTensorRT("")
	if err != nil {
		t.Fatalf("MarshalTensorRT: %v", err)
	}
	if got := calibration.Detect(text); got != calibration.FormatTensorRT {
		t.Errorf("Detect(TensorRT text) = %q", got)
	}
	back, err = calibration.Parse(text, calibration.FormatTensorRT)
	if err != nil {
		t.Fatalf("Parse(tensorrt): %v", err)
	}
	// range -> range/127 -> *127 in float32 may move by one unit in the last
	// place, which is what ONNX Runtime's own reading does too.
	assertTablesEqual(t, back, plain, 2e-7)
}

func TestTheFlatBuffersEncodingIsTheDocumentedLayout(t *testing.T) {
	t.Parallel()

	le := binary.LittleEndian
	want := make([]byte, 64)
	le.PutUint32(want[0:], 12) // root table at 12
	le.PutUint16(want[4:], 6)  // TrtTable vtable: 6 bytes,
	le.PutUint16(want[6:], 8)  // a table of 8 bytes,
	le.PutUint16(want[8:], 4)  // dict at offset 4; two bytes of padding
	le.PutUint32(want[12:], 8) // soffset back to the vtable at 4
	le.PutUint32(want[16:], 4) // dict vector at 16 + 4 = 20
	le.PutUint32(want[20:], 1) // one entry,
	le.PutUint32(want[24:], 12)
	le.PutUint16(want[28:], 8)  // KeyValue vtable: 8 bytes,
	le.PutUint16(want[30:], 12) // a table of 12 bytes,
	le.PutUint16(want[32:], 4)  // key at 4,
	le.PutUint16(want[34:], 8)  // value at 8
	le.PutUint32(want[36:], 8)  // the entry at 24 + 12 = 36, its vtable at 28
	le.PutUint32(want[40:], 8)  // key string at 40 + 8 = 48
	le.PutUint32(want[44:], 12) // value string at 44 + 12 = 56
	le.PutUint32(want[48:], 1)
	want[52] = 'x' // "x", NUL, padding
	le.PutUint32(want[56:], 1)
	want[60] = '2' // "2", NUL, padding

	got, err := calibration.Table{"x": 2}.MarshalONNXRuntime()
	if err != nil {
		t.Fatalf("MarshalONNXRuntime: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("encoding =\n% x\nwant\n% x", got, want)
	}
}

func TestATableLaidOutBackToFrontIsRead(t *testing.T) {
	t.Parallel()

	// The official FlatBuffers builders write back to front, so a vtable can
	// come after its table, with a negative soffset. This buffer is laid out
	// that way by hand: TrtTable at 8 with its vtable at 48, the dict at 16,
	// the entry at 24 with its vtable at 36, the key at 56, the value at 64.
	le := binary.LittleEndian
	buf := make([]byte, 72)
	le.PutUint32(buf[0:], 8)
	le.PutUint32(buf[8:], negative(8-48))
	le.PutUint32(buf[12:], 4)
	le.PutUint32(buf[16:], 1)
	le.PutUint32(buf[20:], 4) // entry at 24
	le.PutUint32(buf[24:], negative(24-36))
	le.PutUint32(buf[28:], 28) // key at 56
	le.PutUint32(buf[32:], 32) // value at 64
	le.PutUint16(buf[36:], 8)
	le.PutUint16(buf[38:], 12)
	le.PutUint16(buf[40:], 4)
	le.PutUint16(buf[42:], 8)
	le.PutUint16(buf[48:], 6)
	le.PutUint16(buf[50:], 8)
	le.PutUint16(buf[52:], 4)
	le.PutUint32(buf[56:], 1)
	buf[60] = 'k'
	le.PutUint32(buf[64:], 3)
	copy(buf[68:], "0.5")
	got, err := calibration.ParseONNXRuntime(buf)
	if err != nil {
		t.Fatalf("ParseONNXRuntime: %v", err)
	}
	if len(got) != 1 || got["k"] != 0.5 {
		t.Fatalf("ParseONNXRuntime = %v, want k: 0.5", got)
	}

	// The same buffer with the value field absent from the entry's vtable.
	le.PutUint16(buf[42:], 0)
	if _, err := calibration.ParseONNXRuntime(buf); !errors.Is(err, calibration.ErrInvalidTable) || !strings.Contains(err.Error(), "no field 1") {
		t.Fatalf("ParseONNXRuntime = %v, want the missing value reported", err)
	}
}

func TestTheTensorRTTextIsTheDocumentedLayout(t *testing.T) {
	t.Parallel()

	got, err := calibration.Table{"a": 127, "b": 1.27}.MarshalTensorRT("")
	if err != nil {
		t.Fatalf("MarshalTensorRT: %v", err)
	}
	// 127 / 127 = 1.0 is 0x3f800000; 1.27 / 127 = 0.01 is 0x3c23d70a.
	want := "TRT-0-AranduCalibration\na: 3f800000\nb: 3c23d70a\n"
	if string(got) != want {
		t.Fatalf("MarshalTensorRT =\n%s\nwant\n%s", got, want)
	}

	// TensorRT's own caches look like this, with its header and no padding.
	table, err := calibration.ParseTensorRT([]byte("TRT-8601-EntropyCalibration2\ninput: 3c010a14\nconv1: 3d8a1b2c\n"))
	if err != nil {
		t.Fatalf("ParseTensorRT: %v", err)
	}
	if r := table["input"]; math.Abs(float64(r)-float64(math.Float32frombits(0x3c010a14)*127)) > 0 {
		t.Errorf("input range = %v", r)
	}
}

func TestTensorRTTextRefusesWhatONNXRuntimeWouldMisread(t *testing.T) {
	t.Parallel()

	if _, err := (calibration.Table{"op:0": 1}).MarshalTensorRT(""); !errors.Is(err, calibration.ErrInvalidTable) {
		t.Errorf("MarshalTensorRT of a name with ':' = %v, want ErrInvalidTable", err)
	}
	if _, err := (calibration.Table{"a": 1}).MarshalTensorRT("not a header"); !errors.Is(err, calibration.ErrInvalidTable) {
		t.Errorf("MarshalTensorRT with a bad header = %v, want ErrInvalidTable", err)
	}
	for name, body := range map[string]string{
		"no header":     "input: 3c010a14\n",
		"two colons":    "TRT-1-X\nop:0: 3c010a14\n",
		"not hex":       "TRT-1-X\ninput: zz\n",
		"too long":      "TRT-1-X\ninput: 3c010a1400\n",
		"zero scale":    "TRT-1-X\ninput: 00000000\n",
		"named twice":   "TRT-1-X\ninput: 3c010a14\ninput: 3c010a14\n",
		"empty":         "",
		"only a header": "TRT-1-X\n",
	} {
		if _, err := calibration.Parse([]byte(body), calibration.FormatTensorRT); !errors.Is(err, calibration.ErrInvalidTable) {
			t.Errorf("%s: Parse = %v, want ErrInvalidTable", name, err)
		}
	}
}

func TestATableTensorRTCannotTakeIsRefused(t *testing.T) {
	t.Parallel()

	for name, table := range map[string]calibration.Table{
		"empty":        {},
		"no name":      {"": 1},
		"line break":   {"a\nb": 1},
		"zero range":   {"a": 0},
		"negative":     {"a": -1},
		"not a number": {"a": float32(math.NaN())},
		"infinite":     {"a": float32(math.Inf(1))},
	} {
		if _, err := table.MarshalONNXRuntime(); !errors.Is(err, calibration.ErrInvalidTable) {
			t.Errorf("%s: MarshalONNXRuntime = %v, want ErrInvalidTable", name, err)
		}
	}
}

// TestAMalformedFlatBufferIsRefusedAndNeverPanics feeds the reader every
// truncation and every single-byte corruption of a valid table. ONNX Runtime
// reads these files with no verifier, so this reader is the check.
func TestAMalformedFlatBufferIsRefusedAndNeverPanics(t *testing.T) {
	t.Parallel()

	valid, err := calibration.Table{"input": 1, "conv": 2.5, "relu": 6}.MarshalONNXRuntime()
	if err != nil {
		t.Fatalf("MarshalONNXRuntime: %v", err)
	}
	// The last string's NUL is the last byte that matters; what follows it is
	// padding, and a table cut there is still whole.
	last := strings.LastIndexFunc(string(valid), func(r rune) bool { return r != 0 }) + 1
	for n := 0; n <= last; n++ {
		if _, err := calibration.Parse(valid[:n], calibration.FormatONNXRuntime); err == nil {
			t.Errorf("a table truncated to %d of %d bytes was accepted", n, len(valid))
		}
	}
	for i := range valid {
		for _, v := range []byte{0x00, 0x7f, 0xff} {
			corrupt := append([]byte(nil), valid...)
			corrupt[i] = v
			_, _ = calibration.Parse(corrupt, calibration.FormatONNXRuntime)
		}
	}
}

func assertTablesEqual(t *testing.T, got, want calibration.Table, relative float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("table has %d tensors, want %d: %v", len(got), len(want), got)
	}
	for name, w := range want {
		g, ok := got[name]
		if !ok {
			t.Errorf("tensor %q is missing", name)
			continue
		}
		if math.Abs(float64(g-w)) > relative*math.Abs(float64(w)) {
			t.Errorf("tensor %q = %v, want %v", name, g, w)
		}
	}
}

// negative is a negative soffset as the uint32 bits it is stored in.
func negative(v int32) uint32 { return uint32(v) }
