package feature_test

import (
	"errors"
	"math"
	"testing"

	onnx "github.com/hyz-is/arandu-onnx"
)

// requantModel is q = Quantize(Dequantize(u)): u is uint8 [1, 4] with scale
// 0.5 and zero point 2, and q is uint8 [1, 4] with scale 0.25 and zero point
// 128, so q = (u - 2) * 2 + 128.
func requantModel() []byte {
	scalar := func(name string, v float32) message {
		return floatInitializer(name, []float32{v})
	}
	zero := func(name string, v byte) message {
		return message{}.integer(2, protoUint8).text(8, name).bytes(9, []byte{v})
	}
	graph := message{}.
		sub(1, node("DequantizeLinear", []string{"u", "s_in", "z_in"}, []string{"f"})).
		sub(1, node("QuantizeLinear", []string{"f", "s_out", "z_out"}, []string{"q"})).
		text(2, "requant").
		sub(5, scalar("s_in", 0.5)).sub(5, zero("z_in", 2)).
		sub(5, scalar("s_out", 0.25)).sub(5, zero("z_out", 128)).
		sub(11, valueInfo("u", protoUint8, 1, 4)).
		sub(12, valueInfo("q", protoUint8, 1, 4))
	return model(graph)
}

func requantManifest() onnx.Manifest {
	m := affineManifest()
	m.ID, m.File = "test.requant", "requant.onnx"
	m.Inputs = []onnx.TensorSpec{{Name: "u", ElementType: onnx.Uint8, Shape: []int64{1, 4}}}
	m.Outputs = []onnx.TensorSpec{{Name: "q", ElementType: onnx.Uint8, Shape: []int64{1, 4}}}
	m.Precision = onnx.PrecisionINT8
	return m
}

func TestABoundSessionTakesAndReturnsUint8AndStatesItsScales(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, requantManifest(), requantModel())
	bound, err := runtime.OpenBound(manifest, dir)
	if err != nil {
		t.Fatalf("opening: %v", err)
	}
	defer bound.Close()

	copy(bound.InputUint8("u"), []uint8{2, 4, 10, 0})
	if err := bound.Run(); err != nil {
		t.Fatalf("running: %v", err)
	}
	got := bound.OutputUint8("q")
	for i, want := range []uint8{128, 132, 144, 124} {
		if got[i] != want {
			t.Errorf("q[%d] = %d, want %d", i, got[i], want)
		}
	}

	in, err := bound.InputQuantization("u")
	if err != nil || in.Scale != 0.5 || in.ZeroPoint != 2 {
		t.Errorf("InputQuantization = %+v, %v, want scale 0.5 and zero point 2", in, err)
	}
	out, err := bound.OutputQuantization("q")
	if err != nil || out.Scale != 0.25 || out.ZeroPoint != 128 {
		t.Errorf("OutputQuantization = %+v, %v, want scale 0.25 and zero point 128", out, err)
	}
	if v := out.Dequantize(144); math.Abs(float64(v)-4) > 1e-6 {
		t.Errorf("Dequantize(144) = %v, want 4", v)
	}
}

func TestQuantizationIsRefusedWhereTheGraphStatesNone(t *testing.T) {
	runtime := newRuntime(t, onnx.Config{})
	manifest, dir := writeModel(t, affineManifest(), affineModel())
	bound, err := runtime.OpenBound(manifest, dir)
	if err != nil {
		t.Fatalf("opening: %v", err)
	}
	defer bound.Close()
	if _, err := bound.OutputQuantization("y"); !errors.Is(err, onnx.ErrNoQuantization) {
		t.Errorf("OutputQuantization of a float output = %v, want ErrNoQuantization", err)
	}
}
