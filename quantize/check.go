package quantize

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/hyz-is/arandu-onnx/graph"
)

// opsetChanges is, for each operator whose definition changed incompatibly
// between opsets 10 and 13, the opsets where it did. A model using one of
// them at an opset below the change cannot be raised past it by changing the
// number alone. From the ONNX operator changelog: Clip, Pad and Resize took
// their attributes as inputs at 11; Dropout took its ratio as an input at
// 12; Softmax, LogSoftmax and Hardmax changed the meaning of axis, and Split,
// Squeeze, Unsqueeze and ReduceSum took their attributes as inputs, at 13.
// Upsample and Scatter are deprecated and never raised.
var opsetChanges = map[string][]int64{
	"Clip": {11}, "Pad": {11}, "Resize": {11},
	"Dropout":    {12},
	"Softmax":    {13},
	"LogSoftmax": {13},
	"Hardmax":    {13},
	"Split":      {13},
	"Squeeze":    {13},
	"Unsqueeze":  {13},
	"ReduceSum":  {13},
	"Upsample":   {0},
	"Scatter":    {0},
}

// opsetPlan returns the default-domain opset of the model and the one QDQ
// writes, or why it cannot raise it.
func opsetPlan(m *graph.Model) (int64, int64, error) {
	v, ok := m.Opset(graph.DefaultDomain)
	if !ok {
		return 0, 0, fmt.Errorf("%w: the model imports no default-domain opset", ErrRefused)
	}
	if v >= MinimumOpset {
		return v, v, nil
	}
	if v < 10 {
		return v, v, fmt.Errorf("%w: the model is at opset %d; QuantizeLinear needs 10 and per-channel DequantizeLinear %d, and raising a model from below 10 changes too many operators to do by number: raise it with rewrite.UpgradeOpset (arandu-onnx-rewrite -opset 13), or export it again at opset %d or later", ErrRefused, v, MinimumOpset, MinimumOpset)
	}
	if m.HasFunctions() || m.Graph.HasSubgraphs() {
		return v, v, fmt.Errorf("%w: the model is at opset %d and has local functions or subgraphs, which are not read to check that raising it to %d is safe: export it again at opset %d or later", ErrRefused, v, MinimumOpset, MinimumOpset)
	}
	var blocking []string
	seen := map[string]bool{}
	for _, n := range m.Graph.Nodes {
		if !n.IsStandard() || seen[n.OpType] {
			continue
		}
		for _, at := range opsetChanges[n.OpType] {
			if at == 0 || (at > v && at <= MinimumOpset) {
				seen[n.OpType] = true
				blocking = append(blocking, n.OpType)
				break
			}
		}
	}
	if len(blocking) > 0 {
		sort.Strings(blocking)
		return v, v, fmt.Errorf("%w: the model is at opset %d, per-channel DequantizeLinear needs %d, and %s changed meaning in between: raise it with rewrite.UpgradeOpset (arandu-onnx-rewrite -opset 13; a Dropout can go first with -dropout-to-identity), or export it again at opset %d or later", ErrRefused, v, MinimumOpset, strings.Join(blocking, ", "), MinimumOpset)
	}
	return v, MinimumOpset, nil
}

// ErrNotTensorRTQDQ is wrapped by the error CheckTensorRT returns for a model
// whose quantization TensorRT would refuse or not run in INT8.
var ErrNotTensorRTQDQ = errors.New("quantize: not the QDQ form TensorRT runs")

// biasInputs is, for each operator with a bias, the index of that input.
var biasInputs = map[string]int{"Conv": 2, "ConvTranspose": 2, "Gemm": 2}

// CheckTensorRT reports every reason TensorRT's explicit quantization would
// not run the model in INT8, and nil when there is none and the model has at
// least one DequantizeLinear.
//
// The rules are TensorRT 10's for Q/DQ networks: every QuantizeLinear and
// DequantizeLinear of the top-level graph has a constant, positive, finite
// float32 scale and a constant int8 zero point that is 0 (absent, it would
// default to uint8); a per-channel scale has its axis; no DequantizeLinear
// reads an int32 tensor or feeds the bias of a Conv, ConvTranspose or Gemm;
// and no operator of the QOperator or dynamic forms (QLinearConv,
// MatMulInteger, ...) is present. Subgraphs are not read.
func CheckTensorRT(m *graph.Model) error {
	if m == nil || m.Graph == nil {
		return fmt.Errorf("%w: the model has no graph", ErrNotTensorRTQDQ)
	}
	g := m.Graph
	var problems []string
	add := func(format string, args ...any) { problems = append(problems, fmt.Sprintf(format, args...)) }

	count := 0
	perChannel := false
	for _, n := range g.Nodes {
		if !n.IsStandard() {
			continue
		}
		switch n.OpType {
		case "QLinearConv", "QLinearMatMul", "ConvInteger", "MatMulInteger", "DynamicQuantizeLinear":
			add("%s %q is the QOperator or dynamic form, which TensorRT does not run", n.OpType, n.Name)
			continue
		case "QuantizeLinear", "DequantizeLinear":
		default:
			continue
		}
		if n.OpType == "DequantizeLinear" {
			count++
			if x := g.Constant(n.Input(0)); x != nil && x.DataType != graph.Int8 {
				add("DequantizeLinear %q reads %q, which is %s: TensorRT dequantizes int8 only, and a bias stays float", n.Name, x.Name, x.DataType)
			} else if x == nil {
				if p := g.Producer(n.Input(0)); p == nil || p.OpType != "QuantizeLinear" || !p.IsStandard() {
					add("DequantizeLinear %q reads %q, which is neither an int8 constant nor the output of a QuantizeLinear: TensorRT dequantizes int8 only", n.Name, n.Input(0))
				}
			}
		}
		scale := g.Constant(n.Input(1))
		if scale == nil {
			add("%s %q has no constant scale", n.OpType, n.Name)
		} else if values, err := scale.Float32s(); err != nil {
			add("%s %q: scale %q: %v", n.OpType, n.Name, scale.Name, err)
		} else {
			for _, s := range values {
				if !(s > 0) || math.IsInf(float64(s), 0) {
					add("%s %q has the scale %v, and TensorRT takes positive finite scales", n.OpType, n.Name, s)
					break
				}
			}
			if len(values) > 1 {
				perChannel = true
				if x := g.Constant(n.Input(0)); n.OpType == "DequantizeLinear" && x != nil {
					axis := n.Int("axis", 1)
					if axis < 0 {
						axis += int64(len(x.Dims))
					}
					if axis < 0 || axis >= int64(len(x.Dims)) || x.Dims[axis] != int64(len(values)) {
						add("DequantizeLinear %q has %d scales along axis %d of %q, shape %v", n.Name, len(values), n.Int("axis", 1), x.Name, x.Dims)
					}
				}
			}
		}
		zero := n.Input(2)
		if zero == "" {
			add("%s %q has no zero point, so its type defaults to uint8; TensorRT needs an int8 zero point of 0", n.OpType, n.Name)
			continue
		}
		zp := g.Constant(zero)
		switch {
		case zp == nil:
			add("%s %q has no constant zero point", n.OpType, n.Name)
		case zp.DataType != graph.Int8:
			add("%s %q has a %s zero point; TensorRT needs int8", n.OpType, n.Name, zp.DataType)
		default:
			values, err := zp.Int8s()
			if err != nil {
				add("%s %q: zero point %q: %v", n.OpType, n.Name, zp.Name, err)
				break
			}
			for _, v := range values {
				if v != 0 {
					add("%s %q has the zero point %d: TensorRT quantization is symmetric, zero point 0", n.OpType, n.Name, v)
					break
				}
			}
		}
	}
	for _, dq := range g.Nodes {
		if dq.OpType != "DequantizeLinear" || !dq.IsStandard() {
			continue
		}
		for _, use := range biasUses(g, dq.Outputs) {
			add("DequantizeLinear %q feeds the bias of %s; TensorRT takes a float bias and quantizes it itself", dq.Name, use)
		}
	}
	for _, problem := range runtimeBiasQuantization(g) {
		add("%s", problem)
	}
	if v, ok := m.Opset(graph.DefaultDomain); perChannel && (!ok || v < MinimumOpset) {
		add("the model quantizes per channel at opset %d, and per-channel DequantizeLinear is opset %d", v, MinimumOpset)
	}
	if count == 0 && len(problems) == 0 {
		add("the model has no DequantizeLinear: it is not quantized")
	}
	if len(problems) > 0 {
		return fmt.Errorf("%w: %s", ErrNotTensorRTQDQ, strings.Join(problems, "; "))
	}
	return nil
}
