package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/quantize"
)

// runQNN exports the model in the form the Qualcomm QNN execution provider
// runs (quantize.QNN), from the measured ranges and options in the JSON file
// named by -qnn. It is the same pipeline as the symmetric export (read the
// model through its manifest, export, verify on the CPU, write the model, the
// report and the manifest), with quantize.QNN in place of quantize.QDQ and no
// calibration table: the ranges are a minimum and a maximum per tensor.
func runQNN(o options) error {
	data, err := os.ReadFile(o.qnn)
	if err != nil {
		return err
	}
	var opts quantize.QNNOptions
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&opts); err != nil {
		return fmt.Errorf("-qnn %s: %w", o.qnn, err)
	}
	opts.ExcludeNodes = append(opts.ExcludeNodes, splitList(o.excludeNode)...)
	opts.ExcludePrefixes = append(opts.ExcludePrefixes, splitList(o.excludePref)...)
	opts.ExcludeOpTypes = append(opts.ExcludeOpTypes, splitList(o.excludeOp)...)

	source, models, body, err := readModel(o.manifest, o.models)
	if err != nil {
		return err
	}
	if source.Precision != onnx.PrecisionFP32 {
		return fmt.Errorf("%s has precision %s, and the QNN export starts from the float32 model its ranges were measured on", source.ID, source.Precision)
	}
	model, err := graph.Load(body)
	if err != nil {
		return err
	}
	report, err := quantize.QNN(model, opts)
	if err != nil {
		return err
	}
	out, err := model.Marshal()
	if err != nil {
		return err
	}

	derived := source
	derived.Calibration = nil
	derived.Precision = onnx.PrecisionINT8
	derived.Quantization = "" // the QNN form is asymmetric, which the TensorRT form's check refuses
	derived.Inputs = append([]onnx.TensorSpec{}, source.Inputs...)
	for i, spec := range derived.Inputs {
		for _, name := range opts.IntegerInputs {
			if spec.Name == name {
				derived.Inputs[i].ElementType = onnx.Uint8
			}
		}
	}
	if o.id != "" {
		derived.ID = o.id
	}
	derived.SHA256 = sha256Hex(out)
	derived.DerivedFrom = &onnx.Derivation{
		ID: source.ID, SHA256: source.SHA256,
		Transform: fmt.Sprintf("arandu-onnx-quantize: QNN QDQ %s, scales sha256 %s, %d nodes quantized, %d left in float",
			report.Activations, report.ScalesSHA256, len(report.Quantized), len(report.Skipped)),
	}

	tmp := o.out + ".tmp"
	if err := os.WriteFile(tmp, out, 0o644); err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp) }()
	var verification *verificationReport
	switch {
	case len(opts.IntegerInputs) > 0:
		fmt.Fprintln(os.Stderr, "not compared with the float model on synthetic inputs: an integer input takes different tensors than the float model")
	case o.verify > 0:
		check := derived
		check.File = filepath.Base(tmp)
		verification, err = verifyEquivalent(o.verifyOptions(), source, models, check, filepath.Dir(tmp))
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "compared on the CPU over %d synthetic inputs: max |float - quantized| = %g, max |float| = %g\n",
			verification.Inputs, verification.MaxAbsDiff, verification.MaxAbsReference)
		if o.tolerance > 0 && verification.MaxAbsDiff > o.tolerance*verification.MaxAbsReference {
			return fmt.Errorf("the QNN model differs from the float model by %g, more than -tolerance %g of %g: nothing written",
				verification.MaxAbsDiff, o.tolerance, verification.MaxAbsReference)
		}
	}
	if err := os.Rename(tmp, o.out); err != nil {
		return err
	}
	derived.File = relativeTo(models, o.out)
	if !filepath.IsLocal(derived.File) {
		derived.File = filepath.Base(o.out)
		fmt.Fprintf(os.Stderr, "%s is outside %s: the printed manifest names it relative to its own directory\n", o.out, models)
	}
	if err := derived.Validate(); err != nil {
		return err
	}

	full := struct {
		Source       sourceReport        `json:"source"`
		Quantization quantize.QNNReport  `json:"quantization"`
		Verification *verificationReport `json:"verification,omitempty"`
		Output       sourceReport        `json:"output"`
	}{
		Source:       sourceReport{ID: source.ID, File: source.File, SHA256: source.SHA256},
		Quantization: report,
		Verification: verification,
		Output:       sourceReport{ID: derived.ID, File: derived.File, SHA256: derived.SHA256},
	}
	if err := writeJSON(o.report, full); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%d nodes quantized, %d left in float; %d activations, %d weights, %d biases; opset %d -> %d; scales %s\n",
		len(report.Quantized), len(report.Skipped), len(report.Tensors), len(report.Weights), len(report.Biases),
		report.OpsetBefore, report.OpsetAfter, report.ScalesSHA256)
	encoded, err := json.MarshalIndent(derived, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}
