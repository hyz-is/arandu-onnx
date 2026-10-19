// Command arandu-onnx-quantize exports a float32 ONNX model as explicit QDQ
// INT8, the form TensorRT 10 and 11 run, from its calibration table.
//
//	arandu-onnx-quantize -manifest models/det/manifest.json \
//	    -mode conv -exclude-prefix /model/model.22/ \
//	    -out models/det/model.qdq.onnx
//
// The model is read through its manifest and its SHA-256 checked. The table
// is the manifest's calibration entry, its digest checked too, or the file
// named by -table (either format arandu-onnx-calibrate writes). The export
// is quantize.QDQ: symmetric int8, zero points 0, activations per tensor,
// weights per output channel, biases in float. The result is checked with
// quantize.CheckTensorRT and, unless -verify 0, run against the float model
// on the ONNX Runtime CPU provider over synthetic inputs, the largest
// difference reported (and held to -tolerance, a fraction of the largest
// float output, when it is set). The model, the JSON report beside it, and
// the printed manifest (precision int8, quantization qdq, digest included)
// are written only when every check passed.
//
// Synthetic inputs exceed the calibrated ranges in ways real inputs do not,
// so their difference is an upper bound on the rounding, not the model's
// accuracy in INT8. That is measured on real samples, against FP16 on the
// card that will run it.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/calibration"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/quantize"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "arandu-onnx-quantize:", err)
		os.Exit(1)
	}
}

type options struct {
	manifest, models, library string
	table, format             string
	mode                      string
	excludeNode, excludePref  string
	excludeOp                 string
	out, report, id           string
	shape                     string
	verify                    int
	seed                      uint64
	low, high, tolerance      float64
}

func run(args []string) error {
	var o options
	fs := flag.NewFlagSet("arandu-onnx-quantize", flag.ContinueOnError)
	fs.StringVar(&o.manifest, "manifest", "", "the float model's manifest (JSON); required")
	fs.StringVar(&o.models, "models", "", "the model directory the manifest's files are relative to; default: the manifest's directory")
	fs.StringVar(&o.table, "table", "", "the calibration table; default: the manifest's calibration entry")
	fs.StringVar(&o.format, "format", "", "the -table format, onnxruntime or tensorrt; default: detected")
	fs.StringVar(&o.mode, "mode", string(quantize.ModeConv), "conv (Conv, ConvTranspose) or conv-matmul (also MatMul and Gemm with a constant weight)")
	fs.StringVar(&o.excludeNode, "exclude-node", "", "comma-separated node names left in float")
	fs.StringVar(&o.excludePref, "exclude-prefix", "", "comma-separated node name prefixes left in float")
	fs.StringVar(&o.excludeOp, "exclude-op", "", "comma-separated operators left in float")
	fs.StringVar(&o.out, "out", "", "the QDQ model to write; required")
	fs.StringVar(&o.report, "report", "", "the JSON report to write; default: -out with .report.json")
	fs.StringVar(&o.id, "id", "", "the id of the printed manifest; default: the source manifest's")
	fs.IntVar(&o.verify, "verify", 3, "run the float and the QDQ model on this many synthetic inputs and compare; 0 skips")
	fs.Float64Var(&o.tolerance, "tolerance", 0, "the largest difference allowed, as a fraction of the largest |float output|; 0 reports only")
	fs.StringVar(&o.shape, "shape", "", "comma-separated input shape for the synthetic inputs of a model with one input whose shape is dynamic")
	fs.Uint64Var(&o.seed, "seed", 1, "the seed of the synthetic inputs")
	fs.Float64Var(&o.low, "low", 0, "the lowest value of a synthetic input")
	fs.Float64Var(&o.high, "high", 1, "the highest value of a synthetic input")
	fs.StringVar(&o.library, "library", "", "the ONNX Runtime shared library; default: onnx.DefaultSharedLibraryPath()")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if o.manifest == "" || o.out == "" {
		return errors.New("-manifest and -out are required")
	}
	if o.report == "" {
		o.report = strings.TrimSuffix(o.out, filepath.Ext(o.out)) + ".report.json"
	}

	source, models, data, err := readModel(o.manifest, o.models)
	if err != nil {
		return err
	}
	if source.Precision != onnx.PrecisionFP32 {
		return fmt.Errorf("%s has precision %s, and QDQ starts from the float32 model its table was calibrated on", source.ID, source.Precision)
	}
	table, tableReport, err := readTable(o, source, models)
	if err != nil {
		return err
	}

	model, err := graph.Load(data)
	if err != nil {
		return err
	}
	report, err := quantize.QDQ(model, table, quantize.Options{
		Mode:            quantize.Mode(o.mode),
		ExcludeNodes:    splitList(o.excludeNode),
		ExcludePrefixes: splitList(o.excludePref),
		ExcludeOpTypes:  splitList(o.excludeOp),
	})
	if err != nil {
		return err
	}
	if err := quantize.CheckTensorRT(model); err != nil {
		return fmt.Errorf("the export does not pass its own check, which is a defect of quantize.QDQ: %w", err)
	}
	out, err := model.Marshal()
	if err != nil {
		return err
	}

	derived := source
	derived.Calibration = nil
	derived.Precision = onnx.PrecisionINT8
	derived.Quantization = onnx.QuantizationQDQ
	if o.id != "" {
		derived.ID = o.id
	}
	derived.SHA256 = sha256Hex(out)

	tmp := o.out + ".tmp"
	if err := os.WriteFile(tmp, out, 0o644); err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp) }()
	var verification *verificationReport
	if o.verify > 0 {
		check := derived
		check.File = filepath.Base(tmp)
		verification, err = verifyEquivalent(o.verifyOptions(), source, models, check, filepath.Dir(tmp))
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "compared on the CPU over %d synthetic inputs: max |float - int8| = %g, max |float| = %g\n",
			verification.Inputs, verification.MaxAbsDiff, verification.MaxAbsReference)
		if o.tolerance > 0 && verification.MaxAbsDiff > o.tolerance*verification.MaxAbsReference {
			return fmt.Errorf("the QDQ model differs from the float model by %g, more than -tolerance %g of %g: nothing written",
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
		Table        tableSource         `json:"table"`
		Quantization quantize.Report     `json:"quantization"`
		Verification *verificationReport `json:"verification,omitempty"`
		Output       sourceReport        `json:"output"`
	}{
		Source:       sourceReport{ID: source.ID, File: source.File, SHA256: source.SHA256},
		Table:        tableReport,
		Quantization: report,
		Verification: verification,
		Output:       sourceReport{ID: derived.ID, File: derived.File, SHA256: derived.SHA256},
	}
	if err := writeJSON(o.report, full); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%d nodes quantized, %d left in float; %d activations, %d weights; opset %d -> %d; scales %s\n",
		len(report.Quantized), len(report.Skipped), len(report.Activations), len(report.Weights),
		report.OpsetBefore, report.OpsetAfter, report.ScalesSHA256)
	encoded, err := json.MarshalIndent(derived, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}

// verifyOptions is the subset the shared verification reads.
func (o options) verifyOptions() verifyOptions {
	return verifyOptions{library: o.library, shape: o.shape, verify: o.verify, seed: o.seed, low: o.low, high: o.high}
}

// tableSource names the table in the report.
type tableSource struct {
	File    string             `json:"file"`
	SHA256  string             `json:"sha256"`
	Format  calibration.Format `json:"format"`
	Tensors int                `json:"tensors"`
}

// readTable reads the table: -table, or the manifest's calibration entry,
// whose digest is checked.
func readTable(o options, m onnx.Manifest, models string) (calibration.Table, tableSource, error) {
	path, format, pinned := o.table, calibration.Format(o.format), ""
	if path == "" {
		if m.Calibration == nil {
			return nil, tableSource{}, fmt.Errorf("%s has no calibration entry: give -table, or write one with arandu-onnx-calibrate", m.ID)
		}
		path = filepath.Join(models, m.Calibration.File)
		format = m.Calibration.Format
		pinned = m.Calibration.SHA256
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, tableSource{}, err
	}
	sum := sha256Hex(data)
	if pinned != "" && sum != pinned {
		return nil, tableSource{}, fmt.Errorf("%w: the calibration table %s has sha256 %s and the manifest records %s", onnx.ErrDigestMismatch, path, sum, pinned)
	}
	if format == "" {
		format = calibration.Detect(data)
	}
	table, err := calibration.Parse(data, format)
	if err != nil {
		return nil, tableSource{}, err
	}
	return table, tableSource{File: relativeTo(models, path), SHA256: sum, Format: format, Tensors: len(table)}, nil
}
