// Command arandu-onnx-rewrite rewrites an ONNX model so that it computes the
// same with other operators: the Col2Im nodes into a Reshape and a grouped
// ConvTranspose, so that TensorRT and the CUDA provider, which have no Col2Im
// kernel, run the whole graph; an old export up to opset 13, which QDQ
// quantization needs; the initializers an old exporter lists as graph inputs
// into constants; and Dropout into Identity.
//
//	arandu-onnx-rewrite -manifest models/example/manifest.json \
//	    -out models/example/model.convtranspose.onnx
//	arandu-onnx-rewrite -manifest old/manifest.json -opset 13 -shape 1,3,48,320 \
//	    -tolerance 1e-5 -out old/model.opset13.onnx
//	arandu-onnx-rewrite -manifest ir3/manifest.json -initializers-out-of-inputs \
//	    -dropout-to-identity -opset 13 -out ir3/model.inference.onnx
//
// The model is read through its manifest and its SHA-256 checked before a
// byte of it is decoded. The rewrites run in this order: initializers out of
// the inputs (rewrite.InitializersOutOfInputs), Dropout to Identity
// (rewrite.DropoutToIdentity), the opset upgrade (rewrite.UpgradeOpset), and
// Col2Im (rewrite.Col2ImToConvTranspose, on by default, a no-op without
// Col2Im). Each is all or nothing, and the command fails when none of them
// changed anything. The rewritten model is then verified: the original and
// the rewritten model run on the ONNX Runtime CPU provider over -verify
// synthetic inputs, and the command fails, writing nothing, when an output
// differs by more than -tolerance. Only then is the file written, the JSON
// report beside it, and the manifest of the rewritten file printed, digest
// and derived_from included.
//
// Synthetic inputs prove the two graphs compute the same function, which is
// what a rewrite has to keep; they say nothing about the model's accuracy,
// which the rewrite does not change.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
	"github.com/hyz-is/arandu-onnx/rewrite"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "arandu-onnx-rewrite:", err)
		os.Exit(1)
	}
}

type options struct {
	manifest, models, library string
	out, report, id           string
	channels, shape           string
	col2im, initializers      bool
	dropout                   bool
	opset                     int64
	verify                    int
	seed                      uint64
	low, high, tolerance      float64
}

func run(args []string) error {
	var o options
	fs := flag.NewFlagSet("arandu-onnx-rewrite", flag.ContinueOnError)
	fs.StringVar(&o.manifest, "manifest", "", "the model's manifest (JSON); required")
	fs.StringVar(&o.models, "models", "", "the model directory the manifest's file is relative to; default: the manifest's directory")
	fs.StringVar(&o.out, "out", "", "the rewritten model to write; required")
	fs.StringVar(&o.report, "report", "", "the JSON report to write; default: -out with .report.json")
	fs.StringVar(&o.id, "id", "", "the id of the printed manifest; default: the source manifest's")
	fs.BoolVar(&o.col2im, "col2im", true, "rewrite every Col2Im node into Reshape and a grouped ConvTranspose (nothing happens without one)")
	fs.StringVar(&o.channels, "channels", "", "comma-separated tensor=size: dimension 1 of a Col2Im input the graph does not record")
	fs.Int64Var(&o.opset, "opset", 0, "raise the default-domain opset to this one, rewriting the operators that changed (13, the only target); 0 leaves it")
	fs.BoolVar(&o.initializers, "initializers-out-of-inputs", false, "drop the graph inputs that have an initializer of the same name, so the weights are constants")
	fs.BoolVar(&o.dropout, "dropout-to-identity", false, "replace every Dropout with an Identity, which is what it computes in inference")
	fs.IntVar(&o.verify, "verify", 3, "run the original and the rewritten model on this many synthetic inputs and compare; 0 skips")
	fs.Float64Var(&o.tolerance, "tolerance", 1e-4, "the largest absolute difference an output may show")
	fs.StringVar(&o.shape, "shape", "", "comma-separated input shape for the synthetic inputs of a model with one input whose shape is dynamic")
	fs.Uint64Var(&o.seed, "seed", 1, "the seed of the synthetic inputs")
	fs.Float64Var(&o.low, "low", -1, "the lowest value of a synthetic input")
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
	channels, err := parseChannels(o.channels)
	if err != nil {
		return err
	}

	source, models, data, err := readModel(o.manifest, o.models)
	if err != nil {
		return err
	}
	model, err := graph.Load(data)
	if err != nil {
		return err
	}
	before := model.Graph.SortedOpCounts()
	done, steps, err := applyRewrites(model, o, channels)
	if err != nil {
		return err
	}
	if len(steps) == 0 {
		return fmt.Errorf("%s: none of the rewrites asked for changed anything (it has no Col2Im node; see -opset, -initializers-out-of-inputs and -dropout-to-identity)", source.ID)
	}
	out, err := model.Marshal()
	if err != nil {
		return err
	}

	derived := source
	derived.Calibration = nil
	if o.id != "" {
		derived.ID = o.id
	}
	derived.SHA256 = sha256Hex(out)
	derived.DerivedFrom = &onnx.Derivation{
		ID: source.ID, SHA256: source.SHA256,
		Transform: "arandu-onnx-rewrite: " + strings.Join(steps, "; "),
	}

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
		fmt.Fprintf(os.Stderr, "verified on the CPU over %d synthetic inputs: max |original - rewritten| = %g (max |original| = %g)\n",
			verification.Inputs, verification.MaxAbsDiff, verification.MaxAbsReference)
		if verification.MaxAbsDiff > o.tolerance {
			return fmt.Errorf("the rewritten model differs from the original by %g, more than -tolerance %g: nothing written", verification.MaxAbsDiff, o.tolerance)
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
		Source       sourceReport                `json:"source"`
		Initializers *rewrite.InitializersReport `json:"initializers,omitempty"`
		Dropout      *rewrite.DropoutReport      `json:"dropout,omitempty"`
		Opset        *rewrite.OpsetReport        `json:"opset,omitempty"`
		Rewrite      *rewrite.Col2ImReport       `json:"rewrite,omitempty"`
		OpsBefore    []string                    `json:"ops_before"`
		OpsAfter     []string                    `json:"ops_after"`
		Verification *verificationReport         `json:"verification,omitempty"`
		Output       sourceReport                `json:"output"`
	}{
		Source:       sourceReport{ID: source.ID, File: source.File, SHA256: source.SHA256},
		Initializers: done.initializers,
		Dropout:      done.dropout,
		Opset:        done.opset,
		Rewrite:      done.col2im,
		OpsBefore:    before,
		OpsAfter:     model.Graph.SortedOpCounts(),
		Verification: verification,
		Output:       sourceReport{ID: derived.ID, File: derived.File, SHA256: derived.SHA256},
	}
	if err := writeJSON(o.report, full); err != nil {
		return err
	}
	for _, step := range steps {
		fmt.Fprintln(os.Stderr, step)
	}
	if source.Calibration != nil {
		fmt.Fprintln(os.Stderr, "the source manifest's calibration table is left out: it pins the original file; calibrate the rewritten model again")
	}
	encoded, err := json.MarshalIndent(derived, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}

// applied is the report of each rewrite that changed the model.
type applied struct {
	initializers *rewrite.InitializersReport
	dropout      *rewrite.DropoutReport
	opset        *rewrite.OpsetReport
	col2im       *rewrite.Col2ImReport
}

// applyRewrites runs the rewrites asked for, in order, and returns what each
// one that changed something did, and a line per change.
func applyRewrites(model *graph.Model, o options, channels map[string]int64) (applied, []string, error) {
	var a applied
	var steps []string
	if o.initializers {
		r, err := rewrite.InitializersOutOfInputs(model)
		if err != nil {
			return a, nil, err
		}
		if len(r.Removed) > 0 {
			a.initializers = &r
			steps = append(steps, fmt.Sprintf("%d initializers taken out of the graph inputs (inputs %d to %d, IR %d to %d)",
				len(r.Removed), r.InputsBefore, r.InputsAfter, r.IRBefore, r.IRAfter))
		}
	}
	if o.dropout {
		r, err := rewrite.DropoutToIdentity(model)
		if err != nil {
			return a, nil, err
		}
		if len(r.Replaced) > 0 {
			a.dropout = &r
			steps = append(steps, fmt.Sprintf("%d Dropout nodes replaced with Identity", len(r.Replaced)))
		}
	}
	if o.opset != 0 {
		r, err := rewrite.UpgradeOpset(model, rewrite.OpsetOptions{Target: o.opset})
		if err != nil {
			return a, nil, err
		}
		if r.OpsetAfter != r.OpsetBefore {
			a.opset = &r
			steps = append(steps, fmt.Sprintf("opset %d to %d, %d nodes converted (IR %d to %d)",
				r.OpsetBefore, r.OpsetAfter, len(r.Converted), r.IRBefore, r.IRAfter))
		}
	}
	if o.col2im {
		r, err := rewrite.Col2ImToConvTranspose(model, rewrite.Col2ImOptions{InputChannels: channels})
		if err != nil {
			return a, nil, err
		}
		if len(r.Rewritten) > 0 {
			a.col2im = &r
			steps = append(steps, fmt.Sprintf("%d Col2Im nodes rewritten; %d nodes before, %d after",
				len(r.Rewritten), r.NodesBefore, r.NodesAfter))
		}
	}
	return a, steps, nil
}

// verifyOptions is the subset the shared verification reads.
func (o options) verifyOptions() verifyOptions {
	return verifyOptions{library: o.library, shape: o.shape, verify: o.verify, seed: o.seed, low: o.low, high: o.high}
}

func parseChannels(value string) (map[string]int64, error) {
	out := map[string]int64{}
	for _, part := range splitList(value) {
		name, size, ok := strings.Cut(part, "=")
		n, err := strconv.ParseInt(size, 10, 64)
		if !ok || name == "" || err != nil || n <= 0 {
			return nil, fmt.Errorf("-channels %q has to be tensor=size pairs", value)
		}
		out[name] = n
	}
	return out, nil
}
