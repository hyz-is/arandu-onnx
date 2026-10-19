// Command arandu-onnx-rewrite rewrites the Col2Im nodes of an ONNX model into
// a Reshape and a grouped ConvTranspose, so that TensorRT and the CUDA
// provider, which have no Col2Im kernel, run the whole graph.
//
//	arandu-onnx-rewrite -manifest models/age/manifest.json \
//	    -out models/age/mivolo_v2_age.convtranspose.onnx
//
// The model is read through its manifest and its SHA-256 checked before a
// byte of it is decoded. The rewrite (rewrite.Col2ImToConvTranspose) is all
// or nothing. The rewritten model is then verified: the original and the
// rewritten model run on the ONNX Runtime CPU provider over -verify
// synthetic inputs, and the command fails, writing nothing, when an output
// differs by more than -tolerance. Only then is the file written, the JSON
// report beside it, and the manifest of the rewritten file printed, digest
// included.
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
	fs.StringVar(&o.channels, "channels", "", "comma-separated tensor=size: dimension 1 of a Col2Im input the graph does not record")
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
	report, err := rewrite.Col2ImToConvTranspose(model, rewrite.Col2ImOptions{InputChannels: channels})
	if err != nil {
		return err
	}
	if len(report.Rewritten) == 0 {
		return fmt.Errorf("%s has no Col2Im node: nothing to rewrite", source.ID)
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
		Source       sourceReport         `json:"source"`
		Rewrite      rewrite.Col2ImReport `json:"rewrite"`
		OpsBefore    []string             `json:"ops_before"`
		OpsAfter     []string             `json:"ops_after"`
		Verification *verificationReport  `json:"verification,omitempty"`
		Output       sourceReport         `json:"output"`
	}{
		Source:       sourceReport{ID: source.ID, File: source.File, SHA256: source.SHA256},
		Rewrite:      report,
		OpsBefore:    before,
		OpsAfter:     model.Graph.SortedOpCounts(),
		Verification: verification,
		Output:       sourceReport{ID: derived.ID, File: derived.File, SHA256: derived.SHA256},
	}
	if err := writeJSON(o.report, full); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%d Col2Im nodes rewritten; %d nodes before, %d after\n",
		len(report.Rewritten), report.NodesBefore, report.NodesAfter)
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
