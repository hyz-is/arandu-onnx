// Command arandu-onnx-calibrate writes the INT8 calibration table of a float32
// ONNX model, for TensorRT to run it in INT8.
//
// It runs the model, verified against its manifest, over a set of samples on
// the CPU with onnx.Runtime.Calibrate, writes the table, and prints the
// "calibration" entry to add to the manifest, digest included.
//
//	arandu-onnx-calibrate -manifest models/det/manifest.json \
//	    -images /data/calibration/det -size 640x640 -letterbox \
//	    -mean 0,0,0 -std 1,1,1 \
//	    -exclude-prefix /model/model.22/ -exclude boxes,scores \
//	    -out models/det/det.calibration.flatbuffers
//
// Samples come from one of three places:
//
//   - -images: a directory of JPEG and PNG files, each decoded upright,
//     resized (or letterboxed) to -size and laid out as NCHW with -mean and
//     -std, for a model with one float32 input. This is the vision engines'
//     common preprocessing; an engine whose preprocessing differs calls
//     Runtime.Calibrate from Go with its own.
//   - -raw: a directory of .f32 files, each the little-endian float32 data of
//     one input of shape -shape, already preprocessed.
//   - -synthetic: that many deterministic pseudo-random inputs of shape
//     -shape, uniform between -low and -high. A table calibrated on them runs and
//     reads wrong; they exist for tests and smoke runs, never for a model in
//     production.
//
// The samples decide the table, and for the KUAA engines they are faces and
// identity documents: personal data. Run the command where the samples are
// allowed to be, keep them out of every repository, and describe them in the
// manifest's source without naming anybody.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image/color"
	"iter"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/calibration"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "arandu-onnx-calibrate:", err)
		os.Exit(1)
	}
}

// options is the command line.
type options struct {
	manifest, models, library     string
	method                        string
	percentile                    float64
	bins                          int
	exclude, excludePrefix        string
	format, out, header           string
	images, raw                   string
	synthetic                     int
	input, shape, size, mean, std string
	letterbox, bgr                bool
	low, high                     float64
	seed                          uint64
	limit                         int
}

func run(args []string) error {
	var o options
	fs := flag.NewFlagSet("arandu-onnx-calibrate", flag.ContinueOnError)
	fs.StringVar(&o.manifest, "manifest", "", "the model's manifest (JSON); required")
	fs.StringVar(&o.models, "models", "", "the model directory the manifest's file is relative to; default: the manifest's directory")
	fs.StringVar(&o.library, "library", "", "the ONNX Runtime shared library; default: onnx.DefaultSharedLibraryPath()")
	fs.StringVar(&o.method, "method", string(calibration.MethodMinMax), "minmax, percentile or entropy")
	fs.Float64Var(&o.percentile, "percentile", calibration.DefaultPercentile, "the percentile of |x| the percentile method keeps")
	fs.IntVar(&o.bins, "bins", calibration.DefaultBins, "the histogram bins of the percentile and entropy methods")
	fs.StringVar(&o.exclude, "exclude", "", "comma-separated tensor names to leave out of the table")
	fs.StringVar(&o.excludePrefix, "exclude-prefix", "", "comma-separated tensor name prefixes to leave out of the table")
	fs.StringVar(&o.format, "format", string(calibration.FormatONNXRuntime), "the table format: onnxruntime (FlatBuffers) or tensorrt (calibration cache text)")
	fs.StringVar(&o.out, "out", "", "the table file to write; required")
	fs.StringVar(&o.header, "header", "", "the first line of a tensorrt table; default: "+calibration.DefaultTensorRTHeader)
	fs.StringVar(&o.images, "images", "", "a directory of JPEG and PNG samples")
	fs.StringVar(&o.raw, "raw", "", "a directory of .f32 samples (little-endian float32, shape -shape)")
	fs.IntVar(&o.synthetic, "synthetic", 0, "this many pseudo-random samples of shape -shape (tests and smoke runs only)")
	fs.StringVar(&o.input, "input", "", "the float32 input the samples feed; default: the model's only float32 input")
	fs.StringVar(&o.shape, "shape", "", "comma-separated input shape for -raw and -synthetic; default: the manifest's, which must then be fixed")
	fs.StringVar(&o.size, "size", "", "WIDTHxHEIGHT the images are brought to; default: the manifest's fixed input shape")
	fs.BoolVar(&o.letterbox, "letterbox", false, "letterbox the images (keep the aspect ratio, pad with 114) instead of stretching them")
	fs.BoolVar(&o.bgr, "bgr", false, "lay the image channels out blue, green, red")
	fs.StringVar(&o.mean, "mean", "0,0,0", "per-channel mean on the 0..1 scale")
	fs.StringVar(&o.std, "std", "1,1,1", "per-channel standard deviation on the 0..1 scale")
	fs.Float64Var(&o.low, "low", 0, "the lowest value of a synthetic sample")
	fs.Float64Var(&o.high, "high", 1, "the highest value of a synthetic sample")
	fs.Uint64Var(&o.seed, "seed", 1, "the seed of the synthetic samples")
	fs.IntVar(&o.limit, "limit", 0, "read at most this many samples; 0 reads them all")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if o.manifest == "" || o.out == "" {
		return errors.New("-manifest and -out are required")
	}
	sources := 0
	for _, set := range []bool{o.images != "", o.raw != "", o.synthetic > 0} {
		if set {
			sources++
		}
	}
	if sources != 1 {
		return errors.New("name exactly one of -images, -raw and -synthetic")
	}
	format := calibration.Format(o.format)
	if !format.Valid() {
		return fmt.Errorf("-format %q is not onnxruntime or tensorrt", o.format)
	}

	body, err := os.ReadFile(o.manifest)
	if err != nil {
		return err
	}
	manifest, err := onnx.LoadManifest(body)
	if err != nil {
		return err
	}
	if o.models == "" {
		o.models = filepath.Dir(o.manifest)
	}
	input, err := floatInput(manifest, o.input)
	if err != nil {
		return err
	}
	samples, describe, err := sampleSource(o, input)
	if err != nil {
		return err
	}

	runtime, err := onnx.NewRuntime(onnx.Config{SharedLibraryPath: o.library})
	if err != nil {
		return err
	}
	if err := runtime.Init(); err != nil {
		return err
	}
	defer func() { _ = runtime.Close() }()

	result, err := runtime.Calibrate(context.Background(), manifest, o.models, samples, calibration.Options{
		Method:          calibration.Method(o.method),
		Percentile:      o.percentile,
		Bins:            o.bins,
		Exclude:         splitList(o.exclude),
		ExcludePrefixes: splitList(o.excludePrefix),
	})
	if err != nil {
		return err
	}

	var table []byte
	if format == calibration.FormatTensorRT {
		table, err = result.Table.MarshalTensorRT(o.header)
	} else {
		table, err = result.Table.MarshalONNXRuntime()
	}
	if err != nil {
		return err
	}
	if err := writeFile(o.out, table); err != nil {
		return err
	}

	sum := sha256Hex(table)
	file := o.out
	if rel, err := filepath.Rel(o.models, o.out); err == nil && filepath.IsLocal(rel) {
		file = filepath.ToSlash(rel)
	}
	entry := onnx.Calibration{
		File:      file,
		SHA256:    sum,
		Format:    format,
		Precision: onnx.PrecisionINT8,
		Method:    onnx.CalibrationMethod(result.Report.Method),
		Source: fmt.Sprintf("arandu-onnx-calibrate, onnxruntime %s, %s over %d samples (%s)",
			runtime.Version(), result.Report.Method, result.Samples, describe),
	}
	encoded, err := json.MarshalIndent(map[string]onnx.Calibration{"calibration": entry}, "", "  ")
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%d tensors calibrated, %d excluded, %d not float32, %d always zero, over %d samples\n",
		len(result.Report.Calibrated), len(result.Excluded), len(result.NotFloat), len(result.Report.Zero), result.Samples)
	for _, name := range sortedKeys(result.Report.NonFinite) {
		fmt.Fprintf(os.Stderr, "warning: %s had %d NaN or infinite values, left out of its range\n", name, result.Report.NonFinite[name])
	}
	fmt.Println(string(encoded))
	return nil
}

// floatInput is the float32 input the samples feed.
func floatInput(m onnx.Manifest, name string) (onnx.TensorSpec, error) {
	if len(m.Inputs) != 1 {
		return onnx.TensorSpec{}, fmt.Errorf("%s has %d inputs, and this command feeds a model with one; call Runtime.Calibrate from Go for the others", m.ID, len(m.Inputs))
	}
	var found []onnx.TensorSpec
	for _, spec := range m.Inputs {
		if spec.ElementType == onnx.Float32 && (name == "" || spec.Name == name) {
			found = append(found, spec)
		}
	}
	if len(found) != 1 {
		return onnx.TensorSpec{}, fmt.Errorf("%s has no float32 input %q", m.ID, name)
	}
	return found[0], nil
}

// sampleSource returns the samples, replayable, and a description of them for
// the manifest's source.
func sampleSource(o options, input onnx.TensorSpec) (iter.Seq2[map[string]onnx.Tensor, error], string, error) {
	switch {
	case o.images != "":
		return imageSamples(o, input)
	case o.raw != "":
		shape, err := sampleShape(o.shape, input)
		if err != nil {
			return nil, "", err
		}
		files, err := listFiles(o.raw, ".f32")
		if err != nil {
			return nil, "", err
		}
		files = limit(files, o.limit)
		return fileSamples(files, func(path string) (onnx.Tensor, error) { return rawTensor(path, shape) }, input.Name),
			fmt.Sprintf("%d raw float32 tensors of shape %v", len(files), shape), nil
	default:
		shape, err := sampleShape(o.shape, input)
		if err != nil {
			return nil, "", err
		}
		count := o.synthetic
		if o.limit > 0 {
			count = min(count, o.limit)
		}
		return syntheticSamples(input.Name, shape, count, o.seed, float32(o.low), float32(o.high)),
			fmt.Sprintf("%d synthetic uniform samples in [%v, %v], seed %d: not representative", count, o.low, o.high, o.seed), nil
	}
}

func imageSamples(o options, input onnx.TensorSpec) (iter.Seq2[map[string]onnx.Tensor, error], string, error) {
	width, height, err := imageSize(o.size, input)
	if err != nil {
		return nil, "", err
	}
	mean, err := triple(o.mean)
	if err != nil {
		return nil, "", fmt.Errorf("-mean: %w", err)
	}
	std, err := triple(o.std)
	if err != nil {
		return nil, "", fmt.Errorf("-std: %w", err)
	}
	files, err := listFiles(o.images, ".jpg", ".jpeg", ".png")
	if err != nil {
		return nil, "", err
	}
	files = limit(files, o.limit)
	norm := onnx.Normalization{Mean: mean, Std: std, BGR: o.bgr}
	load := func(path string) (onnx.Tensor, error) {
		body, err := os.ReadFile(path)
		if err != nil {
			return onnx.Tensor{}, err
		}
		img, err := onnx.DecodeImage(bytes.NewReader(body))
		if err != nil {
			return onnx.Tensor{}, fmt.Errorf("%s: %w", filepath.Base(path), err)
		}
		if o.letterbox {
			boxed, _, err := onnx.Letterbox(img, width, height, color.RGBA{114, 114, 114, 255})
			if err != nil {
				return onnx.Tensor{}, err
			}
			return onnx.NCHW(boxed, norm)
		}
		resized, err := onnx.Resize(img, width, height)
		if err != nil {
			return onnx.Tensor{}, err
		}
		return onnx.NCHW(resized, norm)
	}
	mode := "resized"
	if o.letterbox {
		mode = "letterboxed"
	}
	return fileSamples(files, load, input.Name),
		fmt.Sprintf("%d images %s to %dx%d, mean %s, std %s", len(files), mode, width, height, o.mean, o.std), nil
}

// fileSamples reads each file when the pass reaches it, so the samples are
// never all in memory and every pass reads the same ones.
func fileSamples(files []string, load func(string) (onnx.Tensor, error), name string) iter.Seq2[map[string]onnx.Tensor, error] {
	return func(yield func(map[string]onnx.Tensor, error) bool) {
		for _, path := range files {
			tensor, err := load(path)
			if err != nil {
				yield(nil, err)
				return
			}
			if !yield(map[string]onnx.Tensor{name: tensor}, nil) {
				return
			}
		}
	}
}

// syntheticSamples yields the same pseudo-random inputs on every pass.
func syntheticSamples(name string, shape []int64, count int, seed uint64, low, high float32) iter.Seq2[map[string]onnx.Tensor, error] {
	size := 1
	for _, d := range shape {
		size *= int(d)
	}
	return func(yield func(map[string]onnx.Tensor, error) bool) {
		random := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
		for range count {
			data := make([]float32, size)
			for i := range data {
				data[i] = low + (high-low)*random.Float32()
			}
			if !yield(map[string]onnx.Tensor{name: {Shape: shape, Float32: data}}, nil) {
				return
			}
		}
	}
}

func rawTensor(path string, shape []int64) (onnx.Tensor, error) {
	body, err := os.ReadFile(path)
	if err != nil {
		return onnx.Tensor{}, err
	}
	size := 1
	for _, d := range shape {
		size *= int(d)
	}
	if len(body) != 4*size {
		return onnx.Tensor{}, fmt.Errorf("%s has %d bytes and shape %v needs %d", filepath.Base(path), len(body), shape, 4*size)
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[4*i:]))
	}
	return onnx.Tensor{Shape: shape, Float32: data}, nil
}

func sampleShape(flagValue string, input onnx.TensorSpec) ([]int64, error) {
	shape := input.Shape
	if flagValue != "" {
		shape = nil
		for _, part := range splitList(flagValue) {
			d, err := strconv.ParseInt(part, 10, 64)
			if err != nil || d <= 0 {
				return nil, fmt.Errorf("-shape %q has to be positive integers", flagValue)
			}
			shape = append(shape, d)
		}
	}
	for _, d := range shape {
		if d <= 0 {
			return nil, fmt.Errorf("input %q has shape %v with a dynamic dimension; give -shape", input.Name, input.Shape)
		}
	}
	return shape, nil
}

func imageSize(flagValue string, input onnx.TensorSpec) (int, int, error) {
	if flagValue != "" {
		w, h, ok := strings.Cut(flagValue, "x")
		width, errW := strconv.Atoi(w)
		height, errH := strconv.Atoi(h)
		if !ok || errW != nil || errH != nil || width <= 0 || height <= 0 {
			return 0, 0, fmt.Errorf("-size %q has to be WIDTHxHEIGHT", flagValue)
		}
		return width, height, nil
	}
	if len(input.Shape) != 4 || input.Shape[2] <= 0 || input.Shape[3] <= 0 {
		return 0, 0, fmt.Errorf("input %q has shape %v, so give -size", input.Name, input.Shape)
	}
	return int(input.Shape[3]), int(input.Shape[2]), nil
}

func triple(value string) ([3]float32, error) {
	var out [3]float32
	parts := splitList(value)
	if len(parts) != 3 {
		return out, fmt.Errorf("%q has to be three comma-separated numbers", value)
	}
	for i, part := range parts {
		v, err := strconv.ParseFloat(part, 32)
		if err != nil {
			return out, fmt.Errorf("%q has to be three comma-separated numbers", value)
		}
		out[i] = float32(v)
	}
	return out, nil
}

func listFiles(dir string, extensions ...string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var files []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		for _, want := range extensions {
			if ext == want {
				files = append(files, filepath.Join(dir, entry.Name()))
			}
		}
	}
	sort.Strings(files)
	if len(files) == 0 {
		return nil, fmt.Errorf("%s has no %s file", dir, strings.Join(extensions, " or "))
	}
	return files, nil
}

func limit(files []string, n int) []string {
	if n > 0 && len(files) > n {
		return files[:n]
	}
	return files
}

func splitList(value string) []string {
	var out []string
	for _, part := range strings.Split(value, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func sortedKeys(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// writeFile writes the table next to its final name and renames it, so a
// reader never sees half a table.
func writeFile(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
