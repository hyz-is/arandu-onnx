package calibration

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// Method is how a tensor's dynamic range is chosen from the values it took.
type Method string

// The methods a Collector applies.
const (
	// MethodMinMax takes the largest magnitude the tensor reached over every
	// sample: max(|min|, |max|). It clips nothing, so an outlier widens the
	// range and costs resolution everywhere else. It is ONNX Runtime's MinMax
	// calibrator with symmetric ranges, the table arandu-alpr validated in
	// INT8 on Tesla T4 cards, and the default.
	MethodMinMax Method = "minmax"
	// MethodPercentile takes the magnitude below which Options.Percentile per
	// cent of the tensor's values fall, read from a histogram of |x|. It clips
	// the rarest values to keep resolution for the rest.
	MethodPercentile Method = "percentile"
	// MethodEntropy takes the threshold whose 128-level quantization of the
	// histogram of |x| loses the least information, measured as the
	// Kullback-Leibler divergence. It is the algorithm NVIDIA published for
	// TensorRT's entropy calibration (Migacz, "8-bit Inference with
	// TensorRT", 2017); TensorRT's own EntropyCalibration2 is closed source,
	// so the thresholds are close to, not identical with, what it would pick.
	MethodEntropy Method = "entropy"
)

// The defaults Options fills in.
const (
	// DefaultPercentile is the percentile MethodPercentile uses when
	// Options.Percentile is zero.
	DefaultPercentile = 99.999
	// DefaultBins is the histogram resolution when Options.Bins is zero, the
	// 2048 bins TensorRT's entropy calibration uses.
	DefaultBins = 2048
	// quantizedLevels is the number of non-negative INT8 levels.
	quantizedLevels = 128
)

// Options says how a Collector turns observed values into ranges, and which
// tensors it leaves out.
type Options struct {
	// Method is how a range is chosen. Empty means MethodMinMax.
	Method Method
	// Percentile is MethodPercentile's percentile, in (0, 100]. Zero means
	// DefaultPercentile.
	Percentile float64
	// Bins is the histogram resolution of MethodPercentile and MethodEntropy,
	// at least 2 * 128 for MethodEntropy. Zero means DefaultBins.
	Bins int
	// Exclude is tensors left out of the table by exact name. A tensor left
	// out runs at FP16 or FP32 under TensorRT.
	Exclude []string
	// ExcludePrefixes is tensors left out by name prefix, such as a detector
	// head "/model/model.22/".
	ExcludePrefixes []string
}

// withDefaults fills in the zero values.
func (o Options) withDefaults() Options {
	if o.Method == "" {
		o.Method = MethodMinMax
	}
	if o.Percentile == 0 {
		o.Percentile = DefaultPercentile
	}
	if o.Bins == 0 {
		o.Bins = DefaultBins
	}
	return o
}

// Validate reports options a Collector cannot work with.
func (o Options) Validate() error {
	o = o.withDefaults()
	switch o.Method {
	case MethodMinMax, MethodPercentile, MethodEntropy:
	default:
		return fmt.Errorf("calibration: method %q is not %q, %q or %q", o.Method, MethodMinMax, MethodPercentile, MethodEntropy)
	}
	if math.IsNaN(o.Percentile) || o.Percentile <= 0 || o.Percentile > 100 {
		return fmt.Errorf("calibration: percentile %v has to be in (0, 100]", o.Percentile)
	}
	if o.Bins < 2*quantizedLevels || o.Bins > 1<<20 {
		return fmt.Errorf("calibration: %d bins has to be between %d and %d", o.Bins, 2*quantizedLevels, 1<<20)
	}
	return nil
}

// Excluded reports whether the options leave a tensor out.
func (o Options) Excluded(name string) bool {
	for _, exact := range o.Exclude {
		if name == exact {
			return true
		}
	}
	for _, prefix := range o.ExcludePrefixes {
		if prefix != "" && strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// ErrPhase is a Collector method called out of order.
var ErrPhase = errors.New("calibration: collector used out of order")

// Collector accumulates what each tensor took over the calibration samples.
//
// It works in one or two passes over the same samples. Every method needs the
// first, ObserveRange, which records each tensor's largest magnitude.
// MethodPercentile and MethodEntropy then need a second, ObserveHistogram,
// over the same samples again, which counts |x| into a histogram spanning
// exactly that magnitude: binning over a range known in advance is what keeps
// the resolution the method was asked for. NeedsHistogram says whether the
// second pass is due.
//
// A Collector is not safe for concurrent use.
type Collector struct {
	opts       Options
	absMax     map[string]float64
	histograms map[string][]float64
	nonFinite  map[string]int
	phase      int
}

// NewCollector returns a collector for the options.
func NewCollector(opts Options) (*Collector, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	return &Collector{
		opts:      opts.withDefaults(),
		absMax:    map[string]float64{},
		nonFinite: map[string]int{},
	}, nil
}

// Options returns the options the collector applies, defaults filled in.
func (c *Collector) Options() Options { return c.opts }

// NeedsHistogram reports whether the method needs the second pass.
func (c *Collector) NeedsHistogram() bool { return c.opts.Method != MethodMinMax }

// ObserveRange records the values one tensor took on one sample, in the first
// pass. Excluded tensors are ignored. NaN and infinite values are counted and
// left out of the range.
func (c *Collector) ObserveRange(name string, values []float32) error {
	if c.phase != 0 {
		return fmt.Errorf("%w: ObserveRange after the histogram pass began", ErrPhase)
	}
	if c.opts.Excluded(name) {
		return nil
	}
	current, seen := c.absMax[name]
	if !seen {
		current = 0
	}
	for _, v := range values {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			c.nonFinite[name]++
			continue
		}
		current = max(current, math.Abs(f))
	}
	c.absMax[name] = current
	return nil
}

// ObserveHistogram counts the values one tensor took on one sample into its
// histogram, in the second pass. A tensor the first pass did not see is
// refused: the passes have to cover the same samples.
func (c *Collector) ObserveHistogram(name string, values []float32) error {
	if !c.NeedsHistogram() {
		return fmt.Errorf("%w: %s needs no histogram pass", ErrPhase, c.opts.Method)
	}
	if c.opts.Excluded(name) {
		return nil
	}
	if c.phase == 0 {
		c.phase = 1
		c.histograms = make(map[string][]float64, len(c.absMax))
	}
	limit, seen := c.absMax[name]
	if !seen {
		return fmt.Errorf("%w: tensor %q was not observed in the range pass", ErrPhase, name)
	}
	hist := c.histograms[name]
	if hist == nil {
		hist = make([]float64, c.opts.Bins)
		c.histograms[name] = hist
	}
	if limit == 0 {
		hist[0] += float64(len(values))
		return nil
	}
	width := limit / float64(c.opts.Bins)
	last := c.opts.Bins - 1
	for _, v := range values {
		f := math.Abs(float64(v))
		if math.IsNaN(f) || math.IsInf(f, 0) {
			continue
		}
		bin := int(f / width)
		hist[min(bin, last)]++
	}
	return nil
}

// Report is what a Collector left out of its table, and why.
type Report struct {
	// Method is the method the ranges were chosen with.
	Method Method
	// Calibrated is the tensors in the table, in name order.
	Calibrated []string
	// Zero is the tensors that were 0 on every sample. A range of 0 is not
	// one TensorRT can be given, so they are left out and run at a higher
	// precision.
	Zero []string
	// NonFinite counts, per tensor, the NaN and infinite values that were left
	// out of its range. A tensor with any deserves a look before the table is
	// used.
	NonFinite map[string]int
}

// Table returns the ranges, and the report of what was left out.
func (c *Collector) Table() (Table, Report, error) {
	if c.NeedsHistogram() && c.phase == 0 && len(c.absMax) > 0 {
		return nil, Report{}, fmt.Errorf("%w: %s needs the histogram pass before the table", ErrPhase, c.opts.Method)
	}
	report := Report{Method: c.opts.Method, NonFinite: map[string]int{}}
	for name, n := range c.nonFinite {
		report.NonFinite[name] = n
	}
	table := Table{}
	names := make([]string, 0, len(c.absMax))
	for name := range c.absMax {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		limit := c.absMax[name]
		r := limit
		if c.NeedsHistogram() && limit > 0 {
			hist := c.histograms[name]
			if hist == nil {
				return nil, Report{}, fmt.Errorf("%w: tensor %q was not observed in the histogram pass", ErrPhase, name)
			}
			width := limit / float64(c.opts.Bins)
			switch c.opts.Method {
			case MethodPercentile:
				r = percentileThreshold(hist, width, c.opts.Percentile)
			case MethodEntropy:
				r = entropyThreshold(hist, width)
			}
		}
		if r <= 0 || float32(r) <= 0 {
			report.Zero = append(report.Zero, name)
			continue
		}
		table[name] = float32(r)
		report.Calibrated = append(report.Calibrated, name)
	}
	return table, report, nil
}

// percentileThreshold is the upper edge of the bin where the cumulative count
// reaches the percentile.
func percentileThreshold(hist []float64, width, percentile float64) float64 {
	total := 0.0
	for _, n := range hist {
		total += n
	}
	if total == 0 {
		return 0
	}
	target := total * percentile / 100
	cumulative := 0.0
	for i, n := range hist {
		cumulative += n
		if cumulative >= target {
			return float64(i+1) * width
		}
	}
	return float64(len(hist)) * width
}

// entropyThreshold searches the threshold, from 128 bins to all of them, whose
// quantization to 128 levels has the smallest KL divergence from the clipped
// distribution, and returns the threshold's value: the upper edge of its last bin.
func entropyThreshold(hist []float64, width float64) float64 {
	bins := len(hist)
	best, bestDivergence := bins, math.Inf(1)
	reference := make([]float64, bins)
	candidate := make([]float64, bins)
	for i := quantizedLevels; i <= bins; i++ {
		// The reference: the first i bins, with everything beyond the
		// threshold folded into the last one, as clipping would.
		copy(reference, hist[:i])
		outliers := 0.0
		for _, n := range hist[i:] {
			outliers += n
		}
		reference[i-1] += outliers
		if reference[i-1] == 0 && allZero(hist[:i]) {
			continue
		}

		// The candidate: the first i bins merged into 128 levels, each level's
		// count spread evenly over the bins of it that were not empty.
		quantize(hist[:i], candidate[:i])
		divergence := klDivergence(reference[:i], candidate[:i])
		if divergence < bestDivergence {
			best, bestDivergence = i, divergence
		}
	}
	return float64(best) * width
}

// quantize merges the bins into quantizedLevels levels and expands them back.
func quantize(hist, out []float64) {
	n := len(hist)
	for level := range quantizedLevels {
		start := level * n / quantizedLevels
		end := (level + 1) * n / quantizedLevels
		if level == quantizedLevels-1 {
			end = n
		}
		sum, nonEmpty := 0.0, 0
		for _, v := range hist[start:end] {
			sum += v
			if v != 0 {
				nonEmpty++
			}
		}
		for j := start; j < end; j++ {
			out[j] = 0
			if hist[j] != 0 && nonEmpty > 0 {
				out[j] = sum / float64(nonEmpty)
			}
		}
	}
}

// klDivergence is KL(p || q) of two unnormalised distributions, with q
// smoothed so that a bin p has and q lacks is penalised rather than infinite.
func klDivergence(p, q []float64) float64 {
	const epsilon = 1e-4
	sumP, sumQ := 0.0, 0.0
	for i := range p {
		sumP += p[i]
		sumQ += q[i]
	}
	if sumP == 0 || sumQ == 0 {
		return math.Inf(1)
	}
	divergence := 0.0
	for i := range p {
		if p[i] == 0 {
			continue
		}
		pi := p[i] / sumP
		qi := q[i] / sumQ
		if qi == 0 {
			qi = epsilon
		}
		divergence += pi * math.Log(pi/qi)
	}
	return divergence
}

func allZero(values []float64) bool {
	for _, v := range values {
		if v != 0 {
			return false
		}
	}
	return true
}
