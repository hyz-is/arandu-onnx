package unit_test

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/hyz-is/arandu-onnx/calibration"
)

func TestMinMaxTakesTheLargestMagnitudeOverEverySample(t *testing.T) {
	t.Parallel()

	c, err := calibration.NewCollector(calibration.Options{
		Exclude:         []string{"scores"},
		ExcludePrefixes: []string{"/head/"},
	})
	if err != nil {
		t.Fatalf("NewCollector: %v", err)
	}
	if c.NeedsHistogram() {
		t.Fatal("minmax asked for a histogram pass")
	}
	nan := float32(math.NaN())
	observe(t, c.ObserveRange, map[string][]float32{
		"x":          {-3, 1, 2},
		"y":          {0.25, nan, -0.5},
		"dead":       {0, 0, 0},
		"scores":     {100},
		"/head/conv": {50},
	})
	observe(t, c.ObserveRange, map[string][]float32{
		"x":    {2.5},
		"y":    {0.75},
		"dead": {0},
	})

	table, report, err := c.Table()
	if err != nil {
		t.Fatalf("Table: %v", err)
	}
	assertTablesEqual(t, table, calibration.Table{"x": 3, "y": 0.75}, 0)
	if len(report.Zero) != 1 || report.Zero[0] != "dead" {
		t.Errorf("Report.Zero = %v, want [dead]: a range of 0 is not one TensorRT takes", report.Zero)
	}
	if report.NonFinite["y"] != 1 {
		t.Errorf("Report.NonFinite = %v, want y: 1", report.NonFinite)
	}
	if report.Method != calibration.MethodMinMax || len(report.Calibrated) != 2 {
		t.Errorf("Report = %+v", report)
	}
}

func TestPercentileClipsTheRarestValues(t *testing.T) {
	t.Parallel()

	// 10 000 values spread evenly over [0, 1), and one outlier at 100 that
	// sets the histogram's span. The 99th percentile is at 0.99 of the bulk,
	// within one bin of 100 / 2048.
	values := make([]float32, 0, 10001)
	for i := range 10000 {
		values = append(values, float32(i)/10000)
	}
	values = append(values, 100)

	c, err := calibration.NewCollector(calibration.Options{Method: calibration.MethodPercentile, Percentile: 99})
	if err != nil {
		t.Fatalf("NewCollector: %v", err)
	}
	if !c.NeedsHistogram() {
		t.Fatal("percentile did not ask for the histogram pass")
	}
	observe(t, c.ObserveRange, map[string][]float32{"t": values})
	if _, _, err := c.Table(); !errors.Is(err, calibration.ErrPhase) {
		t.Fatalf("Table before the histogram pass = %v, want ErrPhase", err)
	}
	observe(t, c.ObserveHistogram, map[string][]float32{"t": values})
	if err := c.ObserveRange("t", values); !errors.Is(err, calibration.ErrPhase) {
		t.Errorf("ObserveRange after the histogram pass began = %v, want ErrPhase", err)
	}
	table, _, err := c.Table()
	if err != nil {
		t.Fatalf("Table: %v", err)
	}
	width := 100.0 / 2048
	if r := float64(table["t"]); math.Abs(r-0.99) > width {
		t.Errorf("99th percentile = %v, want 0.99 within one bin (%v)", r, width)
	}
}

func TestEntropyFindsAThresholdInsideTheOutliers(t *testing.T) {
	t.Parallel()

	// A normal distribution of standard deviation 1, and a handful of values
	// at 40: minmax would spend the 127 levels on [0, 40] and leave the bulk
	// a dozen of them. The entropy threshold has to sit well inside the
	// outliers and outside the bulk.
	random := rand.New(rand.NewPCG(7, 11))
	values := make([]float32, 0, 50005)
	for range 50000 {
		values = append(values, float32(random.NormFloat64()))
	}
	for range 5 {
		values = append(values, 40)
	}

	c, err := calibration.NewCollector(calibration.Options{Method: calibration.MethodEntropy})
	if err != nil {
		t.Fatalf("NewCollector: %v", err)
	}
	observe(t, c.ObserveRange, map[string][]float32{"t": values})
	observe(t, c.ObserveHistogram, map[string][]float32{"t": values})
	table, _, err := c.Table()
	if err != nil {
		t.Fatalf("Table: %v", err)
	}
	r := table["t"]
	t.Logf("entropy threshold for a unit normal with outliers at 40: %v", r)
	if r < 2.5 || r > 20 {
		t.Errorf("entropy threshold = %v, want between 2.5 and 20 for a unit normal with outliers at 40", r)
	}
}

func TestTheHistogramPassReadsTheSameTensors(t *testing.T) {
	t.Parallel()

	c, err := calibration.NewCollector(calibration.Options{Method: calibration.MethodEntropy})
	if err != nil {
		t.Fatalf("NewCollector: %v", err)
	}
	observe(t, c.ObserveRange, map[string][]float32{"a": {1}})
	if err := c.ObserveHistogram("b", []float32{1}); !errors.Is(err, calibration.ErrPhase) {
		t.Errorf("ObserveHistogram of an unseen tensor = %v, want ErrPhase", err)
	}

	minmax, _ := calibration.NewCollector(calibration.Options{})
	if err := minmax.ObserveHistogram("a", []float32{1}); !errors.Is(err, calibration.ErrPhase) {
		t.Errorf("ObserveHistogram on minmax = %v, want ErrPhase", err)
	}
}

func TestCalibrationOptionsAreChecked(t *testing.T) {
	t.Parallel()

	for name, opts := range map[string]calibration.Options{
		"method":         {Method: "kl"},
		"percentile 0+":  {Method: calibration.MethodPercentile, Percentile: -1},
		"percentile 101": {Method: calibration.MethodPercentile, Percentile: 101},
		"bins":           {Method: calibration.MethodEntropy, Bins: 100},
	} {
		if _, err := calibration.NewCollector(opts); err == nil {
			t.Errorf("%s: NewCollector accepted %+v", name, opts)
		}
	}
}

func observe(t *testing.T, record func(string, []float32) error, values map[string][]float32) {
	t.Helper()
	for name, v := range values {
		if err := record(name, v); err != nil {
			t.Fatalf("observing %s: %v", name, err)
		}
	}
}
