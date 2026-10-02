package onnx

// Config is what the application passes when it wires this package.
//
// A typed struct rather than a map: a misspelled key in a map is a setting that
// silently keeps its default. Here a field that does not exist does not
// compile.
type Config struct{}

// Validate reports what the configuration cannot be used with.
func (c Config) Validate() error { return nil }

// withDefaults returns the configuration with the optional fields filled in.
// It runs after Validate and never before.
func (c Config) withDefaults() Config { return c }
