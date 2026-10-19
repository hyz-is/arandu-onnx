# Placement log fixtures

What `ParsePlacement` is tested against.

- `darwin-cpu-all.log`, `darwin-coreml-mixed.log`: captured on macOS 26
  (Apple Silicon) with Homebrew ONNX Runtime 1.30.0, building tiny synthetic
  graphs at verbose severity. They are real output, prefixes included.
- `linux-*.log`: written by hand in the format of ONNX Runtime's default
  Linux sink, from `core/framework/session_state.cc` (`VerifyEachNodeIsAssignedToAnEp`)
  and `core/providers/tensorrt/tensorrt_execution_provider.cc` (the
  `TRTKernel_<graph type>_<graph name>_<hash>_<index>` fused-node names) of
  1.30.0. No NVIDIA machine was available to capture them; replace them with
  captured logs once one is.
