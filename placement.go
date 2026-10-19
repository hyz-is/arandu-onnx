package onnx

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// The provider names ONNX Runtime logs, which are not the names Config uses.
const (
	// LogProviderCPU is how ONNX Runtime names ProviderCPU in its logs.
	LogProviderCPU = "CPUExecutionProvider"
	// LogProviderCUDA is how ONNX Runtime names ProviderCUDA.
	LogProviderCUDA = "CUDAExecutionProvider"
	// LogProviderTensorRT is how ONNX Runtime names ProviderTensorRT.
	LogProviderTensorRT = "TensorrtExecutionProvider"
	// LogProviderCoreML is how ONNX Runtime names ProviderCoreML.
	LogProviderCoreML = "CoreMLExecutionProvider"
)

// logProviderNames maps Config's provider names onto the ones in the log.
var logProviderNames = map[string]string{
	ProviderCPU:      LogProviderCPU,
	ProviderCUDA:     LogProviderCUDA,
	ProviderTensorRT: LogProviderTensorRT,
	ProviderCoreML:   LogProviderCoreML,
}

// ErrCPUFallback is a session refused because PlacementOptions.RequireAccelerated
// is set and its work is not on an accelerator, or could not be shown to be.
var ErrCPUFallback = errors.New("onnx: model work fell back to the CPU")

// Placement is where ONNX Runtime put the nodes of one model on one device,
// read from the log of the session build that captured it.
type Placement struct {
	// Captured is whether a placement log was captured and read. Every other
	// field is empty when it is false, and an empty report is never a clean
	// one.
	Captured bool
	// Nodes is the number of nodes per provider, by the name ONNX Runtime
	// logs, such as LogProviderTensorRT. A TensorRT engine is one node however
	// many ONNX nodes it fused.
	Nodes map[string]int
	// CPUOps is the operators of the nodes placed on the CPU, with their
	// counts, when ONNX Runtime listed them. It lists them only when more than
	// one provider got nodes.
	CPUOps map[string]int
}

// clone copies the maps.
func (p Placement) clone() Placement {
	p.Nodes = maps.Clone(p.Nodes)
	p.CPUOps = maps.Clone(p.CPUOps)
	return p
}

// Fallback returns the nodes on the CPU whose operator is not in allowed, by
// operator. A graph placed wholly on the CPU is logged with no operator list,
// so it comes back as one entry, "every node", with the count.
func (p Placement) Fallback(allowed []string) map[string]int {
	if !p.Captured {
		return nil
	}
	ok := make(map[string]bool, len(allowed))
	for _, op := range allowed {
		ok[op] = true
	}
	out := map[string]int{}
	if p.Nodes[LogProviderCPU] > 0 && len(p.CPUOps) == 0 {
		out["every node"] = p.Nodes[LogProviderCPU]
		return out
	}
	for op, n := range p.CPUOps {
		if !ok[op] {
			out[op] += n
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// String is the placement in one line, such as
// "TensorrtExecutionProvider=3 CUDAExecutionProvider=12 CPUExecutionProvider=4".
func (p Placement) String() string {
	if !p.Captured {
		return "not captured"
	}
	names := make([]string, 0, len(p.Nodes))
	for name := range p.Nodes {
		names = append(names, name)
	}
	sort.Strings(names)
	parts := make([]string, 0, len(names))
	for _, name := range names {
		parts = append(parts, name+"="+strconv.Itoa(p.Nodes[name]))
	}
	return strings.Join(parts, " ")
}

// The lines ONNX Runtime logs (core/framework/session_state.cc,
// VerifyEachNodeIsAssignedToAnEp, read in 1.30.0), each behind the logger's own
// prefix, which differs by platform:
//
//	Node placements
//	 All nodes placed on [CUDAExecutionProvider]. Number of nodes: 846
//
// or, when more than one provider got nodes, one block per provider listing
// "OpType (node name)":
//
//	Node placements
//	 Node(s) placed on [CPUExecutionProvider]. Number of nodes: 2
//	  Shape (/Shape)
//	 Node(s) placed on [TensorrtExecutionProvider]. Number of nodes: 1
//	  TRTKernel_graph_main_17_0 (TensorrtExecutionProvider_TRTKernel_graph_main_17_0_0)
//	 Node(s) placed on [CUDAExecutionProvider]. Number of nodes: 498
//	  Conv (/conv1/Conv)
var (
	allNodesPattern  = regexp.MustCompile(`All nodes placed on \[([A-Za-z0-9_]+)\]\. Number of nodes: (\d+)`)
	someNodesPattern = regexp.MustCompile(`Node\(s\) placed on \[([A-Za-z0-9_]+)\]\. Number of nodes: (\d+)`)
	nodeLinePattern  = regexp.MustCompile(`\s{2}(\S+) \(.*\)\s*$`)
)

// ParsePlacement reads the "Node placements" section of an ONNX Runtime
// verbose log. It returns a Placement with Captured false when the log has
// none.
//
// The section ends at the first line that is neither a provider header nor a
// node of the current provider, so a later line naming a provider, such as an
// allocator's, is not counted. When the log holds more than one section, they
// are added together.
func ParsePlacement(log string) Placement {
	p := Placement{Nodes: map[string]int{}, CPUOps: map[string]int{}}
	inSection, current := false, ""
	scanner := bufio.NewScanner(strings.NewReader(log))
	scanner.Buffer(make([]byte, 0, 64*1024), 16<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasSuffix(strings.TrimSpace(line), "Node placements") {
			inSection, current = true, ""
			p.Captured = true
			continue
		}
		if !inSection {
			continue
		}
		if m := allNodesPattern.FindStringSubmatch(line); m != nil {
			n, _ := strconv.Atoi(m[2])
			p.Nodes[m[1]] += n
			inSection = false
			continue
		}
		if m := someNodesPattern.FindStringSubmatch(line); m != nil {
			n, _ := strconv.Atoi(m[2])
			p.Nodes[m[1]] += n
			current = m[1]
			continue
		}
		if m := nodeLinePattern.FindStringSubmatch(line); m != nil && current != "" {
			op := m[1]
			if current == LogProviderCPU {
				p.CPUOps[op]++
			}
			continue
		}
		inSection, current = false, ""
	}
	if !p.Captured {
		return Placement{}
	}
	return p
}

// checkPlacement applies RequireAccelerated to the placement of a session
// built for the accelerators named, first one first.
func checkPlacement(id string, device int, accelerators []string, p Placement, allowed []string) error {
	if len(accelerators) == 0 {
		return fmt.Errorf("%w: %s on device %d: no accelerator could build the session, and it would run on the CPU alone",
			ErrCPUFallback, id, device)
	}
	if !p.Captured {
		return fmt.Errorf("%w: %s on device %d: the node placement could not be captured, so nothing shows the work is on %s",
			ErrCPUFallback, id, device, accelerators[0])
	}
	if first := logProviderNames[accelerators[0]]; p.Nodes[first] == 0 {
		return fmt.Errorf("%w: %s on device %d: %s was asked for first and got no node (%s)",
			ErrCPUFallback, id, device, accelerators[0], p)
	}
	if fallback := p.Fallback(allowed); len(fallback) > 0 {
		return fmt.Errorf("%w: %s on device %d places %s on the CPU (%s). A QLinear*, QGemm or Nhwc* operator comes from a model quantized for the CPU; a DequantizeLinear comes from a static INT8 export that only TensorRT runs",
			ErrCPUFallback, id, device, describeOps(fallback), p)
	}
	return nil
}

// describeOps lists operator counts in a stable order: "QGemm x2, QLinearConv x186".
func describeOps(ops map[string]int) string {
	names := make([]string, 0, len(ops))
	for op := range ops {
		names = append(names, op)
	}
	sort.Strings(names)
	parts := make([]string, 0, len(names))
	for _, op := range names {
		parts = append(parts, fmt.Sprintf("%s x%d", op, ops[op]))
	}
	return strings.Join(parts, ", ")
}

// captureMu serialises the redirection of standard error.
var captureMu sync.Mutex

// capturePlacement runs build with the process's standard error captured, and
// reads the node placement out of what ONNX Runtime logged. Everything
// captured is written back to standard error afterwards.
//
// A capture that cannot be set up does not stop the build: build runs anyway
// and the placement comes back not captured, which RequireAccelerated refuses
// and nothing else mistakes for a clean report.
func capturePlacement(build func() error) (Placement, error) {
	captureMu.Lock()
	defer captureMu.Unlock()

	saved, err := dupStderr()
	if err != nil {
		return Placement{}, build()
	}
	defer closeFD(saved)

	reader, writer, err := os.Pipe()
	if err != nil {
		return Placement{}, build()
	}
	if err := redirectStderr(writer); err != nil {
		_ = reader.Close()
		_ = writer.Close()
		return Placement{}, build()
	}

	// The pipe's buffer is finite and a verbose build fills it, so it is
	// drained while the build runs.
	var captured bytes.Buffer
	drained := make(chan struct{})
	go func() {
		_, _ = io.Copy(&captured, reader)
		close(drained)
	}()

	buildErr := build()

	restoreErr := restoreStderr(saved)
	_ = writer.Close()
	<-drained
	_ = reader.Close()

	// The log still belongs on standard error: this is a capture, not a gag.
	_, _ = os.Stderr.Write(captured.Bytes())
	if restoreErr != nil {
		return Placement{}, buildErr
	}
	return ParsePlacement(captured.String()), buildErr
}
