package rust

import (
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/irsource"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/irwg"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/layered"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/rust/wrapper"
)

func Compile(rc *irsource.RootCircuit) (*irwg.RootCircuit, *layered.RootCircuit, error) {
	s := irsource.SerializeRootCircuit(rc)
	irWgSer, lcSer, err := wrapper.CompileWithRustLib(s, field.GetFieldId(rc.Field))
	if err != nil {
		return nil, nil, err
	}
	irWg := irwg.DeserializeRootCircuit(irWgSer)
	lc := layered.DeserializeRootCircuit(lcSer)
	return irWg, lc, nil
}

func ProveFile(circuitFilename string, witnessBytes []byte) []byte {
	return wrapper.ProveCircuitFile(circuitFilename, witnessBytes, layered.DetectFieldIdFromFile(circuitFilename))
}

func VerifyFile(circuitFilename string, witnessBytes []byte, proofBytes []byte) bool {
	return wrapper.VerifyCircuitFile(circuitFilename, witnessBytes, proofBytes, layered.DetectFieldIdFromFile(circuitFilename))
}
