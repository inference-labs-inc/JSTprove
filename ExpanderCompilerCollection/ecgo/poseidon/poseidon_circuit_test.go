package poseidon

import (
	"testing"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field/m31"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/test"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
)

type MockPoseidonCircuit struct {
	State  [16]frontend.Variable `gnark:",public"`
	Output frontend.Variable     `gnark:",public"`
}

func (c *MockPoseidonCircuit) Define(api frontend.API) (err error) {
	param := NewPoseidonParams()
	engine := m31.Field{}
	t := PoseidonCircuit(api, engine, param, c.State[:], false)
	api.AssertIsEqual(t, c.Output)

	return
}

func TestPoseidonCircuit(t *testing.T) {
	param := NewPoseidonParams()

	var states [16]constraint.Element
	var stateVars [16]frontend.Variable
	var outputVar frontend.Variable

	for j := 0; j < 16; j++ {
		states[j] = constraint.Element{uint64(j)}
		stateVars[j] = frontend.Variable(uint64(j))
	}
	output := PoseidonM31(param, states[:])
	outputVar = frontend.Variable(output[0])

	assignment := &MockPoseidonCircuit{
		State:  stateVars,
		Output: outputVar,
	}

	// Gnark test disabled as it does not support randomness and custom gates
	// err := test.IsSolved(&MockPoseidonCircuit{}, assignment, m31.ScalarField)
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println("Gnark test passed")

	// Ecc test
	circuit, err := ecgo.Compile(m31.ScalarField, &MockPoseidonCircuit{}, frontend.WithCompressThreshold(32))
	if err != nil {
		panic(err)
	}

	layered_circuit := circuit.GetLayeredCircuit()
	// circuit.GetCircuitIr().Print()

	inputSolver := circuit.GetInputSolver()
	witness, err := inputSolver.SolveInputAuto(assignment)
	if err != nil {
		panic(err)
	}

	if !test.CheckCircuit(layered_circuit, witness) {
		panic("verification failed")
	}
}
