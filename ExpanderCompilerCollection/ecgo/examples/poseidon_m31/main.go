package main

import (
	"fmt"
	"os"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field/m31"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/poseidon"
	ecc_test "github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/test"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
)

func main() {
	fmt.Println("Building M31 circuits")
	M31CircuitBuild()
}

const NumRepeat = 120

type MockPoseidonM31Circuit struct {
	State  [NumRepeat][16]frontend.Variable
	Digest [NumRepeat]frontend.Variable `gnark:",public"`
	Params *poseidon.PoseidonParams
}

func (c *MockPoseidonM31Circuit) Define(api frontend.API) (err error) {
	// Define the circuit
	engine := m31.Field{}
	for i := 0; i < NumRepeat; i++ {
		digest := poseidon.PoseidonCircuit(api, engine, c.Params, c.State[i][:], true)
		api.AssertIsEqual(digest, c.Digest[i])
	}

	return
}

func M31CircuitBuild() {

	param := poseidon.NewPoseidonParams()

	var states [NumRepeat][16]constraint.Element
	var stateVars [NumRepeat][16]frontend.Variable
	var outputVars [NumRepeat]frontend.Variable

	for i := 0; i < NumRepeat; i++ {
		for j := 0; j < 16; j++ {
			states[i][j] = constraint.Element{uint64(i)}
			stateVars[i][j] = frontend.Variable(uint64(i))
		}
		output := poseidon.PoseidonM31(param, states[i][:])
		outputVars[i] = frontend.Variable(output[0])

	}

	assignment := &MockPoseidonM31Circuit{
		State:  stateVars,
		Digest: outputVars,
		Params: param,
	}

	// Ecc test
	circuit, err := ecgo.Compile(m31.ScalarField, &MockPoseidonM31Circuit{
		State:  stateVars,
		Digest: outputVars,
		Params: param,
	}, frontend.WithCompressThreshold(32))
	if err != nil {
		panic(err)
	}

	layered_circuit := circuit.GetLayeredCircuit()
	// circuit.GetCircuitIr().Print()
	err = os.WriteFile("poseidon_120_circuit_m31.txt", layered_circuit.Serialize(), 0o644)
	if err != nil {
		panic(err)
	}
	inputSolver := circuit.GetInputSolver()
	witness, err := inputSolver.SolveInputAuto(assignment)
	if err != nil {
		panic(err)
	}
	err = os.WriteFile("poseidon_120_witness_m31.txt", witness.Serialize(), 0o644)
	if err != nil {
		panic(err)
	}
	if !ecc_test.CheckCircuit(layered_circuit, witness) {
		panic("verification failed")
	}

}
