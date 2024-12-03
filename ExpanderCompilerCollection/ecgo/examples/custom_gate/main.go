package main

import (
	"math/big"
	"os"

	"github.com/consensys/gnark/frontend"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field/m31"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/test"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils/customgates"
)

// Suppose we have a x^4 gate, which has id 12345 in the prover
const GATE_4TH_POWER_TYPE = 12345
const GATE_4TH_POWER_COST = 20

type Circuit struct {
	X frontend.Variable
	Y frontend.Variable
	Z frontend.Variable
}

func Power4(field *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	a := big.NewInt(0)
	a.Mul(inputs[0], inputs[0])
	a.Mul(a, a)
	outputs[0] = a
	return nil
}

// Define declares the circuit's constraints
func (circuit *Circuit) Define(api frontend.API) error {
	p := api.(ecgo.API).CustomGate(GATE_4TH_POWER_TYPE, api.Sub(circuit.X, circuit.Y))
	api.AssertIsEqual(p, circuit.Z)
	return nil
}

func main() {
	// Before we use custom gates, we must register it
	customgates.Register(GATE_4TH_POWER_TYPE, Power4, GATE_4TH_POWER_COST)

	circuit, err := ecgo.Compile(m31.ScalarField, &Circuit{})
	if err != nil {
		panic(err)
	}

	c := circuit.GetLayeredCircuit()
	os.WriteFile("circuit.txt", c.Serialize(), 0o644)

	assignment := &Circuit{
		X: 5,
		Y: 3,
		Z: 16,
	}
	inputSolver := circuit.GetInputSolver()
	witness, err := inputSolver.SolveInput(assignment, 8)
	if err != nil {
		panic(err)
	}

	if !test.CheckCircuit(c, witness) {
		panic("error")
	}

	os.WriteFile("inputsolver.txt", inputSolver.Serialize(), 0o644)
	os.WriteFile("circuit.txt", c.Serialize(), 0o644)
	os.WriteFile("witness.txt", witness.Serialize(), 0o644)
}
