package main

import (
	"os"

	"github.com/consensys/gnark/frontend"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field/m31"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/test"
)

type Circuit struct {
	X frontend.Variable
	Y frontend.Variable
}

// Define declares the circuit's constraints
func (circuit *Circuit) Define(api frontend.API) error {
	api.AssertIsEqual(circuit.X, circuit.Y)
	return nil
}

func main() {
	circuit, err := ecgo.Compile(m31.ScalarField, &Circuit{})
	if err != nil {
		panic(err)
	}

	c := circuit.GetLayeredCircuit()
	os.WriteFile("circuit.txt", c.Serialize(), 0o644)

	assignment := &Circuit{
		X: 3,
		Y: 3,
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
