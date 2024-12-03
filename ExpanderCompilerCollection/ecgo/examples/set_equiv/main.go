package main

import (
	"math/rand"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/test"
)

const SetSize = 100

type Circuit struct {
	Set1 [SetSize]frontend.Variable
	Set2 [SetSize]frontend.Variable
}

func MulAll(api frontend.API, vars []frontend.Variable) frontend.Variable {
	if len(vars) == 1 {
		return vars[0]
	}
	mid := len(vars) / 2
	return api.Mul(MulAll(api, vars[:mid]), MulAll(api, vars[mid:]))
}

// Define declares the circuit's constraints
func (circuit *Circuit) Define(api frontend.API) error {
	z := api.(ecgo.API).GetRandomValue()
	diff1 := []frontend.Variable{}
	for _, x := range circuit.Set1 {
		diff1 = append(diff1, api.Sub(z, x))
	}
	diff2 := []frontend.Variable{}
	for _, x := range circuit.Set2 {
		diff2 = append(diff2, api.Sub(z, x))
	}
	prod1 := MulAll(api, diff1)
	prod2 := MulAll(api, diff2)
	api.AssertIsEqual(prod1, prod2)
	return nil
}

func main() {
	circuit, err := ecgo.Compile(ecc.BN254.ScalarField(), &Circuit{})
	if err != nil {
		panic(err)
	}

	c := circuit.GetLayeredCircuit()

	assignment := &Circuit{}
	for i := 0; i < SetSize; i++ {
		assignment.Set1[i] = rand.Intn(1000000)
		assignment.Set2[i] = assignment.Set1[i]
	}
	rand.Shuffle(SetSize, func(i, j int) {
		assignment.Set2[i], assignment.Set2[j] = assignment.Set2[j], assignment.Set2[i]
	})
	inputSolver := circuit.GetInputSolver()
	witness, err := inputSolver.SolveInput(assignment, 1)
	if err != nil {
		panic(err)
	}

	if !test.CheckCircuit(c, witness) {
		panic("error")
	}
}
