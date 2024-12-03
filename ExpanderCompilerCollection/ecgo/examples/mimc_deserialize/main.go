package main

import (
	"errors"
	"os"

	"github.com/consensys/gnark/frontend"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/test"
)

const NHashes = 100

type Circuit struct {
	PreImage [NHashes]frontend.Variable
	Hash     [NHashes]frontend.Variable
}

func (circuit *Circuit) Define(api frontend.API) error {
	return errors.New("this will not be called")
}

func main() {
	assignment := &Circuit{}
	for i := 0; i < NHashes; i++ {
		assignment.PreImage[i] = "16130099170765464552823636852555369511329944820189892919423002775646948828469"
		assignment.Hash[i] = "12886436712380113721405259596386800092738845035233065858332878701083870690753"
	}
	s, _ := os.ReadFile("inputsolver.txt")
	inputSolver := ecgo.DeserializeInputSolver(s)
	s, _ = os.ReadFile("circuit.txt")
	c := ecgo.DeserializeLayeredCircuit(s)
	witness, err := inputSolver.SolveInput(assignment, 8)
	if err != nil {
		panic(err)
	}

	if !test.CheckCircuit(c, witness) {
		panic("error")
	}
}
