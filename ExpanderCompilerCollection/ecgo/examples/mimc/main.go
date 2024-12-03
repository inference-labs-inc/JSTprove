package main

import (
	"os"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/hash/mimc"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/test"
)

const NHashes = 100

type Circuit struct {
	PreImage [NHashes]frontend.Variable
	Hash     [NHashes]frontend.Variable
}

func mimcHash(api frontend.API, preImage frontend.Variable, hash frontend.Variable) {
	mimc, _ := mimc.NewMiMC(api)
	mimc.Write(preImage)
	api.AssertIsEqual(mimc.Sum(), hash)
}

// Define declares the circuit's constraints
func (circuit *Circuit) Define(api frontend.API) error {
	for i := 0; i < NHashes; i++ {
		api.(ecgo.API).MemorizedCall(mimcHash, circuit.PreImage[i], circuit.Hash[i])
	}
	return nil
}

func main() {
	circuit, err := ecgo.Compile(ecc.BN254.ScalarField(), &Circuit{})
	if err != nil {
		panic(err)
	}

	c := circuit.GetLayeredCircuit()
	os.WriteFile("circuit.txt", c.Serialize(), 0o644)

	assignment := &Circuit{}
	for i := 0; i < NHashes; i++ {
		assignment.PreImage[i] = "16130099170765464552823636852555369511329944820189892919423002775646948828469"
		assignment.Hash[i] = "12886436712380113721405259596386800092738845035233065858332878701083870690753"
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
