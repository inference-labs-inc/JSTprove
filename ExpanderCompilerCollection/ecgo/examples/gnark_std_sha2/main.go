package main

import (
	"crypto/sha256"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/test"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/hash/sha2"
	"github.com/consensys/gnark/std/math/uints"
)

const N = 100

type Circuit struct {
	Input [N]frontend.Variable
}

func (t *Circuit) Define(api frontend.API) error {
	h, err := sha2.New(api)
	if err != nil {
		return err
	}
	s := make([]uints.U8, N)
	for i := 0; i < N; i++ {
		s[i].Val = t.Input[i]
	}
	h.Write(s)
	sum := h.Sum()
	for _, x := range sum {
		api.(ecgo.API).Output(x.Val)
	}
	api.AssertIsDifferent(t.Input[0], 0)
	return nil
}

func main() {
	var circuit Circuit

	cr, err := ecgo.Compile(ecc.BN254.ScalarField(), &circuit)
	if err != nil {
		panic(err)
	}

	c := cr.GetLayeredCircuit()

	inputSolver := cr.GetInputSolver()
	for i := 0; i < N; i++ {
		circuit.Input[i] = i + 1
	}
	witness, err := inputSolver.SolveInput(&circuit, 8)
	if err != nil {
		panic(err)
	}

	res := test.EvalCircuit(c, witness)
	//fmt.Println(res)

	s := make([]byte, N)
	for i := 0; i < N; i++ {
		s[i] = byte(i + 1)
	}
	sum := sha256.Sum256(s)
	//fmt.Println(sum)

	for i := 0; i < 8; i++ {
		for j := 0; j < 4; j++ {
			if res[i*4+j+1].Int64() != int64(sum[i*4+3-j]) {
				panic("gg")
			}
		}
	}
}
