package test

import (
	"crypto/rand"
	"math/big"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/irwg"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/layered"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils/customgates"
)

// check if first output is zero
func CheckCircuit(rc *layered.RootCircuit, witness *irwg.Witness) bool {
	if witness.NumWitnesses != 1 {
		panic("expected 1 witness, if you need to check multiple witnesses, use CheckCircuitMultiWitness")
	}
	out := evalCircuit(rc, witness.Values[:witness.NumInputsPerWitness], witness.Values[witness.NumInputsPerWitness:])
	for i := 0; i < rc.ExpectedNumOutputZeroes; i++ {
		if out[i].Cmp(big.NewInt(0)) != 0 {
			return false
		}
	}
	return true
}

func CheckCircuitMultiWitness(rc *layered.RootCircuit, witness *irwg.Witness) []bool {
	if witness.NumWitnesses == 0 {
		panic("expected at least 1 witness")
	}
	res := make([]bool, witness.NumWitnesses)
	a := witness.NumInputsPerWitness
	b := witness.NumPublicInputsPerWitness
	for i := 0; i < witness.NumWitnesses; i++ {
		out := evalCircuit(rc, witness.Values[i*(a+b):i*(a+b)+a], witness.Values[i*(a+b)+a:i*(a+b)+a+b])
		res[i] = true
		for j := 0; j < rc.ExpectedNumOutputZeroes; j++ {
			if out[j].Cmp(big.NewInt(0)) != 0 {
				res[i] = false
				break
			}
		}
	}
	return res
}

func EvalCircuit(rc *layered.RootCircuit, witness *irwg.Witness) []*big.Int {
	if witness.NumWitnesses != 1 {
		panic("expected 1 witness, if you need to check multiple witnesses, use CheckCircuitMultiWitness")
	}
	out := evalCircuit(rc, witness.Values[:witness.NumInputsPerWitness], witness.Values[witness.NumInputsPerWitness:])
	return out
}

func evalCircuit(rc *layered.RootCircuit, input []*big.Int, publicInput []*big.Int) []*big.Int {
	if len(input) != int(rc.Circuits[rc.Layers[0]].InputLen) {
		panic("input length mismatch")
	}
	// Current version of Expander does not support public input
	/*if len(publicInput) != rc.NumPublicInputs {
		panic("public input length mismatch")
	}*/
	cur := input
	// for layer_i, id := range rc.Layers {
	for _, id := range rc.Layers {
		next := make([]*big.Int, rc.Circuits[id].OutputLen)
		for i := range next {
			next[i] = big.NewInt(0)
		}
		applyCircuit(rc, rc.Circuits[id], cur, next, publicInput)
		cur = next
		for i := range cur {
			cur[i].Mod(cur[i], rc.Field)
		}
		// fmt.Printf("eval layer %d: %v\n", layer_i, cur[0])
	}
	return cur
}

func randInt() *big.Int {
	buf := make([]byte, 64) // just 2 times longer
	rand.Read(buf)
	return new(big.Int).SetBytes(buf)
}

func sampleCoef(coef *big.Int, coefType uint8, publicInputId uint64, publicInput []*big.Int) *big.Int {
	if coefType == 1 {
		return coef
	} else if coefType == 2 {
		return randInt()
	} else {
		return publicInput[publicInputId]
	}
}

func applyCircuit(rc *layered.RootCircuit, circuit *layered.Circuit, cur []*big.Int, next []*big.Int, publicInput []*big.Int) {
	tmp := big.NewInt(0)
	for _, m := range circuit.Mul {
		coef := sampleCoef(m.Coef, m.CoefType, m.PublicInputId, publicInput)
		tmp.Mul(cur[m.In0], cur[m.In1])
		next[m.Out].Add(next[m.Out], tmp.Mul(tmp, coef))
	}
	for _, a := range circuit.Add {
		coef := sampleCoef(a.Coef, a.CoefType, a.PublicInputId, publicInput)
		next[a.Out].Add(next[a.Out], tmp.Mul(cur[a.In], coef))
	}
	for _, c := range circuit.Cst {
		coef := sampleCoef(c.Coef, c.CoefType, c.PublicInputId, publicInput)
		next[c.Out].Add(next[c.Out], coef)
	}
	for _, ct := range circuit.Custom {
		inB := make([]*big.Int, len(ct.In))
		outB := []*big.Int{big.NewInt(0)}
		for i, e := range ct.In {
			inB[i] = cur[e]
		}
		hintFunc := customgates.GetFunc(ct.GateType)
		err := hintFunc(rc.Field, inB, outB)
		if err != nil {
			panic(err)
		}
		coef := sampleCoef(ct.Coef, ct.CoefType, ct.PublicInputId, publicInput)
		next[ct.Out].Add(next[ct.Out], tmp.Mul(outB[0], coef))
	}
	for _, sub := range circuit.SubCircuits {
		sc := rc.Circuits[sub.Id]
		for _, alloc := range sub.Allocations {
			applyCircuit(rc, sc,
				cur[alloc.InputOffset:alloc.InputOffset+sc.InputLen],
				next[alloc.OutputOffset:alloc.OutputOffset+sc.OutputLen],
				publicInput,
			)
		}
	}
}
