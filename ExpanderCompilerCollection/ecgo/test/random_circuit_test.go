package test

import (
	"math/big"
	"testing"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/consensys/gnark-crypto/ecc"
)

func testRandomCircuit(t *testing.T, conf *randomCircuitConfig, seedL int, seedR int, nCase int) {
	a := NewAssert(t)
	for seed := seedL; seed <= seedR; seed++ {
		conf.seed = seed
		rcg := newRandomCircuitGenerator(conf)
		circuit := rcg.circuit()
		c, err := ecgo.Compile(ecc.BN254.ScalarField(), circuit)
		if err != nil {
			t.Fatal(err)
		}
		for i := 1; i <= nCase; i++ {
			assignment := rcg.randomAssignment(i)
			a.ProveSucceeded(c, assignment)
			t := big.NewInt(1)
			assignment.Output = t.Add(t, assignment.Output.(*big.Int))
			a.ProveFailed(c, assignment)
		}
	}
}

func TestRandomCircuit1(t *testing.T) {
	testRandomCircuit(t, &randomCircuitConfig{
		seed:       11,
		scNum:      randRange{5, 20},
		scInput:    randRange{5, 50},
		scOutput:   randRange{5, 30},
		scInsn:     randRange{20, 50},
		rootInsn:   randRange{30, 200},
		field:      ecc.BN254.ScalarField(),
		addPercent: 60,
		mulPercent: 90,
		divPercent: 97,
	}, 1, 1000, 2)
}

func TestRandomCircuit2(t *testing.T) {
	testRandomCircuit(t, &randomCircuitConfig{
		seed:       12,
		scNum:      randRange{1, 1},
		scInput:    randRange{100, 100},
		scOutput:   randRange{100, 100},
		scInsn:     randRange{1000, 1000},
		rootInsn:   randRange{1000, 1000},
		field:      ecc.BN254.ScalarField(),
		addPercent: 60,
		mulPercent: 90,
		divPercent: 97,
	}, 11, 20, 10)
}

func TestRandomCircuit3(t *testing.T) {
	testRandomCircuit(t, &randomCircuitConfig{
		seed:       13,
		scNum:      randRange{50, 50},
		scInput:    randRange{1, 50},
		scOutput:   randRange{2, 2},
		scInsn:     randRange{10, 10},
		rootInsn:   randRange{10, 10},
		field:      ecc.BN254.ScalarField(),
		addPercent: 20,
		mulPercent: 40,
		divPercent: 40,
	}, 11, 20, 20)
}
