package poseidon

import (
	"log"
	"math/big"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field/m31"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils/customgates"
	"github.com/consensys/gnark/frontend"
)

type PoseidonInternalStateVar struct {
	AfterHalfFullRound    [16]frontend.Variable
	AfterHalfPartialRound [16]frontend.Variable
	AfterPartialRound     [16]frontend.Variable
}

// Suppose we have a x^4 gate, which has id 12345 in the prover
const GATE_5TH_POWER_TYPE = 12345
const GATE_4TH_POWER_COST = 20

const GATE_MUL_TYPE = 12346
const GATE_MUL_COST = 20

func Mul(field *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	a := big.NewInt(0)
	a.Mul(inputs[0], big.NewInt(1))
	outputs[0] = a
	return nil
}

func init() {
	customgates.Register(GATE_5TH_POWER_TYPE, Power5, GATE_4TH_POWER_COST)
	customgates.Register(GATE_MUL_TYPE, Mul, GATE_MUL_COST)
}

// Main function of proving poseidon in circuit.
//
// To obtain a more efficient layered circuit representation, we also feed the internal state of the hash to this function.
func PoseidonCircuit(
	api frontend.API,
	engine m31.Field,
	param *PoseidonParams,
	input []frontend.Variable,
	useRandomness bool) frontend.Variable {
	// todo: pad the input if it is too short
	if len(input) != param.NumStates {
		log.Println("input length", len(input), "does not match the number of states in the Poseidon parameters")
		panic("")
	}

	// ============================
	// Applies the full rounds.
	// ============================
	state := input

	for i := 0; i < param.NumHalfFullRounds; i++ {
		// add round constant
		for j := 0; j < param.NumStates; j++ {
			state[j] = api.Add(state[j], param.ExternalRoundConstant[j][i])
		}
		// apply affine transform
		tmp := applyMdsMatrixCircuit(api, state, param.MdsMatrix)
		state = tmp[:]
		// sbox
		for j := 0; j < param.NumStates; j++ {
			state[j] = sBoxCircuit(api, state[j])
		}
	}

	// ============================
	// Applies the first half of partial rounds.
	// ============================

	for i := 0; i < param.NumPartRounds; i++ {
		// add round constant
		state[0] = api.Add(state[0], param.InternalRoundConstant[i])
		// apply affine transform
		tmp := applyMdsMatrixCircuit(api, state, param.MdsMatrix)
		state = tmp[:]
		// sbox
		state[0] = sBoxCircuit(api, state[0])
		for j := 1; j < param.NumStates; j++ {
			state[j] = api.(ecgo.API).CustomGate(GATE_MUL_TYPE, state[j])
		}
	}

	// ============================
	// Applies the full rounds.
	// ============================

	for i := 0; i < param.NumHalfFullRounds; i++ {
		// add round constant
		for j := 0; j < param.NumStates; j++ {
			state[j] = api.Add(state[j], param.ExternalRoundConstant[j][i+param.NumHalfFullRounds])
		}
		// apply affine transform
		tmp := applyMdsMatrixCircuit(api, state, param.MdsMatrix)
		state = tmp[:]
		// sbox
		for j := 0; j < param.NumStates; j++ {
			state[j] = sBoxCircuit(api, state[j])
		}
	}

	return state[0]
}

func accumulate(api frontend.API, a []frontend.Variable) frontend.Variable {
	return api.Add(a[0], a[1], a[2:]...)
}

func applyMdsMatrixCircuit(api frontend.API, x []frontend.Variable, mds [][]uint32) [16]frontend.Variable {
	var res [16]frontend.Variable
	for i := 0; i < 16; i++ {
		var tmp [16]frontend.Variable
		for j := 0; j < 16; j++ {
			tmp[j] = api.Mul(x[j], mds[j][i])
		}
		res[i] = accumulate(api, tmp[:])
	}
	return res
}

func Power5(field *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	a := big.NewInt(0)
	a.Mul(inputs[0], inputs[0])
	a.Mul(a, a)
	a.Mul(a, inputs[0])
	outputs[0] = a
	return nil
}

// S-Box: raise element to the power of 5
func sBoxCircuit(api frontend.API, input frontend.Variable) frontend.Variable {
	return api.(ecgo.API).CustomGate(GATE_5TH_POWER_TYPE, input)
}
