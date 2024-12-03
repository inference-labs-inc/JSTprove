package poseidon

import "math/rand"

type PoseidonParams struct {
	// number of full rounds
	NumFullRounds int
	// number of half full rounds
	NumHalfFullRounds int
	// number of partial rounds
	NumPartRounds int
	// number of half full rounds
	NumHalfPartialRounds int
	// number of states
	NumStates int
	// mds matrix
	MdsMatrix [][]uint32
	// external round constants
	ExternalRoundConstant [][]uint32
	// internal round constants
	InternalRoundConstant []uint32
}

// TODOs: the parameters are not secure. use a better way to generate the constants
func NewPoseidonParams() *PoseidonParams {
	r := rand.New(rand.NewSource(42))

	num_full_rounds := 8
	num_part_rounds := 14
	num_states := 16

	external_round_constant := make([][]uint32, num_states)
	for i := 0; i < num_states; i++ {
		external_round_constant[i] = make([]uint32, num_full_rounds)
		for j := 0; j < num_full_rounds; j++ {
			external_round_constant[i][j] = randomM31(r)
		}
	}

	internal_round_constant := make([]uint32, num_part_rounds)
	for i := 0; i < num_part_rounds; i++ {
		internal_round_constant[i] = randomM31(r)
	}

	// mds parameters adopted from Plonky3
	// https://github.com/Plonky3/Plonky3/blob/eeb4e37b20127c4daa871b2bad0df30a7c7380db/mersenne-31/src/mds.rs#L176
	mds := make([][]uint32, num_states)
	mds[0] = make([]uint32, 16)
	mds[0] = []uint32{1, 1, 51, 1, 11, 17, 2, 1, 101, 63, 15, 2, 67, 22, 13, 3}
	for i := 1; i < 16; i++ {
		mds[i] = make([]uint32, 16)
		// cyclic rotation of the first row
		for j := 0; j < 16; j++ {
			mds[i][j] = mds[0][(j+i)%16]
		}

	}

	return &PoseidonParams{
		NumFullRounds:         num_full_rounds,
		NumHalfFullRounds:     num_full_rounds / 2,
		NumPartRounds:         num_part_rounds,
		NumHalfPartialRounds:  num_part_rounds / 2,
		NumStates:             num_states,
		MdsMatrix:             mds,
		ExternalRoundConstant: external_round_constant,
		InternalRoundConstant: internal_round_constant,
	}
}

func randomM31(r *rand.Rand) uint32 {
	t := r.Uint32() & 0x7FFFFFFF

	for t == 0x7fffffff {
		t = rand.Uint32() & 0x7FFFFFFF
	}

	return t
}
