package poseidon

import (
	"testing"

	"github.com/consensys/gnark/constraint"
	"github.com/stretchr/testify/assert"
)

func TestPoseidon(t *testing.T) {
	param := NewPoseidonParams()

	state := make([]constraint.Element, param.NumStates)
	PoseidonM31(param, state)

	state = make([]constraint.Element, param.NumStates+1)
	assert.Panics(t, func() { PoseidonM31(param, state) })
}
