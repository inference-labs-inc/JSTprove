package test

import (
	"testing"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/consensys/gnark/frontend"
)

type Assert struct {
	t *testing.T
}

func NewAssert(t *testing.T) *Assert {
	return &Assert{t: t}
}

func (a *Assert) ProveSucceeded(cr *ecgo.CompileResult, assignment frontend.Circuit) {
	lc := cr.GetLayeredCircuit()
	is := cr.GetInputSolver()
	witness, err := is.SolveInput(assignment, 1)
	if err != nil {
		a.t.Fatal(err)
	}
	if !CheckCircuit(lc, witness) {
		a.t.Fatal("should succeed")
	}
}

func (a *Assert) ProveFailed(cr *ecgo.CompileResult, assignment frontend.Circuit) {
	lc := cr.GetLayeredCircuit()
	is := cr.GetInputSolver()
	witness, err := is.SolveInput(assignment, 1)
	if err != nil {
		a.t.Fatal(err)
	}
	if CheckCircuit(lc, witness) {
		a.t.Fatal("should fail")
	}
}
