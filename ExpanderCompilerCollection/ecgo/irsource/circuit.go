package irsource

import "github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field"

type Circuit struct {
	Instructions []Instruction
	Constraints  []Constraint
	Outputs      []int
	NumInputs    int
}

type RootCircuit struct {
	NumPublicInputs         int
	ExpectedNumOutputZeroes int
	Circuits                map[uint64]*Circuit
	Field                   field.Field
}
