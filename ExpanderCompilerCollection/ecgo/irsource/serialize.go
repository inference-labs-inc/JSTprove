package irsource

import (
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils"
)

func serializeInstruction(o *utils.OutputBuf, i *Instruction, field field.Field) {
	o.AppendUint8(uint8(i.Type))
	switch i.Type {
	case LinComb:
		if len(i.Inputs) != len(i.LinCombCoef) {
			panic("gg")
		}
		o.AppendUint64(uint64(len(i.Inputs)))
		for _, x := range i.Inputs {
			o.AppendUint64(uint64(x))
		}
		for _, x := range i.LinCombCoef {
			o.AppendFieldElement(field, x)
		}
		o.AppendFieldElement(field, i.Const)
	case Mul:
		o.AppendIntSlice(i.Inputs)
	case Div:
		o.AppendUint64(uint64(i.X))
		o.AppendUint64(uint64(i.Y))
		o.AppendUint8(uint8(i.ExtraId))
	case BoolBinOp:
		o.AppendUint64(uint64(i.X))
		o.AppendUint64(uint64(i.Y))
		o.AppendUint8(uint8(i.ExtraId))
	case IsZero:
		o.AppendUint64(uint64(i.X))
	case Commit:
		o.AppendIntSlice(i.Inputs)
	case Hint:
		o.AppendUint64(uint64(i.ExtraId))
		o.AppendIntSlice(i.Inputs)
		o.AppendUint64(uint64(i.NumOutputs))
	case ConstantLike:
		if i.ExtraId == 0 {
			o.AppendUint8(1)
			o.AppendFieldElement(field, i.Const)
		} else if i.ExtraId == 1 {
			o.AppendUint8(2)
		} else {
			o.AppendUint8(3)
			o.AppendUint64(uint64(i.ExtraId) - 2)
		}
	case SubCircuitCall:
		o.AppendUint64(uint64(i.ExtraId))
		o.AppendIntSlice(i.Inputs)
		o.AppendUint64(uint64(i.NumOutputs))
	case UnconstrainedBinOp:
		panic("no unconstrained binop in gnark")
	case UnconstrainedSelect:
		panic("no unconstrained select in gnark")
	case CustomGate:
		o.AppendUint64(uint64(i.ExtraId))
		o.AppendIntSlice(i.Inputs)
	}
}

func serializeCircuit(o *utils.OutputBuf, c *Circuit, field field.Field) {
	o.AppendUint64(uint64(len(c.Instructions)))
	for _, i := range c.Instructions {
		serializeInstruction(o, &i, field)
	}
	o.AppendUint64(uint64(len(c.Constraints)))
	for _, i := range c.Constraints {
		o.AppendUint8(uint8(i.Typ))
		o.AppendUint64(uint64(i.Var))
	}
	o.AppendIntSlice(c.Outputs)
	o.AppendUint64(uint64(c.NumInputs))
}

func serializeRootCircuit(o *utils.OutputBuf, c *RootCircuit, field field.Field) {
	o.AppendUint64(uint64(c.NumPublicInputs))
	o.AppendUint64(uint64(c.ExpectedNumOutputZeroes))
	o.AppendUint64(uint64(len(c.Circuits)))
	for k, c := range c.Circuits {
		o.AppendUint64(uint64(k))
		serializeCircuit(o, c, field)
	}
}

func SerializeRootCircuit(c *RootCircuit) []byte {
	o := &utils.OutputBuf{}
	o.AppendUint64(field.GetFieldId(c.Field))
	serializeRootCircuit(o, c, c.Field)
	return o.Bytes()
}
