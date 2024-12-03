package irwg

import (
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils"
	"github.com/consensys/gnark/constraint"
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
	o.AppendIntSlice(c.Constraints)
	o.AppendIntSlice(c.Outputs)
	o.AppendUint64(uint64(c.NumInputs))
}

func serializeRootCircuit(o *utils.OutputBuf, c *RootCircuit, field field.Field) {
	o.AppendUint64(uint64(len(c.Circuits)))
	for k, c := range c.Circuits {
		o.AppendUint64(uint64(k))
		serializeCircuit(o, c, field)
	}
}

func SerializeRootCircuit(c *RootCircuit) []byte {
	o := &utils.OutputBuf{}
	o.AppendUint64(field.GetFieldId(c.Field))
	o.AppendUint64(uint64(c.NumPublicInputs))
	o.AppendUint64(uint64(c.ExpectedNumOutputZeroes))
	serializeRootCircuit(o, c, c.Field)
	return o.Bytes()
}

func (c *RootCircuit) Serialize() []byte {
	return SerializeRootCircuit(c)
}

func deserializeInstruction(field field.Field, i *utils.InputBuf) Instruction {
	var ins Instruction
	ins.Type = InstructionType(i.ReadUint8())
	switch ins.Type {
	case LinComb:
		n := i.ReadUint64()
		ins.Inputs = make([]int, n)
		for j := uint64(0); j < n; j++ {
			ins.Inputs[j] = int(i.ReadUint64())
		}
		ins.LinCombCoef = make([]constraint.Element, n)
		for j := uint64(0); j < n; j++ {
			ins.LinCombCoef[j] = i.ReadFieldElement(field)
		}
		ins.Const = i.ReadFieldElement(field)
	case Mul:
		ins.Inputs = i.ReadIntSlice()
	case Hint:
		ins.ExtraId = i.ReadUint64()
		ins.Inputs = i.ReadIntSlice()
		ins.NumOutputs = int(i.ReadUint64())
	case ConstantLike:
		typ := i.ReadUint8()
		if typ == 1 {
			ins.ExtraId = 0
			ins.Const = i.ReadFieldElement(field)
		} else if typ == 2 {
			ins.ExtraId = 1
		} else {
			ins.ExtraId = 2 + i.ReadUint64()
		}
	case SubCircuitCall:
		ins.ExtraId = i.ReadUint64()
		ins.Inputs = i.ReadIntSlice()
		ins.NumOutputs = int(i.ReadUint64())
	case CustomGate:
		ins.ExtraId = i.ReadUint64()
		ins.Inputs = i.ReadIntSlice()
	}
	return ins
}

func deserializeCircuit(field field.Field, i *utils.InputBuf) *Circuit {
	var c Circuit
	n := i.ReadUint64()
	c.Instructions = make([]Instruction, n)
	for j := uint64(0); j < n; j++ {
		c.Instructions[j] = deserializeInstruction(field, i)
	}
	c.Constraints = i.ReadIntSlice()
	c.Outputs = i.ReadIntSlice()
	c.NumInputs = int(i.ReadUint64())
	return &c
}

func deserializeRootCircuit(field field.Field, i *utils.InputBuf) *RootCircuit {
	var rc RootCircuit
	rc.NumPublicInputs = int(i.ReadUint64())
	rc.ExpectedNumOutputZeroes = int(i.ReadUint64())
	n := i.ReadUint64()
	rc.Circuits = make(map[uint64]*Circuit)
	for j := uint64(0); j < n; j++ {
		k := i.ReadUint64()
		rc.Circuits[k] = deserializeCircuit(field, i)
	}
	rc.Field = field
	return &rc
}

func DeserializeRootCircuit(buf []byte) *RootCircuit {
	i := utils.NewInputBuf(buf)
	fieldId := i.ReadUint64()
	field := field.GetFieldById(fieldId)
	rc := deserializeRootCircuit(field, i)
	if !i.IsEnd() {
		panic("invalid binary format")
	}
	return rc
}
