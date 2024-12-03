package layered

import (
	"math/big"
	"os"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils"
)

const MAGIC = 3914834606642317635

func serializeCoef(o *utils.OutputBuf, bnlen int, coef *big.Int, coefType uint8, publicInputId uint64) {
	if coefType == 1 {
		o.AppendUint8(1)
		o.AppendBigInt(bnlen, coef)
	} else if coefType == 2 {
		o.AppendUint8(2)
	} else {
		o.AppendUint8(3)
		o.AppendUint64(publicInputId)
	}
}

func deserializeCoef(in *utils.InputBuf, bnlen int) (*big.Int, uint8, uint64) {
	coefType := in.ReadUint8()
	if coefType == 1 {
		return in.ReadBigInt(bnlen), 1, 0
	} else if coefType == 2 {
		return big.NewInt(0), 2, 0
	} else {
		return big.NewInt(0), 3, in.ReadUint64()
	}
}

// Serialize converts a RootCircuit into a byte array for storage or transmission.
func (rc *RootCircuit) Serialize() []byte {
	bnlen := field.GetFieldFromOrder(rc.Field).SerializedLen()
	o := utils.OutputBuf{}
	o.AppendUint64(MAGIC)
	o.AppendBigInt(32, rc.Field)
	o.AppendUint64(uint64(rc.NumPublicInputs))
	o.AppendUint64(uint64(rc.NumActualOutputs))
	o.AppendUint64(uint64(rc.ExpectedNumOutputZeroes))
	o.AppendUint64(uint64(len(rc.Circuits)))
	for _, c := range rc.Circuits {
		o.AppendUint64(c.InputLen)
		o.AppendUint64(c.OutputLen)
		o.AppendUint64(uint64(len(c.SubCircuits)))
		for _, sub := range c.SubCircuits {
			o.AppendUint64(sub.Id)
			o.AppendUint64(uint64(len(sub.Allocations)))
			for _, a := range sub.Allocations {
				o.AppendUint64(a.InputOffset)
				o.AppendUint64(a.OutputOffset)
			}
		}
		o.AppendUint64(uint64(len(c.Mul)))
		for _, m := range c.Mul {
			o.AppendUint64(m.In0)
			o.AppendUint64(m.In1)
			o.AppendUint64(m.Out)
			serializeCoef(&o, bnlen, m.Coef, m.CoefType, m.PublicInputId)
		}
		o.AppendUint64(uint64(len(c.Add)))
		for _, a := range c.Add {
			o.AppendUint64(a.In)
			o.AppendUint64(a.Out)
			serializeCoef(&o, bnlen, a.Coef, a.CoefType, a.PublicInputId)
		}
		o.AppendUint64(uint64(len(c.Cst)))
		for _, cst := range c.Cst {
			o.AppendUint64(cst.Out)
			serializeCoef(&o, bnlen, cst.Coef, cst.CoefType, cst.PublicInputId)
		}
		o.AppendUint64(uint64(len(c.Custom)))
		for _, cu := range c.Custom {
			o.AppendUint64(cu.GateType)
			o.AppendUint64(uint64(len(cu.In)))
			for _, in := range cu.In {
				o.AppendUint64(in)
			}
			o.AppendUint64(cu.Out)
			serializeCoef(&o, bnlen, cu.Coef, cu.CoefType, cu.PublicInputId)
		}
	}
	o.AppendUint64(uint64(len(rc.Layers)))
	for _, l := range rc.Layers {
		o.AppendUint64(l)
	}
	return o.Bytes()
}

func DeserializeRootCircuit(buf []byte) *RootCircuit {
	in := utils.NewInputBuf(buf)
	if in.ReadUint64() != MAGIC {
		panic("invalid file header")
	}
	rc := &RootCircuit{}
	rc.Field = in.ReadBigInt(32)
	rc.NumPublicInputs = int(in.ReadUint64())
	rc.NumActualOutputs = int(in.ReadUint64())
	rc.ExpectedNumOutputZeroes = int(in.ReadUint64())
	bnlen := field.GetFieldFromOrder(rc.Field).SerializedLen()
	nbCircuits := in.ReadUint64()
	rc.Circuits = make([]*Circuit, nbCircuits)
	for i := uint64(0); i < nbCircuits; i++ {
		c := &Circuit{}
		c.InputLen = in.ReadUint64()
		c.OutputLen = in.ReadUint64()
		nbSubCircuits := in.ReadUint64()
		c.SubCircuits = make([]SubCircuit, nbSubCircuits)
		for j := uint64(0); j < nbSubCircuits; j++ {
			sub := SubCircuit{}
			sub.Id = in.ReadUint64()
			nbAllocations := in.ReadUint64()
			sub.Allocations = make([]Allocation, nbAllocations)
			for k := uint64(0); k < nbAllocations; k++ {
				sub.Allocations[k].InputOffset = in.ReadUint64()
				sub.Allocations[k].OutputOffset = in.ReadUint64()
			}
			c.SubCircuits[j] = sub
		}
		nbMul := in.ReadUint64()
		c.Mul = make([]GateMul, nbMul)
		for j := uint64(0); j < nbMul; j++ {
			c.Mul[j].In0 = in.ReadUint64()
			c.Mul[j].In1 = in.ReadUint64()
			c.Mul[j].Out = in.ReadUint64()
			c.Mul[j].Coef, c.Mul[j].CoefType, c.Mul[j].PublicInputId = deserializeCoef(in, bnlen)
		}
		nbAdd := in.ReadUint64()
		c.Add = make([]GateAdd, nbAdd)
		for j := uint64(0); j < nbAdd; j++ {
			c.Add[j].In = in.ReadUint64()
			c.Add[j].Out = in.ReadUint64()
			c.Add[j].Coef, c.Add[j].CoefType, c.Add[j].PublicInputId = deserializeCoef(in, bnlen)
		}
		nbCst := in.ReadUint64()
		c.Cst = make([]GateCst, nbCst)
		for j := uint64(0); j < nbCst; j++ {
			c.Cst[j].Out = in.ReadUint64()
			c.Cst[j].Coef, c.Cst[j].CoefType, c.Cst[j].PublicInputId = deserializeCoef(in, bnlen)
		}
		nbCustom := in.ReadUint64()
		c.Custom = make([]GateCustom, nbCustom)
		for j := uint64(0); j < nbCustom; j++ {
			c.Custom[j].GateType = in.ReadUint64()
			nbIn := in.ReadUint64()
			c.Custom[j].In = make([]uint64, nbIn)
			for k := uint64(0); k < nbIn; k++ {
				c.Custom[j].In[k] = in.ReadUint64()
			}
			c.Custom[j].Out = in.ReadUint64()
			c.Custom[j].Coef, c.Custom[j].CoefType, c.Custom[j].PublicInputId = deserializeCoef(in, bnlen)
		}
		rc.Circuits[i] = c
	}
	nbLayers := in.ReadUint64()
	rc.Layers = make([]uint64, nbLayers)
	for i := uint64(0); i < nbLayers; i++ {
		rc.Layers[i] = in.ReadUint64()
	}
	if !in.IsEnd() {
		panic("invalid binary format")
	}
	return rc
}

func DetectFieldIdFromFile(fn string) uint64 {
	// Read the first 4 bytes of the file
	file, err := os.Open(fn)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	buf := make([]byte, 40)
	n, err := file.Read(buf)
	if err != nil || n != 40 {
		panic(err)
	}
	in := utils.NewInputBuf(buf)
	if in.ReadUint64() != MAGIC {
		panic("invalid file header")
	}
	f := in.ReadBigInt(32)
	return field.GetFieldId(field.GetFieldFromOrder(f))
}
