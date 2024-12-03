package irwg

import (
	"errors"
	"math/big"
	"reflect"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils/customgates"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/schema"
)

// Witness represents the solved values of the circuit's inputs.
type Witness struct {
	NumWitnesses              int
	NumInputsPerWitness       int
	NumPublicInputsPerWitness int
	Field                     *big.Int
	Values                    []*big.Int
}

var TVariable reflect.Type

func init() {
	TVariable = reflect.ValueOf(struct{ A frontend.Variable }{}).FieldByName("A").Type()
}

// GetCircuitVariables reimplements frontend.NewWitness to support fields that are not present in gnark.
func GetCircuitVariables(assignment frontend.Circuit, field field.Field) ([]constraint.Element, []constraint.Element) {
	chPubValues := make(chan any)
	chSecValues := make(chan any)
	go func() {
		schema.Walk(assignment, TVariable, func(leaf schema.LeafInfo, tValue reflect.Value) error {
			if leaf.Visibility == schema.Public {
				chPubValues <- tValue.Interface()
			}
			return nil
		})
		close(chPubValues)
		schema.Walk(assignment, TVariable, func(leaf schema.LeafInfo, tValue reflect.Value) error {
			if leaf.Visibility == schema.Secret {
				chSecValues <- tValue.Interface()
			}
			return nil
		})
		close(chSecValues)
	}()
	resPub := []constraint.Element{}
	for v := range chPubValues {
		resPub = append(resPub, field.FromInterface(v))
	}
	resSec := []constraint.Element{}
	for v := range chSecValues {
		resSec = append(resSec, field.FromInterface(v))
	}
	return resPub, resSec
}

func (rc *RootCircuit) solveInput(assignment frontend.Circuit) ([]*big.Int, int, int, error) {
	vecPub, vecSec := GetCircuitVariables(assignment, rc.Field)
	res, err := rc.eval(vecSec, vecPub)
	if err != nil {
		return nil, 0, 0, err
	}
	witness := make([]*big.Int, len(res)+len(vecPub))
	for i, x := range res {
		witness[i] = rc.Field.ToBigInt(x)
	}
	for i, x := range vecPub {
		witness[i+len(res)] = rc.Field.ToBigInt(x)
	}
	return witness, len(res), len(vecPub), nil
}

// SolveInput is the entry point to solve the final input of the given assignment using a specified number of threads.
func (rc *RootCircuit) SolveInput(assignment frontend.Circuit, _ int) (*Witness, error) {
	return rc.SolveInputAuto(assignment)
}

func (rc *RootCircuit) SolveInputAuto(assignment frontend.Circuit) (*Witness, error) {
	witness, lenSec, lenPub, err := rc.solveInput(assignment)
	if err != nil {
		return nil, err
	}
	return &Witness{
		NumWitnesses:              1,
		NumInputsPerWitness:       lenSec,
		NumPublicInputsPerWitness: lenPub,
		Field:                     rc.Field.Field(),
		Values:                    witness,
	}, nil
}

func (rc *RootCircuit) SolveInputs(assignments []frontend.Circuit) (*Witness, error) {
	witnesses := []*big.Int{}
	witness := []*big.Int{}
	var err error
	lenSec := 0
	lenPub := 0
	for _, assignment := range assignments {
		witness, lenSec, lenPub, err = rc.solveInput(assignment)
		if err != nil {
			return nil, err
		}
		witnesses = append(witnesses, witness...)
	}
	return &Witness{
		NumWitnesses:              len(assignments),
		NumInputsPerWitness:       lenSec,
		NumPublicInputsPerWitness: lenPub,
		Field:                     rc.Field.Field(),
		Values:                    witnesses,
	}, nil
}

func (rc *RootCircuit) eval(inputs []constraint.Element, publicInputs []constraint.Element) ([]constraint.Element, error) {
	return rc.evalSub(0, inputs, publicInputs)
}

func (rc *RootCircuit) evalSub(circuitId uint64, inputs []constraint.Element, publicInputs []constraint.Element) ([]constraint.Element, error) {
	values := append([]constraint.Element{{}}, inputs...)
	for _, insn := range rc.Circuits[circuitId].Instructions {
		switch insn.Type {
		case LinComb:
			res := insn.Const
			for i, x := range insn.Inputs {
				res = rc.Field.Add(res, rc.Field.Mul(values[x], insn.LinCombCoef[i]))
			}
			values = append(values, res)
		case Mul:
			res := rc.Field.One()
			for _, x := range insn.Inputs {
				res = rc.Field.Mul(res, values[x])
			}
			values = append(values, res)
		case Hint:
			hint_inputs := []*big.Int{}
			for _, x := range insn.Inputs {
				hint_inputs = append(hint_inputs, rc.Field.ToBigInt(values[x]))
			}
			hint_outputs := make([]*big.Int, insn.NumOutputs)
			for i := range hint_outputs {
				hint_outputs[i] = big.NewInt(0)
			}
			err := callHint(insn.ExtraId, rc.Field.Field(), hint_inputs, hint_outputs)
			if err != nil {
				return nil, err
			}
			for _, x := range hint_outputs {
				values = append(values, rc.Field.FromInterface(x))
			}
		case ConstantLike:
			if insn.ExtraId == 0 {
				values = append(values, insn.Const)
			} else if insn.ExtraId == 1 {
				return nil, errors.New("random constant not supported")
			} else {
				values = append(values, publicInputs[insn.ExtraId-2])
			}
		case SubCircuitCall:
			sub_inputs := []constraint.Element{}
			for _, x := range insn.Inputs {
				sub_inputs = append(sub_inputs, values[x])
			}
			sub_outputs, err := rc.evalSub(insn.ExtraId, sub_inputs, publicInputs)
			if err != nil {
				return nil, err
			}
			values = append(values, sub_outputs...)
		case CustomGate:
			custom_inputs := []*big.Int{}
			for _, x := range insn.Inputs {
				custom_inputs = append(custom_inputs, rc.Field.ToBigInt(values[x]))
			}
			custom_outputs := make([]*big.Int, 1)
			err := customgates.GetFunc(insn.ExtraId)(rc.Field.Field(), custom_inputs, custom_outputs)
			if err != nil {
				return nil, err
			}
			values = append(values, rc.Field.FromInterface(custom_outputs[0]))
		}
	}
	outputs := []constraint.Element{}
	for _, x := range rc.Circuits[circuitId].Outputs {
		outputs = append(outputs, values[x])
	}
	return outputs, nil
}

func callHint(hintId uint64, field *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	// The only required builtin hint (Div)
	if hintId == 0xCCC000000001 {
		if len(inputs) != 2 || len(outputs) != 1 {
			return errors.New("Div hint requires 2 inputs and 1 output")
		}
		x := (&big.Int{}).Mod(inputs[0], field)
		y := (&big.Int{}).Mod(inputs[1], field)
		if y.Cmp(big.NewInt(0)) == 0 {
			outputs[0] = big.NewInt(0)
			return nil
		}
		a := (&big.Int{}).ModInverse(y, field)
		a.Mul(a, x)
		a.Mod(a, field)
		outputs[0] = a
		return nil
	}
	return solver.GetRegisteredHint(solver.HintID(hintId))(field, inputs, outputs)
}

// Serialize converts the Witness into a byte slice for storage or transmission.
func (w *Witness) Serialize() []byte {
	o := utils.OutputBuf{}
	o.AppendUint64(uint64(w.NumWitnesses))
	o.AppendUint64(uint64(w.NumInputsPerWitness))
	o.AppendUint64(uint64(w.NumPublicInputsPerWitness))
	o.AppendBigInt(32, w.Field)
	bnlen := field.GetFieldFromOrder(w.Field).SerializedLen()
	for _, x := range w.Values {
		o.AppendBigInt(bnlen, x)
	}
	return o.Bytes()
}
