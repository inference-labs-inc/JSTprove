// Some content of this file is copied from gnark/frontend/cs/r1cs/api.go

package builder

import (
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/irsource"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/bits"
)

func init() {
	solver.RegisterHint(IdentityHint)
}

// API defines a set of methods for interacting with the circuit builder.
type API interface {
	// ToSingleVariable converts an expression to a single base variable.
	ToSingleVariable(frontend.Variable) frontend.Variable
	// Output adds a variable to the circuit's output.
	Output(frontend.Variable)
	// LayerOf returns an approximation of the layer in which a variable will be placed after compilation.
	LayerOf(frontend.Variable) int // For debug usage.
	// ToFirstLayer uses a hint to pull a variable back to the first layer.
	ToFirstLayer(frontend.Variable) frontend.Variable
	// GetRandomValue returns a random value for use within the circuit.
	GetRandomValue() frontend.Variable
	// CustomGate registers a hint, but it compiles to a custom gate in the layered circuit.
	CustomGate(gateType uint64, inputs ...frontend.Variable) frontend.Variable
}

// ---------------------------------------------------------------------------------------------
// Arithmetic

// Add computes the sum i1+i2+...in and returns the result.
func (builder *builder) Add(i1, i2 frontend.Variable, in ...frontend.Variable) frontend.Variable {
	// extract frontend.Variables from input
	vars := builder.toVariableIds(append([]frontend.Variable{i1, i2}, in...)...)
	return builder.add(vars, false)
}

func (builder *builder) MulAcc(a, b, c frontend.Variable) frontend.Variable {
	return builder.Add(builder.Mul(b, c), a)
}

// Sub computes the difference between the given variables.
// When more than two variables are provided, the difference is computed as i1 - Σ(i2...).
func (builder *builder) Sub(i1, i2 frontend.Variable, in ...frontend.Variable) frontend.Variable {
	vars := builder.toVariableIds(append([]frontend.Variable{i1, i2}, in...)...)
	return builder.add(vars, true)
}

// returns res = Σ(vars) or res = vars[0] - Σ(vars[1:]) if sub == true.
func (builder *builder) add(vars []int, sub bool) frontend.Variable {
	coef := make([]constraint.Element, len(vars))
	coef[0] = builder.tOne
	if sub {
		for i := 1; i < len(vars); i++ {
			coef[i] = builder.field.Neg(builder.tOne)
		}
	} else {
		for i := 1; i < len(vars); i++ {
			coef[i] = builder.tOne
		}
	}
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:        irsource.LinComb,
		Inputs:      vars,
		LinCombCoef: coef,
	})
	return builder.addVar()
}

// Neg returns the negation of the given variable.
func (builder *builder) Neg(i frontend.Variable) frontend.Variable {
	v := builder.toVariableId(i)
	coef := []constraint.Element{builder.field.Neg(builder.tOne)}
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:        irsource.LinComb,
		Inputs:      []int{v},
		LinCombCoef: coef,
	})
	return builder.addVar()
}

// Mul computes the product of the given variables.
func (builder *builder) Mul(i1, i2 frontend.Variable, in ...frontend.Variable) frontend.Variable {
	vars := builder.toVariableIds(append([]frontend.Variable{i1, i2}, in...)...)
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:   irsource.Mul,
		Inputs: vars,
	})
	return builder.addVar()
}

// DivUnchecked returns i1 divided by i2 and returns 0 if both i1 and i2 are zero.
func (builder *builder) DivUnchecked(i1, i2 frontend.Variable) frontend.Variable {
	vars := builder.toVariableIds(i1, i2)
	v1 := vars[0]
	v2 := vars[1]
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:    irsource.Div,
		X:       v1,
		Y:       v2,
		ExtraId: 1,
	})
	return builder.addVar()
}

// Div returns the result of i1 divided by i2.
func (builder *builder) Div(i1, i2 frontend.Variable) frontend.Variable {
	vars := builder.toVariableIds(i1, i2)
	v1 := vars[0]
	v2 := vars[1]
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:    irsource.Div,
		X:       v1,
		Y:       v2,
		ExtraId: 0,
	})
	return builder.addVar()
}

// Inverse returns the multiplicative inverse of the given variable.
func (builder *builder) Inverse(i1 frontend.Variable) frontend.Variable {
	return builder.Div(1, i1)
}

// ---------------------------------------------------------------------------------------------
// Bit operations

// ToBinary unpacks a frontend.Variable in binary,
// n is the number of bits to select (starting from lsb)
// n default value is fr.Bits the number of bits needed to represent a field element
//
// The result in in little endian (first bit= lsb)
func (builder *builder) ToBinary(i1 frontend.Variable, n ...int) []frontend.Variable {
	// TODO: Currently this doesn't support MarkBoolean
	nbBits := builder.field.FieldBitLen()
	if len(n) == 1 {
		nbBits = n[0]
		if nbBits < 0 {
			panic("invalid n")
		}
	} else {
		panic("only one argument is supported")
	}

	return bits.ToBinary(builder, i1, bits.WithNbDigits(nbBits))
}

// FromBinary packs the given variables, seen as a fr.Element in little endian, into a single variable.
func (builder *builder) FromBinary(_b ...frontend.Variable) frontend.Variable {
	return bits.FromBinary(builder, _b)
}

// Xor computes the logical XOR between two frontend.Variables.
func (builder *builder) Xor(_a, _b frontend.Variable) frontend.Variable {
	vars := builder.toVariableIds(_a, _b)
	a := vars[0]
	b := vars[1]
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:    irsource.BoolBinOp,
		X:       a,
		Y:       b,
		ExtraId: 1,
	})
	return builder.addVar()
}

// Or computes the logical OR between two frontend.Variables.
func (builder *builder) Or(_a, _b frontend.Variable) frontend.Variable {
	vars := builder.toVariableIds(_a, _b)
	a := vars[0]
	b := vars[1]
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:    irsource.BoolBinOp,
		X:       a,
		Y:       b,
		ExtraId: 2,
	})
	return builder.addVar()
}

// And computes the logical AND between two frontend.Variables.
func (builder *builder) And(_a, _b frontend.Variable) frontend.Variable {
	vars := builder.toVariableIds(_a, _b)
	a := vars[0]
	b := vars[1]
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:    irsource.BoolBinOp,
		X:       a,
		Y:       b,
		ExtraId: 3,
	})
	return builder.addVar()
}

// ---------------------------------------------------------------------------------------------
// Conditionals

// Select yields the second variable if the first is true, otherwise yields the third variable.
func (builder *builder) Select(i0, i1, i2 frontend.Variable) frontend.Variable {
	cond := i0

	// ensures that cond is boolean
	builder.AssertIsBoolean(cond)

	v := builder.Sub(i1, i2) // no constraint is recorded
	w := builder.Mul(cond, v)
	return builder.Add(w, i2)
}

// Lookup2 performs a 2-bit lookup based on the given bits and values.
func (builder *builder) Lookup2(b0, b1 frontend.Variable, i0, i1, i2, i3 frontend.Variable) frontend.Variable {
	vars := []frontend.Variable{b0, b1, i0, i1, i2, i3}
	s0, s1 := vars[0], vars[1]
	in0, in1, in2, in3 := vars[2], vars[3], vars[4], vars[5]

	// ensure that bits are actually bits. Adds no constraints if the variables
	// are already constrained.
	builder.AssertIsBoolean(s0)
	builder.AssertIsBoolean(s1)

	// two-bit lookup for the general case can be done with three constraints as
	// following:
	//    (1) (in3 - in2 - in1 + in0) * s1 = tmp1 - in1 + in0
	//    (2) tmp1 * s0 = tmp2
	//    (3) (in2 - in0) * s1 = RES - tmp2 - in0
	// the variables tmp1 and tmp2 are new internal variables and the variables
	// RES will be the returned result

	tmp1 := builder.Add(in3, in0)
	tmp1 = builder.Sub(tmp1, in2, in1)
	tmp1 = builder.Mul(tmp1, s1)
	tmp1 = builder.Add(tmp1, in1)
	tmp1 = builder.Sub(tmp1, in0) // (1) tmp1 = s1 * (in3 - in2 - in1 + in0) + in1 - in0
	tmp2 := builder.Mul(tmp1, s0) // (2) tmp2 = tmp1 * s0
	res := builder.Sub(in2, in0)
	res = builder.Mul(res, s1)
	res = builder.Add(res, tmp2, in0) // (3) res = (v2 - v0) * s1 + tmp2 + in0
	return res
}

// IsZero returns 1 if the given variable is zero, otherwise returns 0.
func (builder *builder) IsZero(i1 frontend.Variable) frontend.Variable {
	a := builder.toVariableId(i1)
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type: irsource.IsZero,
		X:    a,
	})
	return builder.addVar()
}

// Cmp compares i1 and i2 and returns 1 if i1>i2, 0 if i1=i2, -1 if i1<i2.
func (builder *builder) Cmp(i1, i2 frontend.Variable) frontend.Variable {
	nbBits := builder.field.FieldBitLen()
	// in AssertIsLessOrEq we omitted comparison against modulus for the left
	// side as if `a+r<b` implies `a<b`, then here we compute the inequality
	// directly.
	bi1 := bits.ToBinary(builder, i1, bits.WithNbDigits(nbBits))
	bi2 := bits.ToBinary(builder, i2, bits.WithNbDigits(nbBits))

	res := newVariable(builder.toVariableId(0))

	for i := builder.field.FieldBitLen() - 1; i >= 0; i-- {

		iszeroi1 := builder.IsZero(bi1[i])
		iszeroi2 := builder.IsZero(bi2[i])

		i1i2 := builder.And(bi1[i], iszeroi2)
		i2i1 := builder.And(bi2[i], iszeroi1)

		n := builder.Select(i2i1, -1, 0)
		m := builder.Select(i1i2, 1, n)

		res = newVariable(builder.toVariableId(builder.Select(builder.IsZero(res), m, res)))

	}
	return res
}

// Println is not implemented and will panic if called.
func (builder *builder) Println(a ...frontend.Variable) {
	panic("unimplemented")
}

// Compiler returns itself as it implements the frontend.Compiler interface.
func (builder *builder) Compiler() frontend.Compiler {
	return builder
}

// Commit is faulty in its current implementation as it merely returns a compile-time random number.
func (builder *builder) Commit(v ...frontend.Variable) (frontend.Variable, error) {
	vars := builder.toVariableIds(v...)
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:   irsource.Commit,
		Inputs: vars,
	})
	return builder.addVar(), nil
}

// SetGkrInfo is not implemented and will panic if called.
func (builder *builder) SetGkrInfo(info constraint.GkrInfo) error {
	panic("unimplemented")
}

// Output adds the given variable to the circuit's output.
func (builder *builder) Output(x_ frontend.Variable) {
	if builder.root.builder != builder {
		panic("Output can only be called on root circuit")
	}
	x := builder.toVariableId(x_)
	builder.output = append(builder.output, x)
}
