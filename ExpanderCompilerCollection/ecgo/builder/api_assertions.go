// Some content of this file is copied from gnark/frontend/cs/r1cs/api_assertions.go

package builder

import (
	"math/big"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/irsource"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/std/math/bits"
)

// AssertIsEqual adds an assertion that i1 is equal to i2.
func (builder *builder) AssertIsEqual(i1, i2 frontend.Variable) {
	x := builder.toVariableId(builder.Sub(i1, i2))
	builder.constraints = append(builder.constraints, irsource.Constraint{
		Typ: irsource.Zero,
		Var: x,
	})
}

// AssertIsDifferent constrains i1 and i2 to have different values.
func (builder *builder) AssertIsDifferent(i1, i2 frontend.Variable) {
	x := builder.toVariableId(builder.Sub(i1, i2))
	builder.constraints = append(builder.constraints, irsource.Constraint{
		Typ: irsource.NonZero,
		Var: x,
	})
}

// AssertIsBoolean adds an assertion that the variable is either 0 or 1.
func (builder *builder) AssertIsBoolean(i1 frontend.Variable) {
	x := builder.toVariableId(i1)
	builder.constraints = append(builder.constraints, irsource.Constraint{
		Typ: irsource.Bool,
		Var: x,
	})
}

// AssertIsCrumb adds an assertion that the variable is a 2-bit value, also known as a crumb.
func (builder *builder) AssertIsCrumb(i1 frontend.Variable) {
	i1 = builder.MulAcc(builder.Mul(-3, i1), i1, i1)
	i1 = builder.MulAcc(builder.Mul(2, i1), i1, i1)
	builder.AssertIsEqual(i1, 0)
}

// AssertIsLessOrEqual adds an assertion that v is less than or equal to bound.
func (builder *builder) AssertIsLessOrEqual(v frontend.Variable, bound frontend.Variable) {
	builder.mustBeLessOrEqVar(v, bound)
}

func (builder *builder) mustBeLessOrEqVar(a, bound frontend.Variable) {
	// here bound is NOT a constant,
	// but a can be either constant or a wire.

	nbBits := builder.field.FieldBitLen()

	aBits := bits.ToBinary(builder, a, bits.WithNbDigits(nbBits), bits.WithUnconstrainedOutputs(), bits.OmitModulusCheck())
	boundBits := bits.ToBinary(builder, bound, bits.WithNbDigits(nbBits))

	p := make([]frontend.Variable, nbBits+1)
	p[nbBits] = newVariable(builder.toVariableId(1))

	zero := newVariable(builder.toVariableId(0))

	for i := nbBits - 1; i >= 0; i-- {

		// if bound[i] == 0
		// 		p[i] = p[i+1]
		//		t = p[i+1]
		// else
		// 		p[i] = p[i+1] * a[i]
		//		t = 0
		v := builder.Mul(p[i+1], aBits[i])
		p[i] = builder.Select(boundBits[i], v, p[i+1])

		t := builder.Select(boundBits[i], zero, p[i+1])

		// (1 - t - ai) * ai == 0
		var l frontend.Variable
		l = newVariable(builder.toVariableId(1))
		l = builder.Sub(l, t, aBits[i])

		// note if bound[i] == 1, this constraint is (1 - ai) * ai == 0
		// → this is a boolean constraint
		// if bound[i] == 0, t must be 0 or 1, thus ai must be 0 or 1 too

		builder.AssertIsEqual(builder.Mul(l, aBits[i]), zero)
	}
}

// MustBeLessOrEqCst asserts that the value represented by its bit decomposition is less than or equal to a constant bound.
func (builder *builder) MustBeLessOrEqCst(aBits []frontend.Variable, bound *big.Int, aForDebug frontend.Variable) {
	nbBits := builder.field.FieldBitLen()
	if len(aBits) > nbBits {
		panic("more input bits than field bit length")
	}
	for i := len(aBits); i < nbBits; i++ {
		aBits = append(aBits, 0)
	}

	// ensure the bound is positive, it's bit-len doesn't matter
	if bound.Sign() == -1 {
		panic("AssertIsLessOrEqual: bound must be positive")
	}
	if bound.BitLen() > nbBits {
		panic("AssertIsLessOrEqual: bound is too large, constraint will never be satisfied")
	}

	// t trailing bits in the bound
	t := 0
	for i := 0; i < nbBits; i++ {
		if bound.Bit(i) == 0 {
			break
		}
		t++
	}

	p := make([]frontend.Variable, nbBits+1)
	// p[i] == 1 → a[j] == c[j] for all j ⩾ i
	p[nbBits] = newVariable(builder.toVariableId(1))

	for i := nbBits - 1; i >= t; i-- {
		if bound.Bit(i) == 0 {
			p[i] = p[i+1]
		} else {
			p[i] = builder.Mul(p[i+1], aBits[i])
		}
	}

	for i := nbBits - 1; i >= 0; i-- {
		if bound.Bit(i) == 0 {
			// (1 - p(i+1) - ai) * ai == 0
			l := builder.Sub(1, p[i+1])
			l = builder.Sub(l, aBits[i])

			builder.AssertIsEqual(builder.Mul(l, aBits[i]), newVariable(builder.toVariableId(0)))
		} else {
			builder.AssertIsBoolean(aBits[i])
		}
	}
}
