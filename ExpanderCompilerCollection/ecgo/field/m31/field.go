package m31

import (
	"math/big"
	"strconv"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils"
	"github.com/consensys/gnark/constraint"
)

const P = 0x7fffffff

var Pbig = big.NewInt(P)
var ScalarField = Pbig

type Field struct{}

func modReduce(x uint64) uint64 {
	x = (x & P) + (x >> 31)
	if x >= P {
		x -= P
	}
	return x
}

func (engine *Field) FromInterface(i interface{}) constraint.Element {
	b := utils.FromInterface(i)
	b.Mod(&b, Pbig)
	return constraint.Element{b.Uint64()}
}

func (engine *Field) ToBigInt(c constraint.Element) *big.Int {
	return big.NewInt(int64(c[0]))
}

func (engine *Field) Mul(a, b constraint.Element) constraint.Element {
	return constraint.Element{modReduce(a[0] * b[0])}
}

func (engine *Field) Add(a, b constraint.Element) constraint.Element {
	res := a[0] + b[0]
	if res >= P {
		res -= P
	}
	return constraint.Element{res}
}

func (engine *Field) Sub(a, b constraint.Element) constraint.Element {
	res := int64(a[0]) - int64(b[0])
	if res < 0 {
		res += P
	}
	return constraint.Element{uint64(res)}
}

func (engine *Field) Neg(a constraint.Element) constraint.Element {
	return constraint.Element{modReduce(P - a[0])}
}

func (engine *Field) Inverse(a constraint.Element) (constraint.Element, bool) {
	if a[0] == 0 {
		return a, false
	}
	var res uint64 = 1
	b := a[0]
	for i := P - 2; i > 0; i >>= 1 {
		if (i & 1) != 0 {
			res = modReduce(res * b)
		}
		b = modReduce(b * b)
	}
	return constraint.Element{res}, true
}

func (engine *Field) IsOne(a constraint.Element) bool {
	return a[0] == 1
}

func (engine *Field) One() constraint.Element {
	return constraint.Element{1}
}

func (engine *Field) String(a constraint.Element) string {
	return strconv.Itoa(int(a[0]))
}

func (engine *Field) Uint64(a constraint.Element) (uint64, bool) {
	return a[0], true
}

func (engine *Field) Field() *big.Int {
	return ScalarField
}

func (engine *Field) FieldBitLen() int {
	return 31
}

func (engine *Field) SerializedLen() int {
	return 4
}
