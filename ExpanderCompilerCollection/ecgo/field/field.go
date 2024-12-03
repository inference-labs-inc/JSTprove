package field

import (
	"fmt"
	"math/big"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field/bn254"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field/gf2"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field/m31"
	"github.com/consensys/gnark/constraint"
)

type Field interface {
	constraint.Field
	Field() *big.Int
	FieldBitLen() int
	SerializedLen() int
}

func GetFieldFromOrder(x *big.Int) Field {
	if x.Cmp(bn254.ScalarField) == 0 {
		return &bn254.Field{}
	}
	if x.Cmp(m31.ScalarField) == 0 {
		return &m31.Field{}
	}
	if x.Cmp(gf2.ScalarField) == 0 {
		return &gf2.Field{}
	}
	panic(fmt.Sprintf("unknown field %v", x))
}

func GetFieldId(f Field) uint64 {
	if f.Field().Cmp(m31.ScalarField) == 0 {
		return 1
	}
	if f.Field().Cmp(bn254.ScalarField) == 0 {
		return 2
	}
	if f.Field().Cmp(gf2.ScalarField) == 0 {
		return 3
	}
	panic(fmt.Sprintf("unsupported field %v", f))
}

func GetFieldById(id uint64) Field {
	switch id {
	case 1:
		return &m31.Field{}
	case 2:
		return &bn254.Field{}
	case 3:
		return &gf2.Field{}
	}
	panic(fmt.Sprintf("unsupported field id %v", id))
}
