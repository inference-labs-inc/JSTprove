package gnarkexpr

import (
	"reflect"

	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

var builder frontend.Builder

type Expr interface {
	WireID() int
}

func init() {
	var err error
	builder, err = r1cs.NewBuilder(bn254.ID.ScalarField(), frontend.CompileConfig{})
	if err != nil {
		panic(err)
	}
}

func NewVar(x int) Expr {
	v := builder.InternalVariable(uint32(x))
	t := reflect.ValueOf(v).Index(0).Interface().(Expr)
	if t.WireID() != x {
		panic("variable id mismatch, please check gnark version")
	}
	return t
}
