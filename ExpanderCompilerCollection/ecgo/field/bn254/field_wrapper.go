// Copyright 2020 ConsenSys Software Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bn254

import (
	"math/big"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/constraint"
)

var ScalarField = fr.Modulus()

type Field struct{}

func (engine *Field) FromInterface(i interface{}) constraint.Element {
	var e fr.Element
	if _, err := e.SetInterface(i); err != nil {
		// need to clean that --> some code path are dissimilar
		// for example setting a fr.Element from an fp.Element
		// fails with the above but succeeds through big int... (2-chains)
		b := utils.FromInterface(i)
		e.SetBigInt(&b)
	}
	var r constraint.Element
	copy(r[:], e[:])
	return r
}
func (engine *Field) ToBigInt(c constraint.Element) *big.Int {
	e := (*fr.Element)(c[:])
	r := new(big.Int)
	e.BigInt(r)
	return r

}
func (engine *Field) Mul(a, b constraint.Element) constraint.Element {
	_a := (*fr.Element)(a[:])
	_b := (*fr.Element)(b[:])
	_a.Mul(_a, _b)
	return a
}

func (engine *Field) Add(a, b constraint.Element) constraint.Element {
	_a := (*fr.Element)(a[:])
	_b := (*fr.Element)(b[:])
	_a.Add(_a, _b)
	return a
}
func (engine *Field) Sub(a, b constraint.Element) constraint.Element {
	_a := (*fr.Element)(a[:])
	_b := (*fr.Element)(b[:])
	_a.Sub(_a, _b)
	return a
}
func (engine *Field) Neg(a constraint.Element) constraint.Element {
	e := (*fr.Element)(a[:])
	e.Neg(e)
	return a

}
func (engine *Field) Inverse(a constraint.Element) (constraint.Element, bool) {
	if a.IsZero() {
		return a, false
	}
	e := (*fr.Element)(a[:])
	if e.IsZero() {
		return a, false
	} else if e.IsOne() {
		return a, true
	} else {
		var t fr.Element
		t.Neg(e)
		if t.IsOne() {
			return a, true
		}

		e.Inverse(e)
		return a, true
	}
}

func (engine *Field) IsOne(a constraint.Element) bool {
	e := (*fr.Element)(a[:])
	return e.IsOne()
}

func (engine *Field) One() constraint.Element {
	e := fr.One()
	var r constraint.Element
	copy(r[:], e[:])
	return r
}

func (engine *Field) String(a constraint.Element) string {
	e := (*fr.Element)(a[:])
	return e.String()
}

func (engine *Field) Uint64(a constraint.Element) (uint64, bool) {
	e := (*fr.Element)(a[:])
	if !e.IsUint64() {
		return 0, false
	}
	return e.Uint64(), true
}

func (engine *Field) Field() *big.Int {
	return fr.Modulus()
}

func (engine *Field) FieldBitLen() int {
	return fr.Modulus().BitLen()
}

func (engine *Field) SerializedLen() int {
	return 32
}
