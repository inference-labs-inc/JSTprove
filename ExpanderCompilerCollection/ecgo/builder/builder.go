// Some content of this file is copied from gnark/frontend/cs/r1cs/builder.go

// Package builder provides an implementation based on the gnark frontend builder with the following modifications:
// - LinearExpression has been changed to allow for quadratic terms in the form of Expression.
// - Assert series functions are recorded first and later solidified into the IR.
// - Support for subcircuits is integrated within the builder.
package builder

import (
	"math/big"
	"reflect"

	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/irsource"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils/gnarkexpr"
)

// builder implements frontend.API and frontend.Compiler, and builds a circuit
// it can be a root circuit or a sub circuit
type builder struct {
	field field.Field

	// builder of the root circuit
	root *Root

	// widely used expressions
	tOne constraint.Element

	instructions []irsource.Instruction
	constraints  []irsource.Constraint

	nbExternalInput int
	maxVar          int

	// defers (for gnark API)
	defers []func(frontend.API) error

	// we have to implement kvstore.Store (required by gnark/internal/circuitdefer/defer.go:30)
	db map[any]any

	// output of sub circuit
	output []int
}

// newBuilder returns a builder with known number of external input
func (r *Root) newBuilder(nbExternalInput int) *builder {
	builder := builder{
		field:           r.field,
		root:            r,
		db:              make(map[any]any),
		nbExternalInput: nbExternalInput,
	}

	builder.tOne = builder.field.One()

	builder.maxVar = nbExternalInput

	return &builder
}

// ToSingleVariable converts an expression to a single base variable without a constant term.
func (builder *builder) ToSingleVariable(ein frontend.Variable) frontend.Variable {
	// TODO: noop
	return ein
}

// Field returns the value of the current field being used.
func (builder *builder) Field() *big.Int {
	return builder.field.Field()
}

// FieldBitLen returns the bit length of the current field being used.
func (builder *builder) FieldBitLen() int {
	return builder.field.FieldBitLen()
}

// LayerOf returns the expected layer of the variable, though the actual layer may vary.
func (builder *builder) LayerOf(e frontend.Variable) int {
	// TODO: noop
	return 0
}

// MarkBoolean sets (but do not **constraint**!) v to be boolean
// This is useful in scenarios where a variable is known to be boolean through a constraint
// that is not api.AssertIsBoolean. If v is a constant, this is a no-op.
func (builder *builder) MarkBoolean(v frontend.Variable) {
	// TODO: noop
}

// IsBoolean returns true if given variable was marked as boolean in the compiler (see MarkBoolean)
// Use with care; variable may not have been **constrained** to be boolean
// This returns true if the v is a constant and v == 0 || v == 1.
func (builder *builder) IsBoolean(v frontend.Variable) bool {
	// TODO: noop
	return false
}

// Compile is a placeholder for gnark API compatibility; it does nothing.
func (builder *builder) Compile() (constraint.ConstraintSystem, error) {
	return nil, nil
}

// ConstantValue returns always returns (nil, false) now, since the Golang frontend doesn't know the values of variables.
func (builder *builder) ConstantValue(v frontend.Variable) (*big.Int, bool) {
	return nil, false
}

func (builder *builder) addVarId() int {
	builder.maxVar += 1
	return builder.maxVar
}

func (builder *builder) addVar() frontend.Variable {
	return newVariable(builder.addVarId())
}

func (builder *builder) ceToId(x constraint.Element) int {
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:    irsource.ConstantLike,
		ExtraId: 0,
		Const:   x,
	})
	return builder.addVarId()
}

// toVariable will return (and allocate if neccesary) an Expression from given value
//
// if input is already an Expression, does nothing
// else, attempts to convert input to a big.Int (see utils.FromInterface) and returns a toVariable Expression
func (builder *builder) toVariableId(input interface{}) int {

	switch t := input.(type) {
	case gnarkexpr.Expr:
		return t.WireID()
	case constraint.Element:
		return builder.ceToId(t)
	case *constraint.Element:
		return builder.ceToId(*t)
	default:
		// try to make it into a constant
		c := builder.field.FromInterface(t)
		return builder.ceToId(c)
	}
}

// toVariables return frontend.Variable corresponding to inputs and the total size of the linear expressions
func (builder *builder) toVariableIds(in ...frontend.Variable) []int {
	r := make([]int, 0, len(in))
	e := func(i frontend.Variable) {
		v := builder.toVariableId(i)
		r = append(r, v)
	}
	for i := 0; i < len(in); i++ {
		e(in[i])
	}
	return r
}

// NewHint initializes internal variables whose value will be evaluated using
// the provided hint function at run time from the inputs. Inputs must be either
// variables or convertible to *big.Int. The function returns an error if the
// number of inputs is not compatible with f.
//
// The hint function is provided at the input solving time and is not embedded
// into the circuit. From the prover point of view, the variable returned by
// the hint function is equivalent to the user-supplied witness, but its actual
// value is assigned by the InputSolver, not the caller.
//
// No new constraints are added to the newly created wire and must be added
// manually in the circuit. Failing to do so leads to solver failure.
func (builder *builder) NewHint(f solver.Hint, nbOutputs int, inputs ...frontend.Variable) ([]frontend.Variable, error) {
	return builder.newHintForId(solver.GetHintID(f), nbOutputs, inputs)
}

// NewHintForId is not implemented and will panic if called.
func (builder *builder) NewHintForId(id solver.HintID, nbOutputs int, inputs ...frontend.Variable) ([]frontend.Variable, error) {
	return builder.newHintForId(id, nbOutputs, inputs)
}

func (builder *builder) newHintForId(id solver.HintID, nbOutputs int, inputs []frontend.Variable) ([]frontend.Variable, error) {
	hintInputs := builder.toVariableIds(inputs...)

	builder.instructions = append(builder.instructions,
		irsource.Instruction{
			Type:       irsource.Hint,
			ExtraId:    uint64(id),
			Inputs:     hintInputs,
			NumOutputs: nbOutputs,
		},
	)

	res := make([]frontend.Variable, nbOutputs)
	for i := 0; i < nbOutputs; i++ {
		builder.maxVar += 1
		res[i] = newVariable(builder.maxVar)
	}
	return res, nil
}

func (builder *builder) CustomGate(gateType uint64, inputs ...frontend.Variable) frontend.Variable {
	hintInputs := builder.toVariableIds(inputs...)

	builder.instructions = append(builder.instructions,
		irsource.Instruction{
			Type:    irsource.CustomGate,
			ExtraId: gateType,
			Inputs:  hintInputs,
		},
	)
	return builder.addVar()
}

// assertIsSet

// IdentityHint sets output[0] to input[0] and is used to implement ToFirstLayer.
func IdentityHint(field *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	a := big.NewInt(0)
	a.Set(inputs[0])
	outputs[0] = a
	return nil
}

// ToFirstLayer adds a hint to the target variable to bring it to the first layer.
func (builder *builder) ToFirstLayer(v frontend.Variable) frontend.Variable {
	x, _ := builder.NewHint(IdentityHint, 1, v)
	builder.AssertIsEqual(x[0], v)
	return x[0]
}

// Defer adds a callback function to the defer list to be processed later.
func (builder *builder) Defer(cb func(frontend.API) error) {
	builder.defers = append(builder.defers, cb)
}

// AddInstruction is not implemented and will panic if called.
func (builder *builder) AddInstruction(bID constraint.BlueprintID, calldata []uint32) []uint32 {
	panic("unimplemented")
}

// AddBlueprint is not implemented and will panic if called.
func (builder *builder) AddBlueprint(b constraint.Blueprint) constraint.BlueprintID {
	panic("unimplemented")
}

// InternalVariable is not implemented and will panic if called.
func (builder *builder) InternalVariable(wireID uint32) frontend.Variable {
	panic("unimplemented")
}

// ToCanonicalVariable is not implemented and will panic if called.
func (builder *builder) ToCanonicalVariable(in frontend.Variable) frontend.CanonicalVariable {
	panic("unimplemented")
}

// SetKeyValue implements kvstore for the gnark frontend.
func (builder *builder) SetKeyValue(key, value any) {
	if !reflect.TypeOf(key).Comparable() {
		panic("key type not comparable")
	}
	builder.db[key] = value
}

// GetKeyValue implements kvstore for the gnark frontend.
func (builder *builder) GetKeyValue(key any) any {
	if !reflect.TypeOf(key).Comparable() {
		panic("key type not comparable")
	}
	return builder.db[key]
}

// GetRandomValue returns a random value determined during the proving time.
// The return value cannot be used in hints, since it's unknown at the input solving phase
func (builder *builder) GetRandomValue() frontend.Variable {
	builder.instructions = append(builder.instructions, irsource.Instruction{
		Type:    irsource.ConstantLike,
		ExtraId: 1,
	})
	return builder.addVar()
}
