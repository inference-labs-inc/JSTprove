package builder

import (
	"math/big"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/field"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/irsource"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/schema"
)

// Root is a builder for the root circuit. It implements functions from ExpanderCompilerCollection.API and also handles functions
// such as PublicVariable that are only called at the root circuit level. Additionally, Root maintains a subcircuit registry.
type Root struct {
	*builder
	field  field.Field
	config frontend.CompileConfig

	registry *SubCircuitRegistry

	nbPublicInputs int
}

// NewRoot returns a new Root instance.
func NewRoot(fieldorder *big.Int, config frontend.CompileConfig) *Root {
	root := Root{
		config: config,
	}
	root.field = field.GetFieldFromOrder(fieldorder)
	root.registry = newSubCircuitRegistry()

	root.builder = root.newBuilder(0)
	root.registry.m[0] = &SubCircuit{
		builder: root.builder,
	}

	return &root
}

// PublicVariable creates a new public variable for the circuit.
func (r *Root) PublicVariable(f schema.LeafInfo) frontend.Variable {
	r.instructions = append(r.instructions, irsource.Instruction{
		Type:    irsource.ConstantLike,
		ExtraId: 2 + uint64(r.nbPublicInputs),
	})
	r.nbPublicInputs++
	return r.addVar()
}

// SecretVariable creates a new secret variable for the circuit.
func (r *Root) SecretVariable(f schema.LeafInfo) frontend.Variable {
	r.builder.nbExternalInput++
	return r.addVar()
}
