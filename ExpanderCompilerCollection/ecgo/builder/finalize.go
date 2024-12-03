package builder

import (
	"fmt"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/irsource"
)

// Finalize processes deferred functions, converts boolean and nonzero assertions to zero assertions,
// and adds public variables to the output.
func (r *Root) Finalize() *irsource.RootCircuit {
	res := make(map[uint64]*irsource.Circuit)
	for x, b := range r.registry.m {
		res[x] = b.builder.Finalize()
	}
	return &irsource.RootCircuit{
		NumPublicInputs:         r.nbPublicInputs,
		ExpectedNumOutputZeroes: 0,
		Circuits:                res,
		Field:                   r.field,
	}
}

func (builder *builder) Finalize() *irsource.Circuit {
	// defers may change during the process
	for i := 0; i < len(builder.defers); i++ {
		cb := builder.defers[i]
		err := cb(builder)
		if err != nil {
			panic(fmt.Sprintf("deferred function failed: %v", err))
		}
	}

	return &irsource.Circuit{
		Instructions: builder.instructions,
		Constraints:  builder.constraints,
		Outputs:      builder.output,
		NumInputs:    builder.nbExternalInput,
	}
}
