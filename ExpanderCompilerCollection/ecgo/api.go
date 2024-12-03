// Package ecgo wraps the most commonly used compiler APIs and provides an entry point for compilation.
// This package simplifies the interaction with the compiler by exposing a unified API interface.
package ecgo

import (
	"errors"
	"fmt"
	"math/big"
	"reflect"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/builder"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/irsource"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/irwg"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/layered"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/rust"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/schema"
	"github.com/consensys/gnark/logger"
)

// API encapsulates the ecgo's frontend.API along with two new APIs added to facilitate
// direct invocation of ecgo.API within the codebase.
type API interface {
	frontend.API
	builder.SubCircuitAPI
	builder.API
}

// CompileResult represents the result of a compilation process.
// It contains unexported fields and provides methods to retrieve various components
// like the intermediate representation (IR) of the circuit, the WitnessGenerator, and the Layered Circuit.
type CompileResult struct {
	irs  *irsource.RootCircuit
	irwg *irwg.RootCircuit
	lc   *layered.RootCircuit
}

// Compile is similar to gnark's frontend.Compile. It compiles the given circuit and returns
// a pointer to CompileResult along with any error encountered during the compilation process.
func Compile(field *big.Int, circuit frontend.Circuit, opts ...frontend.CompileOption) (*CompileResult, error) {
	log := logger.Logger()
	log.Info().Msg("compiling circuit")

	opt := frontend.CompileConfig{CompressThreshold: 0}
	for _, o := range opts {
		if err := o(&opt); err != nil {
			log.Err(err).Msg("applying compile option")
			return nil, fmt.Errorf("apply option: %w", err)
		}
	}

	root := builder.NewRoot(field, opt)
	schema.Walk(circuit, irwg.TVariable, func(f schema.LeafInfo, tInput reflect.Value) error {
		if tInput.CanSet() {
			if f.Visibility == schema.Unset {
				return errors.New("can't set val " + f.FullName() + " visibility is unset")
			}
			if f.Visibility == schema.Secret {
				tInput.Set(reflect.ValueOf(root.SecretVariable(f)))
			}
			return nil
		}
		return errors.New("can't set val " + f.FullName())
	})
	schema.Walk(circuit, irwg.TVariable, func(f schema.LeafInfo, tInput reflect.Value) error {
		if tInput.CanSet() {
			if f.Visibility == schema.Unset {
				return errors.New("can't set val " + f.FullName() + " visibility is unset")
			}
			if f.Visibility == schema.Public {
				tInput.Set(reflect.ValueOf(root.PublicVariable(f)))
			}
			return nil
		}
		return errors.New("can't set val " + f.FullName())
	})

	err := circuit.Define(root)
	if err != nil {
		return nil, err
	}
	rc := root.Finalize()
	_ = rc
	//os.WriteFile("p1.txt", irsource.SerializeRootCircuit(rc), 0644)
	irwg, lc, err := rust.Compile(rc)
	if err != nil {
		return nil, err
	}
	return &CompileResult{irs: rc, irwg: irwg, lc: lc}, nil
}

// GetCircuitIr returns the intermediate representation (IR) of the compiled circuit as *ir.RootCircuit.
func (c *CompileResult) GetCircuitIr() *irsource.RootCircuit {
	return c.irs
}

// GetInputSolver returns the InputSolver component of the compilation result as *ir.InputSolver.
func (c *CompileResult) GetLayeredCircuit() *layered.RootCircuit {
	return c.lc
}

// GetLayeredCircuit returns the Layered Circuit component of the compilation result as *layered.RootCircuit.
func (c *CompileResult) GetInputSolver() *irwg.RootCircuit {
	return c.irwg
}

// DeserializeLayeredCircuit takes a byte buffer and returns a pointer to a layered.RootCircuit
// which represents a deserialized layered circuit.
func DeserializeLayeredCircuit(buf []byte) *layered.RootCircuit {
	return layered.DeserializeRootCircuit(buf)
}

// DeserializeInputSolver takes a byte buffer and returns a pointer to an ir.InputSolver
// which represents a deserialized input solver.
func DeserializeInputSolver(buf []byte) *irwg.RootCircuit {
	return irwg.DeserializeRootCircuit(buf)
}
