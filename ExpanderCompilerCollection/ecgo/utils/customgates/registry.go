package customgates

import (
	"fmt"

	"github.com/consensys/gnark/constraint/solver"
)

type customGateEntry struct {
	hintFunc solver.Hint
	cost     int
}

var customGateHintFunc = make(map[uint64]customGateEntry)

// Register a custom gate. It also registers in the gnark hint registry
func Register(customGateType uint64, f solver.Hint, cost int) {
	customGateHintFunc[customGateType] = customGateEntry{
		hintFunc: f,
		cost:     cost,
	}
	solver.RegisterHint(f)
}

func GetFunc(customGateType uint64) solver.Hint {
	if h, ok := customGateHintFunc[customGateType]; ok {
		return h.hintFunc
	}
	panic(fmt.Sprintf("custom gate %d not registered", customGateType))
}

func GetCost(customGateType uint64) int {
	if h, ok := customGateHintFunc[customGateType]; ok {
		return h.cost
	}
	panic(fmt.Sprintf("custom gate %d not registered", customGateType))
}
