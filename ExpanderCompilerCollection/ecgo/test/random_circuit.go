package test

import (
	"math/big"
	"math/rand"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/consensys/gnark/frontend"
)

const randomCircuitInputSize = 1024

type randomCircuit struct {
	Input  [randomCircuitInputSize]frontend.Variable
	Output frontend.Variable
	rcg    *randomCircuitGenerator
}

type randomCircuitConfig struct {
	seed       int
	scNum      randRange
	scInput    randRange
	scOutput   randRange
	scInsn     randRange
	rootInsn   randRange
	field      *big.Int
	addPercent int
	mulPercent int
	divPercent int
}

type randRange struct {
	l int
	r int
}

type randomCircuitGenerator struct {
	conf        *randomCircuitConfig
	inputMap    map[int]int
	inputSize   []int
	outputSize  []int
	subCircuits [][]int
}

func (rr *randRange) sample(r *rand.Rand) int {
	return r.Intn(rr.r-rr.l+1) + rr.l
}

func (circuit *randomCircuit) Define(api frontend.API) error {
	return circuit.rcg.define(api, circuit.Input[:], circuit.Output)
}

func newRandomCircuitGenerator(conf *randomCircuitConfig) *randomCircuitGenerator {
	rand := rand.New(rand.NewSource(int64(conf.seed)))

	// generate a DAG of circuits
	n := conf.scNum.sample(rand)
	is := make([]int, n+1)
	os := make([]int, n+1)
	sub := make([][]int, n+1)
	for i := 0; i < n; i++ {
		os[i] = conf.scOutput.sample(rand)
	tryagain:
		for {
			x := conf.scInput.sample(rand)
			for j := 0; j < i; j++ {
				if is[j] == x {
					continue tryagain
				}
			}
			is[i] = x
			break
		}
	}
	for i := 0; i < n; i++ {
		sub[i] = []int{}
		for j := 0; (1 << j) < i; j++ {
			sub[i] = append(sub[i], rand.Intn(i))
		}
	}
	is[n] = randomCircuitInputSize
	os[n] = 1
	sub[n] = make([]int, n)
	for i := 0; i < n; i++ {
		sub[n][i] = i
	}

	im := make(map[int]int)
	for i, x := range is {
		im[x] = i
	}

	return &randomCircuitGenerator{
		conf:        conf,
		inputMap:    im,
		inputSize:   is,
		outputSize:  os,
		subCircuits: sub,
	}
}

func (rcg *randomCircuitGenerator) circuit() *randomCircuit {
	return &randomCircuit{
		rcg: rcg,
	}
}

func (rcg *randomCircuitGenerator) define(api frontend.API, input []frontend.Variable, expected frontend.Variable) error {
	out := rcg.randomCircuit(api, input)
	api.AssertIsEqual(out[0], expected)
	return nil
}

// randomCircuit generates a random circuit
// the behavior is deterministic based on the seed and input size, so the builder can memorize it
func (rcg *randomCircuitGenerator) randomCircuit(api frontend.API, input []frontend.Variable) []frontend.Variable {
	id := rcg.inputMap[len(input)]
	rand := rand.New(rand.NewSource(int64(id<<32) | int64(rcg.conf.seed)))
	vars := make([]frontend.Variable, len(input))
	copy(vars, input)
	m := rcg.conf.scInsn.sample(rand)
	if len(input) == randomCircuitInputSize {
		m = rcg.conf.rootInsn.sample(rand)
	}
	for i := 0; i < m || len(vars) < rcg.outputSize[id]; i++ {
		op := rand.Intn(100)
		if op < rcg.conf.addPercent {
			x := rand.Intn(len(vars))
			y := rand.Intn(len(vars))
			vars = append(vars, api.Add(
				api.Mul(vars[x], big.NewInt(0).Rand(rand, rcg.conf.field)),
				api.Mul(vars[y], big.NewInt(0).Rand(rand, rcg.conf.field)),
			))
		} else if op < rcg.conf.mulPercent || len(vars) < 2 {
			x := rand.Intn(len(vars))
			y := rand.Intn(len(vars))
			vars = append(vars, api.Mul(vars[x], vars[y]))
		} else if op < rcg.conf.divPercent || len(rcg.subCircuits[id]) == 0 {
			for {
				x := rand.Intn(len(vars))
				y := rand.Intn(len(vars))
				if x != y {
					vars = append(vars, api.Div(vars[x], vars[y]))
					break
				}
			}
		} else {
			p := rand.Perm(len(vars))
			sub := rcg.subCircuits[id][rand.Intn(len(rcg.subCircuits[id]))]
			if len(vars) >= rcg.inputSize[sub] {
				subIn := make([]frontend.Variable, rcg.inputSize[sub])
				for i := 0; i < rcg.inputSize[sub]; i++ {
					subIn[i] = vars[p[i]]
				}
				//subOut := rcg.randomCircuit(api, subIn)
				subOut := api.(ecgo.API).MemorizedSimpleCall(rcg.randomCircuit, subIn)
				vars = append(vars, subOut...)
			}
		}
	}
	p := rand.Perm(len(vars))
	out := make([]frontend.Variable, rcg.outputSize[id])
	for i := 0; i < rcg.outputSize[id]; i++ {
		out[i] = vars[p[i]]
	}
	return out
}

func (rcg *randomCircuitGenerator) randomAssignment(subSeed int) *randomCircuit {
	rand := rand.New(rand.NewSource(int64(subSeed<<48) | int64(rcg.conf.seed)))
	input := make([]*big.Int, randomCircuitInputSize)
	for i := 0; i < randomCircuitInputSize; i++ {
		input[i] = big.NewInt(0).Rand(rand, rcg.conf.field)
	}
	output := rcg.eval(input)
	res := &randomCircuit{}
	for i, x := range input {
		res.Input[i] = x
	}
	res.Output = output
	return res
}

func (rcg *randomCircuitGenerator) eval(input []*big.Int) *big.Int {
	out := rcg.randomEval(input)
	return out[0]
}

// randomCircuit generates a random circuit
// the behavior is deterministic based on the seed and input size, so the builder can memorize it
func (rcg *randomCircuitGenerator) randomEval(input []*big.Int) []*big.Int {
	id := rcg.inputMap[len(input)]
	rand := rand.New(rand.NewSource(int64(id<<32) | int64(rcg.conf.seed)))
	vars := make([]*big.Int, len(input))
	copy(vars, input)
	m := rcg.conf.scInsn.sample(rand)
	if len(input) == randomCircuitInputSize {
		m = rcg.conf.rootInsn.sample(rand)
	}
	for i := 0; i < m || len(vars) < rcg.outputSize[id]; i++ {
		op := rand.Intn(100)
		if op < rcg.conf.addPercent {
			x := rand.Intn(len(vars))
			y := rand.Intn(len(vars))
			a := big.NewInt(0).Rand(rand, rcg.conf.field)
			a = a.Mul(a, vars[x])
			b := big.NewInt(0).Rand(rand, rcg.conf.field)
			b = b.Mul(b, vars[y])
			a = a.Add(a, b)
			vars = append(vars, a.Mod(a, rcg.conf.field))
		} else if op < rcg.conf.mulPercent || len(vars) < 2 {
			x := rand.Intn(len(vars))
			y := rand.Intn(len(vars))
			a := big.NewInt(0).Mul(vars[x], vars[y])
			vars = append(vars, a.Mod(a, rcg.conf.field))
		} else if op < rcg.conf.divPercent || len(rcg.subCircuits[id]) == 0 {
			for {
				x := rand.Intn(len(vars))
				y := rand.Intn(len(vars))
				if x != y {
					a := big.NewInt(0).ModInverse(vars[y], rcg.conf.field)
					a = a.Mul(a, vars[x])
					vars = append(vars, a.Mod(a, rcg.conf.field))
					break
				}
			}
		} else {
			p := rand.Perm(len(vars))
			sub := rcg.subCircuits[id][rand.Intn(len(rcg.subCircuits[id]))]
			if len(vars) >= rcg.inputSize[sub] {
				subIn := make([]*big.Int, rcg.inputSize[sub])
				for i := 0; i < rcg.inputSize[sub]; i++ {
					subIn[i] = vars[p[i]]
				}
				subOut := rcg.randomEval(subIn)
				vars = append(vars, subOut...)
			}
		}
	}
	p := rand.Perm(len(vars))
	out := make([]*big.Int, rcg.outputSize[id])
	for i := 0; i < rcg.outputSize[id]; i++ {
		out[i] = vars[p[i]]
	}
	return out
}
