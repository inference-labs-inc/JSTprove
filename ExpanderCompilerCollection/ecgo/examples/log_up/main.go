package main

import (
	"math"
	"math/rand"
	"os"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"

	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo"
	"github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/test"
)

type RationalNumber struct {
	Numerator   frontend.Variable
	Denominator frontend.Variable
}

func (r *RationalNumber) Add(api frontend.API, other *RationalNumber) RationalNumber {
	return RationalNumber{
		Numerator:   api.Add(api.Mul(r.Numerator, other.Denominator), api.Mul(other.Numerator, r.Denominator)),
		Denominator: api.Mul(r.Denominator, other.Denominator),
	}
}

// Construct a binary summation tree to sum all the values
func SumRationalNumbers(api frontend.API, vs []RationalNumber) RationalNumber {
	n := len(vs)
	if n == 0 {
		return RationalNumber{Numerator: 0, Denominator: 1}
	}

	vvs := make([]RationalNumber, len(vs))
	copy(vvs, vs)

	n_values_to_sum := len(vvs)
	for n_values_to_sum > 1 {
		half_size_floor := n_values_to_sum / 2
		for i := 0; i < half_size_floor; i++ {
			vvs[i] = vvs[i].Add(api, &vvs[i+half_size_floor])
		}

		if n_values_to_sum&1 != 0 {
			vvs[half_size_floor] = vvs[n_values_to_sum-1]
		}

		n_values_to_sum = (n_values_to_sum + 1) / 2
	}

	return vvs[0]
}

type LogUpCircuit struct {
	TableKeys   [][]frontend.Variable
	TableValues [][]frontend.Variable
	QueryKeys   [][]frontend.Variable
	QueryResult [][]frontend.Variable

	QueryCount []frontend.Variable
}

func NewRandomCircuit(
	key_len uint,
	n_table_rows uint,
	n_queries uint,
	n_columns uint,
	fill_values bool,
) *LogUpCircuit {
	c := &LogUpCircuit{}

	c.QueryCount = make([]frontend.Variable, n_table_rows)
	if fill_values {
		for i := 0; i < int(n_table_rows); i++ {
			c.QueryCount[i] = uint(0)
		}
	}

	c.TableKeys = make([][]frontend.Variable, n_table_rows)
	for i := 0; i < int(n_table_rows); i++ {
		c.TableKeys[i] = make([]frontend.Variable, key_len)
		if fill_values {
			for j := 0; j < int(key_len); j++ {
				c.TableKeys[i][j] = rand.Intn(math.MaxInt)
			}
		}
	}

	c.TableValues = make([][]frontend.Variable, n_table_rows)
	for i := 0; i < int(n_table_rows); i++ {
		c.TableValues[i] = make([]frontend.Variable, n_columns)
		if fill_values {
			for j := 0; j < int(n_columns); j++ {
				c.TableValues[i][j] = rand.Intn(math.MaxInt)
			}
		}
	}

	c.QueryKeys = make([][]frontend.Variable, n_queries)
	c.QueryResult = make([][]frontend.Variable, n_queries)

	for i := 0; i < int(n_queries); i++ {
		c.QueryKeys[i] = make([]frontend.Variable, key_len)
		c.QueryResult[i] = make([]frontend.Variable, n_columns)
		if fill_values {
			query_id := rand.Intn(int(n_table_rows))
			c.QueryKeys[i] = c.TableKeys[query_id]
			c.QueryResult[i] = c.TableValues[query_id]
			c.QueryCount[query_id] = c.QueryCount[query_id].(uint) + 1
		}
	}

	return c
}

type ColumnCombineOptions int

const (
	Poly = iota
	FullRandom
)

func SimpleMin(a uint, b uint) uint {
	if a < b {
		return a
	} else {
		return b
	}
}

func GetColumnRandomness(api ecgo.API, n_columns uint, column_combine_options ColumnCombineOptions) []frontend.Variable {
	var randomness = make([]frontend.Variable, n_columns)
	if column_combine_options == Poly { // not tested yet, don't use
		beta := api.GetRandomValue()
		randomness[0] = 1
		randomness[1] = beta

		// Hopefully this will generate fewer layers than sequential pows
		max_deg := uint(1)
		for max_deg < n_columns {
			for i := max_deg + 1; i <= SimpleMin(max_deg*2, n_columns-1); i++ {
				randomness[i] = api.Mul(randomness[max_deg], randomness[i-max_deg])
			}
			max_deg *= 2
		}

		// Debug Code:
		// for i := 1; i < n_columns; i++ {
		// 	api.AssertIsEqual(randomness[i], api.Mul(randomness[i - 1], beta))
		// }

	} else if column_combine_options == FullRandom {
		randomness[0] = 1
		for i := 1; i < int(n_columns); i++ {
			randomness[i] = api.GetRandomValue()
		}
	} else {
		panic("Unknown poly combine options")
	}
	return randomness
}

func CombineColumn(api ecgo.API, vec_2d [][]frontend.Variable, randomness []frontend.Variable) []frontend.Variable {
	n_rows := len(vec_2d)
	if n_rows == 0 {
		return make([]frontend.Variable, 0)
	}

	n_columns := len(vec_2d[0])
	if n_columns != len(randomness) {
		panic("Inconsistent randomness length and column size")
	}

	vec_return := make([]frontend.Variable, 0)
	for i := 0; i < n_rows; i++ {
		var v_at_row_i frontend.Variable = 0
		for j := 0; j < n_columns; j++ {
			v_at_row_i = api.Add(v_at_row_i, api.Mul(randomness[j], vec_2d[i][j]))
		}
		vec_return = append(vec_return, v_at_row_i)
	}
	return vec_return
}

func LogUpPolyValsAtAlpha(api ecgo.API, vec_1d []frontend.Variable, count []frontend.Variable, x frontend.Variable) RationalNumber {
	poly := make([]RationalNumber, len(vec_1d))
	for i := 0; i < len(vec_1d); i++ {
		poly[i] = RationalNumber{
			Numerator:   count[i],
			Denominator: api.Sub(x, vec_1d[i]),
		}
	}
	return SumRationalNumbers(api, poly)
}

func CombineVecAt2d(a [][]frontend.Variable, b [][]frontend.Variable) [][]frontend.Variable {
	if len(a) != len(b) {
		panic("Length does not match at combine 2d")
	}

	r := make([][]frontend.Variable, len(a))
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[i]); j++ {
			r[i] = append(r[i], a[i][j])
		}

		for j := 0; j < len(b[i]); j++ {
			r[i] = append(r[i], b[i][j])
		}
	}

	return r
}

func (c *LogUpCircuit) Check(api ecgo.API, column_combine_option ColumnCombineOptions) error {

	// The challenge used to complete polynomial identity check
	alpha := api.GetRandomValue()
	// The randomness used to combine the columns
	column_combine_randomness := GetColumnRandomness(api, uint(len(c.TableKeys[0])+len(c.TableValues[0])), column_combine_option)

	// Table Polynomial
	table_combined := CombineVecAt2d(c.TableKeys, c.TableValues)
	table_single_column := CombineColumn(api, table_combined, column_combine_randomness)
	table_poly_at_alpha := LogUpPolyValsAtAlpha(api, table_single_column, c.QueryCount, alpha)

	// Query Polynomial
	query_combined := CombineVecAt2d(c.QueryKeys, c.QueryResult)
	query_single_column := CombineColumn(api, query_combined, column_combine_randomness)
	dummy_count := make([]frontend.Variable, len(query_single_column))
	for i := 0; i < len(dummy_count); i++ {
		dummy_count[i] = 1
	}
	query_poly_at_alpha := LogUpPolyValsAtAlpha(api, query_single_column, dummy_count, alpha)

	api.AssertIsEqual(
		api.Mul(table_poly_at_alpha.Numerator, query_poly_at_alpha.Denominator),
		api.Mul(query_poly_at_alpha.Numerator, table_poly_at_alpha.Denominator),
	)
	return nil
}

const ColumnCombineOption ColumnCombineOptions = FullRandom

// Define declares the circuit's constraints
func (c *LogUpCircuit) Define(api frontend.API) error {
	return c.Check(api.(ecgo.API), ColumnCombineOption)
}

func main() {
	KEY_LEN := uint(8)
	N_TABLE_ROWS := uint(128)
	N_QUERIES := uint(512)
	COLUMN_SIZE := uint(8)

	circuit, err := ecgo.Compile(ecc.BN254.ScalarField(), NewRandomCircuit(KEY_LEN, N_TABLE_ROWS, N_QUERIES, COLUMN_SIZE, false))
	if err != nil {
		panic(err.Error())
	}

	c := circuit.GetLayeredCircuit()
	os.WriteFile("circuit.txt", c.Serialize(), 0o644)

	assignment := NewRandomCircuit(KEY_LEN, N_TABLE_ROWS, N_QUERIES, COLUMN_SIZE, true)
	inputSolver := circuit.GetInputSolver()
	witness, err := inputSolver.SolveInput(assignment, 0)
	if err != nil {
		panic(err.Error())
	}

	if !test.CheckCircuit(c, witness) {
		panic("Circuit not satisfied")
	}

	// os.WriteFile("inputsolver.txt", inputSolver.Serialize(), 0o644)
	os.WriteFile("witness.txt", witness.Serialize(), 0o644)
}
