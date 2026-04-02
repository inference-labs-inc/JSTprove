# ceck - Constraint Equivalence Checker

A tool for verifying equivalence between constraint systems using random testing and SMT checking.

## Quick Start

```bash
# Run on a test file
ceck test.ceck

# Read from stdin
echo "(assert_eqv (constraint_set) (constraint_set))" | ceck -

# Skip SMT checking (only randblast)
ceck test.ceck --skip-smt

# Skip random testing (only SMT)
ceck test.ceck --skip-random
```

## Command-Line Options

Run `ceck --help` for full options:
- `-r, --random-tests <N>` - Number of random tests (default: 10000)
- `-s, --skip-random` - Skip random testing
- `--skip-smt` - Skip SMT checking
- `--seed <SEED>` - Random seed (default: 42)
- `-o, --optimistic` - Don't fail on inconclusive results

## Test File Format

Test files contain assertions that check constraint system equivalence:

```lisp
; Assert two constraint systems are equivalent
(assert_eqv
  (constraint_set
    (and $v0 $v1 $v2)
  )
  (constraint_set
    (and $v0 $v1 $v2)
  )
)

; Assert two constraint systems are NOT equivalent
(assert_not_eqv
  (constraint_set
    (and $v0 $v1 $v2)
  )
  (constraint_set
    (and $v0 $v1 0x0)
  )
)
```

## Grammar Cheatsheet

### Constraint Types
- **AND constraint**: `(and <operand> <operand> <operand>)` - Checks `(A & B) ^ C = 0`
- **MUL constraint**: `(mul <operand> <operand> <operand> <operand>)` - Checks `A * B = (HI << 64) | LO`

### Operands
- **Wire**: `$v0`, `$v1`, `$foo` - Named 64-bit witness variables
- **Literal**: `0x0`, `0xFFFFFFFF_FFFFFFFF`, `42` - 64-bit constants
- **XOR expression**: `(xor $v0 $v1 $v2)` - XOR of multiple terms
- **Shifted term**: `(sll $v0 5)`, `(slr $v1 32)`, `(sar $v2 16)` - Bit shifts

### Shift Operations
- `sll` - Logical left shift
- `slr` - Logical right shift
- `sar` - Arithmetic right shift

## Examples

See `testsuite/` directory for comprehensive examples:
- `basic_and.ceck` - Simple AND constraint examples
- `xor.ceck` - XOR operand usage
- `shift.ceck` - Shift operations
- `mul.ceck` - Multiplication constraints
- `mixed.ceck` - Complex constraint combinations
- `assertion_types.ceck` - Both assert_eqv and assert_not_eqv

## Exit Codes

- `0` - All assertions passed
- `1` - One or more assertions failed

## Complete Example

Create a file `example.ceck`:
```lisp
; This should pass - both implement v2 = v0 & v1
(assert_eqv
  (constraint_set
    (and $v0 $v1 $v2)
  )
  (constraint_set
    (and $v1 $v0 $v2)  ; AND is commutative
  )
)

; This should fail - different constraints
(assert_not_eqv
  (constraint_set
    (and $v0 $v1 $v2)
  )
  (constraint_set
    (and $v0 $v0 $v2)  ; Different operand
  )
)
```

Run it:
```bash
$ ceck example.ceck
Test 1: EQUIVALENT
Test 2: NOT EQUIVALENT (as expected)
```

## How It Works

1. **Randblast**: Tests with random 64-bit values to quickly find counterexamples
2. **SMT Check**: Uses Z3 theorem prover for complete verification (requires z3 feature)

The tool reports systems as:
- **EQUIVALENT** - Both constraint systems accept the same witnesses
- **NOT EQUIVALENT** - Found witness satisfying one system but not the other

## Building

By default, ceck is built without Z3 (randblast only):
```bash
cargo build -p ceck
```

To build with Z3 support for complete verification:
```bash
cargo build -p ceck --features z3
```

For z3 you would need to make sure that Z3 could be linked. On apple silicon macOS, you can install
it via Homebrew. Note, that the z3 crate relies on pkg-config to find the libraries so you would
need that to be installed as well.

```bash
brew install z3 pkg-config
```

Note: Without Z3, the tool provides some assurance through random testing but cannot guarantee equivalence.
By default, inconclusive results will cause tests to fail unless you use the `--optimistic` flag.
