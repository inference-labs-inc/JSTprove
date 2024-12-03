package irsource

type ConstraintType = int

const (
	_ ConstraintType = iota
	Zero
	NonZero
	Bool
)

type Constraint struct {
	Typ ConstraintType
	Var int
}
