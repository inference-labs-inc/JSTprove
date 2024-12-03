package builder

import "github.com/PolyhedraZK/ExpanderCompilerCollection/ecgo/utils/gnarkexpr"

func newVariable(id int) gnarkexpr.Expr {
	return gnarkexpr.NewVar(id)
}
