package jinja

type node any

type program struct {
	nodes []node
}

type textNode struct {
	text string
}

type exprNode struct {
	expr expr
}

type ifNode struct {
	cond     expr
	body     []node
	elifs    []elifBranch
	elseBody []node
}

type elifBranch struct {
	cond expr
	body []node
}

type forNode struct {
	vars     []string
	iter     expr
	body     []node
	elseBody []node
}

type setNode struct {
	assigns []assignment
}

type assignment struct {
	target expr
	value  expr
}

type expr any

type litExpr struct {
	value value
}

type varExpr struct {
	name string
}

type attrExpr struct {
	obj  expr
	attr string
}

type indexExpr struct {
	obj expr
	key expr
}

type binExpr struct {
	op    tokenType
	left  expr
	right expr
}

type unaryExpr struct {
	op   tokenType
	expr expr
}

type callExpr struct {
	fn   expr
	args []expr
}

type filterExpr struct {
	input expr
	name  string
	args  []expr
}

type testExpr struct {
	left expr
	name string
	not  bool
}

type objectLiteralExpr struct {
	fields map[string]expr
}

type arrayLiteralExpr struct {
	items []expr
}

type kwargExpr struct {
	name  string
	value expr
}

type loopVar struct {
	index   int
	index0  int
	first   bool
	last    bool
	length  int
	revIdx  int
	revIdx0 int
}
