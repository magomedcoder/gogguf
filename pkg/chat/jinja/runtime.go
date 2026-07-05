package jinja

import (
	"encoding/json"
	"maps"
	"math"
	"strconv"
	"strings"
)

type valueKind int

const (
	valUndefined valueKind = iota
	valNone
	valBool
	valNumber
	valString
	valArray
	valObject
	valNamespace
)

type value struct {
	kind valueKind
	num  float64
	str  string
	b    bool
	arr  []value
	obj  map[string]value
}

func undefinedVal() value {
	return value{
		kind: valUndefined,
	}
}

func noneVal() value {
	return value{
		kind: valNone,
	}
}

func boolVal(b bool) value {
	return value{
		kind: valBool,
		b:    b,
	}
}

func strVal(s string) value {
	return value{
		kind: valString,
		str:  s,
	}
}

func numVal(s string) value {
	f, _ := strconv.ParseFloat(s, 64)
	return value{
		kind: valNumber,
		num:  f,
	}
}

func intVal(n int) value {
	return value{
		kind: valNumber,
		num:  float64(n),
	}
}

func arrayVal(items []value) value {
	return value{
		kind: valArray,
		arr:  items,
	}
}
func objectVal(obj map[string]value) value {
	return value{
		kind: valObject,
		obj:  obj,
	}
}
func namespaceVal(obj map[string]value) value {
	return value{
		kind: valNamespace,
		obj:  obj,
	}
}

type execCtx struct {
	vars map[string]value
}

func newExecCtx(root map[string]any) *execCtx {
	return &execCtx{vars: fromAnyMap(root)}
}

func fromAnyMap(m map[string]any) map[string]value {
	out := make(map[string]value, len(m))
	for k, v := range m {
		out[k] = fromAny(v)
	}

	return out
}

func fromAny(v any) value {
	switch x := v.(type) {
	case nil:
		return noneVal()
	case bool:
		return boolVal(x)
	case int:
		return intVal(x)
	case int64:
		return intVal(int(x))
	case float64:
		return value{kind: valNumber, num: x}
	case float32:
		return value{kind: valNumber, num: float64(x)}
	case string:
		return strVal(x)
	case []any:
		items := make([]value, len(x))
		for i, it := range x {
			items[i] = fromAny(it)
		}
		return arrayVal(items)
	case map[string]any:
		return objectVal(fromAnyMap(x))
	default:
		return undefinedVal()
	}
}

func (c *execCtx) get(name string) value {
	if v, ok := c.vars[name]; ok {
		return v
	}

	return undefinedVal()
}

func (c *execCtx) set(name string, v value) {
	c.vars[name] = v
}

func (c *execCtx) fork() *execCtx {
	vars := make(map[string]value, len(c.vars))
	maps.Copy(vars, c.vars)

	return &execCtx{
		vars: vars,
	}
}

func execProgram(prog *program, root map[string]any, b *strings.Builder) error {
	ctx := newExecCtx(root)
	return execNodes(prog.nodes, ctx, b)
}

func execNodes(nodes []node, ctx *execCtx, b *strings.Builder) error {
	for _, n := range nodes {
		if err := execNode(n, ctx, b); err != nil {
			return err
		}
	}

	return nil
}

func execNode(n node, ctx *execCtx, b *strings.Builder) error {
	switch n := n.(type) {
	case textNode:
		b.WriteString(n.text)
	case exprNode:
		v, err := evalExpr(n.expr, ctx)
		if err != nil {
			return err
		}

		b.WriteString(v.toString())
	case ifNode:
		ok, err := evalTruthy(n.cond, ctx)
		if err != nil {
			return err
		}

		if ok {
			return execNodes(n.body, ctx, b)
		}

		for _, elif := range n.elifs {
			ok, err := evalTruthy(elif.cond, ctx)
			if err != nil {
				return err
			}

			if ok {
				return execNodes(elif.body, ctx, b)
			}
		}
		return execNodes(n.elseBody, ctx, b)
	case forNode:
		iter, err := evalExpr(n.iter, ctx)
		if err != nil {
			return err
		}

		items := iter.toArray()
		if len(items) == 0 {
			return execNodes(n.elseBody, ctx, b)
		}

		for i, item := range items {
			loop := loopVar{
				index:   i + 1,
				index0:  i,
				first:   i == 0,
				last:    i == len(items)-1,
				length:  len(items),
				revIdx:  len(items) - i,
				revIdx0: len(items) - i - 1,
			}
			local := ctx.fork()
			local.set("loop", fromLoop(loop))

			if len(n.vars) == 1 {
				local.set(n.vars[0], item)
			} else if len(n.vars) == 2 {
				local.set(n.vars[0], intVal(loop.index0))
				local.set(n.vars[1], item)
			}

			if err := execNodes(n.body, local, b); err != nil {
				return err
			}
		}
		return nil
	case setNode:
		for _, a := range n.assigns {
			v, err := evalExpr(a.value, ctx)
			if err != nil {
				return err
			}

			if err := assignTarget(a.target, v, ctx); err != nil {
				return err
			}
		}

		return nil
	default:
		return errf("неизвестный узел")
	}

	return nil
}

func fromLoop(l loopVar) value {
	return objectVal(map[string]value{
		"index":     intVal(l.index),
		"index0":    intVal(l.index0),
		"first":     boolVal(l.first),
		"last":      boolVal(l.last),
		"length":    intVal(l.length),
		"revindex":  intVal(l.revIdx),
		"revindex0": intVal(l.revIdx0),
	})
}

func assignTarget(target expr, v value, ctx *execCtx) error {
	switch t := target.(type) {
	case varExpr:
		ctx.set(t.name, v)
		return nil
	case attrExpr:
		if ve, ok := t.obj.(varExpr); ok {
			base := ctx.get(ve.name)
			if base.obj == nil {
				base.obj = make(map[string]value)
			}

			base.obj[t.attr] = v
			ctx.set(ve.name, base)
			return nil
		}

		base, err := evalExpr(t.obj, ctx)
		if err != nil {
			return err
		}

		if base.kind != valNamespace && base.kind != valObject {
			return errf("нельзя присвоить атрибут %s", t.attr)
		}

		if base.obj == nil {
			base.obj = make(map[string]value)
		}

		base.obj[t.attr] = v
		return storeBase(t.obj, base, ctx)
	default:
		return errf("неподдерживаемая цель присваивания")
	}
}

func storeBase(target expr, v value, ctx *execCtx) error {
	switch t := target.(type) {
	case varExpr:
		ctx.set(t.name, v)
	case attrExpr:
		parent, err := evalExpr(t.obj, ctx)
		if err != nil {
			return err
		}

		if parent.obj == nil {
			parent.obj = make(map[string]value)
		}

		parent.obj[t.attr] = v
		return storeBase(t.obj, parent, ctx)
	}

	return nil
}

func evalTruthy(e expr, ctx *execCtx) (bool, error) {
	v, err := evalExpr(e, ctx)
	if err != nil {
		return false, err
	}

	return v.truthy(), nil
}

func evalExpr(e expr, ctx *execCtx) (value, error) {
	switch e := e.(type) {
	case litExpr:
		return e.value, nil
	case arrayLiteralExpr:
		return evalArrayLiteral(e.items, ctx)
	case objectLiteralExpr:
		return evalObjectLiteral(e.fields, ctx)
	case varExpr:
		return ctx.get(e.name), nil
	case attrExpr:
		obj, err := evalExpr(e.obj, ctx)
		if err != nil {
			return value{}, err
		}
		return obj.getAttr(e.attr), nil
	case indexExpr:
		obj, err := evalExpr(e.obj, ctx)
		if err != nil {
			return value{}, err
		}

		key, err := evalExpr(e.key, ctx)
		if err != nil {
			return value{}, err
		}

		return obj.getIndex(key), nil
	case binExpr:
		return evalBin(e, ctx)
	case unaryExpr:
		return evalUnary(e, ctx)
	case callExpr:
		return evalCall(e, ctx)
	case filterExpr:
		return evalFilter(e, ctx)
	case testExpr:
		return evalTest(e, ctx)
	case kwargExpr:
		return value{}, errf("kwarg вне вызова функции")
	default:
		return value{}, errf("неизвестное выражение")
	}
}

func evalArrayLiteral(items []expr, ctx *execCtx) (value, error) {
	out := make([]value, len(items))
	for i, it := range items {
		v, err := evalExpr(it, ctx)
		if err != nil {
			return value{}, err
		}

		out[i] = v
	}

	return arrayVal(out), nil
}

func evalObjectLiteral(fields map[string]expr, ctx *execCtx) (value, error) {
	out := make(map[string]value, len(fields))
	for k, e := range fields {
		v, err := evalExpr(e, ctx)
		if err != nil {
			return value{}, err
		}
		out[k] = v
	}

	return objectVal(out), nil
}

func evalBin(e binExpr, ctx *execCtx) (value, error) {
	left, err := evalExpr(e.left, ctx)
	if err != nil {
		return value{}, err
	}

	right, err := evalExpr(e.right, ctx)
	if err != nil {
		return value{}, err
	}

	switch e.op {
	case tokPlus:
		if left.kind == valString || right.kind == valString {
			return strVal(left.toString() + right.toString()), nil
		}

		return value{kind: valNumber, num: left.num + right.num}, nil
	case tokMinus:
		return value{kind: valNumber, num: left.num - right.num}, nil
	case tokStar:
		return value{kind: valNumber, num: left.num * right.num}, nil
	case tokSlash:
		return value{kind: valNumber, num: left.num / right.num}, nil
	case tokPercent:
		return intVal(int(left.num) % int(right.num)), nil
	case tokAnd:
		return boolVal(left.truthy() && right.truthy()), nil
	case tokOr:
		return boolVal(left.truthy() || right.truthy()), nil
	case tokEq:
		return boolVal(valuesEqual(left, right)), nil
	case tokNe:
		return boolVal(!valuesEqual(left, right)), nil
	case tokLt:
		return boolVal(left.num < right.num), nil
	case tokGt:
		return boolVal(left.num > right.num), nil
	case tokLe:
		return boolVal(left.num <= right.num), nil
	case tokGe:
		return boolVal(left.num >= right.num), nil
	case tokIn:
		if right.kind == valString {
			return boolVal(strings.Contains(right.str, left.toString())), nil
		}

		for _, it := range right.toArray() {
			if valuesEqual(left, it) {
				return boolVal(true), nil
			}
		}

		return boolVal(false), nil
	default:
		return value{}, errf("неизвестный бинарный оператор")
	}
}

func evalUnary(e unaryExpr, ctx *execCtx) (value, error) {
	v, err := evalExpr(e.expr, ctx)
	if err != nil {
		return value{}, err
	}

	switch e.op {
	case tokNot:
		return boolVal(!v.truthy()), nil
	case tokMinus:
		return value{kind: valNumber, num: -v.num}, nil
	default:
		return value{}, errf("неизвестный унарный оператор")
	}
}

func evalCall(e callExpr, ctx *execCtx) (value, error) {
	if ae, ok := e.fn.(attrExpr); ok {
		base, err := evalExpr(ae.obj, ctx)
		if err != nil {
			return value{}, err
		}

		return callStringMethod(base, ae.attr, e.args, ctx)
	}

	name := ""
	switch fn := e.fn.(type) {
	case varExpr:
		name = fn.name
	default:
		return value{}, errf("вызов не-функции")
	}

	switch name {
	case "namespace":
		return evalNamespaceCall(e.args, ctx)
	case "range":
		args := make([]value, len(e.args))
		for i, a := range e.args {
			v, err := evalExpr(a, ctx)
			if err != nil {
				return value{}, err
			}
			args[i] = v
		}

		return evalRange(args)
	default:
		if name == "split" || name == "lstrip" || name == "rstrip" || name == "strip" {
			return value{}, errf("метод %s нужно вызывать через точку", name)
		}

		return value{}, errf("неизвестная функция %s", name)
	}
}

func evalNamespaceCall(args []expr, ctx *execCtx) (value, error) {
	obj := make(map[string]value)
	for i := range args {
		if kw, ok := args[i].(kwargExpr); ok {
			v, err := evalExpr(kw.value, ctx)
			if err != nil {
				return value{}, err
			}
			obj[kw.name] = v
		}
	}

	return namespaceVal(obj), nil
}

func evalRange(args []value) (value, error) {
	switch len(args) {
	case 1:
		end := max(int(args[0].num), 0)

		items := make([]value, end)
		for i := range items {
			items[i] = intVal(i)
		}

		return arrayVal(items), nil
	case 2:
		start := int(args[0].num)
		end := int(args[1].num)
		if end < start {
			return arrayVal(nil), nil
		}

		items := make([]value, 0, end-start)
		for i := start; i < end; i++ {
			items = append(items, intVal(i))
		}

		return arrayVal(items), nil
	case 3:
		start := int(args[0].num)
		end := int(args[1].num)
		step := int(args[2].num)
		if step == 0 {
			return value{}, errf("range step 0")
		}

		var items []value
		if step > 0 {
			for i := start; i < end; i += step {
				items = append(items, intVal(i))
			}
		} else {
			for i := start; i > end; i += step {
				items = append(items, intVal(i))
			}
		}

		return arrayVal(items), nil
	default:
		return value{}, errf("range ожидает 1-3 аргумента")
	}
}

func evalFilter(e filterExpr, ctx *execCtx) (value, error) {
	v, err := evalExpr(e.input, ctx)
	if err != nil {
		return value{}, err
	}

	switch e.name {
	case "tojson":
		return filterToJSON(v)
	case "length":
		switch v.kind {
		case valString:
			return intVal(len(v.str)), nil
		case valArray:
			return intVal(len(v.arr)), nil
		default:
			return intVal(0), nil
		}
	default:
		return value{}, errf("неизвестный фильтр %s", e.name)
	}
}

func filterToJSON(v value) (value, error) {
	raw := valueToJSON(v)
	b, err := json.Marshal(raw)
	if err != nil {
		return value{}, err
	}

	return strVal(string(b)), nil
}

func valueToJSON(v value) any {
	switch v.kind {
	case valString:
		return v.str
	case valNumber:
		if math.Mod(v.num, 1) == 0 {
			return int(v.num)
		}

		return v.num
	case valBool:
		return v.b
	case valArray:
		arr := make([]any, len(v.arr))
		for i, it := range v.arr {
			arr[i] = valueToJSON(it)
		}

		return arr
	case valObject, valNamespace:
		m := make(map[string]any)
		for k, it := range v.obj {
			m[k] = valueToJSON(it)
		}

		return m
	case valNone, valUndefined:
		return nil
	default:
		return nil
	}
}

func evalTest(e testExpr, ctx *execCtx) (value, error) {
	left, err := evalExpr(e.left, ctx)
	if err != nil {
		return value{}, err
	}

	var ok bool
	switch e.name {
	case "defined":
		ok = left.kind != valUndefined
	case "none":
		ok = left.kind == valNone
	default:
		return value{}, errf("неизвестный тест is %s", e.name)
	}

	if e.not {
		ok = !ok
	}

	return boolVal(ok), nil
}

func (v value) truthy() bool {
	switch v.kind {
	case valUndefined, valNone:
		return false
	case valBool:
		return v.b
	case valNumber:
		return v.num != 0
	case valString:
		return v.str != ""
	case valArray:
		return len(v.arr) > 0
	case valObject, valNamespace:
		return len(v.obj) > 0
	default:
		return false
	}
}

func (v value) toString() string {
	switch v.kind {
	case valUndefined, valNone:
		return ""
	case valBool:
		if v.b {
			return "true"
		}
		return "false"
	case valNumber:
		if math.Mod(v.num, 1) == 0 {
			return strconv.Itoa(int(v.num))
		}

		return strconv.FormatFloat(v.num, 'f', -1, 64)
	case valString:
		return v.str
	default:
		return ""
	}
}

func (v value) toArray() []value {
	if v.kind == valArray {
		return v.arr
	}

	return nil
}

func (v value) getAttr(name string) value {
	if v.kind == valObject || v.kind == valNamespace {
		if sub, ok := v.obj[name]; ok {
			return sub
		}
	}

	if v.kind == valString {
		switch name {
		case "split", "lstrip", "rstrip", "strip":
			return objectVal(map[string]value{
				"__method__": strVal(name),
				"__self__":   v,
			})
		}
	}

	if v.kind == valObject && v.obj != nil {
		if m, ok := v.obj["__method__"]; ok && m.str != "" {
			return objectVal(map[string]value{"__bound__": strVal(m.str), "__self__": v.obj["__self__"]})
		}
	}

	return undefinedVal()
}

func (v value) getIndex(key value) value {
	switch v.kind {
	case valArray:
		i := int(key.num)
		if i >= 0 && i < len(v.arr) {
			return v.arr[i]
		}
	case valObject, valNamespace:
		if key.kind == valString {
			if sub, ok := v.obj[key.str]; ok {
				return sub
			}
		}
	}

	return undefinedVal()
}

func valuesEqual(a, b value) bool {
	if a.kind == valString && b.kind == valString {
		return a.str == b.str
	}

	if a.kind == valNumber && b.kind == valNumber {
		return a.num == b.num
	}

	if a.kind == valBool && b.kind == valBool {
		return a.b == b.b
	}

	if a.kind == valNone && b.kind == valNone {
		return true
	}

	return a.toString() == b.toString()
}

func callStringMethod(self value, method string, argExprs []expr, ctx *execCtx) (value, error) {
	if self.kind != valString {
		return value{}, errf("метод %s только для строк", method)
	}

	args := make([]value, len(argExprs))
	for i, a := range argExprs {
		v, err := evalExpr(a, ctx)
		if err != nil {
			return value{}, err
		}
		args[i] = v
	}

	switch method {
	case "split":
		sep := "\n"
		if len(args) > 0 {
			sep = args[0].toString()
		}
		parts := strings.Split(self.str, sep)
		items := make([]value, len(parts))
		for i, p := range parts {
			items[i] = strVal(p)
		}
		return arrayVal(items), nil
	case "lstrip":
		chars := "\n"
		if len(args) > 0 {
			chars = args[0].toString()
		}
		return strVal(strings.TrimLeft(self.str, chars)), nil
	case "rstrip":
		chars := "\n"
		if len(args) > 0 {
			chars = args[0].toString()
		}
		return strVal(strings.TrimRight(self.str, chars)), nil
	case "strip":
		chars := "\n"
		if len(args) > 0 {
			chars = args[0].toString()
		}
		return strVal(strings.Trim(self.str, chars)), nil
	default:
		return value{}, errf("неизвестный метод %s", method)
	}
}
