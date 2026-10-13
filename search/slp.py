"""Exact PLinOpt-format linear programs (SLPs), operation counting, and the
4x4 matrix-multiplication admission check.

Programs are lists of (target, expression_text). Maps returned by eval_map are
sparse; load_progs returns dense Fraction matrices, with rows as outputs.
SLP indices are zero-based. For the complete scheme, i[4*r+c] means A[r,c] or
B[r,c], and P's o[4*r+c] means C[r,c].
Products pair L's o[k] with R's o[k]; P's i[k] is that product. No transpose.

Counting is syntactic on linear values: binary +/- costs one, nonunit scalar
multiply/divide costs one, and unary minus costs one. Constant arithmetic and
signs on constants are free. A right-hand unary minus in +/- is absorbed by
flipping the binary operation. Thus (a+b)/2 costs one add and one scaling;
-a-b costs one negation and one subtract; a+(-b) costs one subtract only.
Copies and multiplication/division by +1 cost zero. Multiplication/division
by -1 costs one negation, agreeing with DAG export and verify_dag.
No circuit optimization or common-subexpression elimination is performed.
"""

import argparse
import ast
from fractions import Fraction as F
import json
from pathlib import Path
import re


def _tree(text):
    """Read only names, integer/rational constants and linear arithmetic."""
    def read(node):
        if isinstance(node, ast.Name):
            if not re.fullmatch(r'[A-Za-z_]\w*', node.id):
                raise ValueError('Invalid variable name')
            return node.id
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return F(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = read(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            return -value if isinstance(value, F) else ('negative', value)
        if isinstance(node, ast.BinOp):
            left, right = read(node.left), read(node.right)
            ops = {ast.Add: 'add', ast.Sub: 'subtract',
                   ast.Mult: 'multiply', ast.Div: 'divide'}
            op = ops.get(type(node.op))
            if op is None:
                raise ValueError('Unsupported operator')
            if isinstance(left, F) and isinstance(right, F):
                if op == 'add':
                    return left + right
                if op == 'subtract':
                    return left - right
                if op == 'multiply':
                    return left * right
                return left / right
            if op in ('add', 'subtract'):
                if isinstance(left, F) or isinstance(right, F):
                    raise ValueError('Scalar offsets are not homogeneous linear expressions')
                if isinstance(right, tuple) and right[0] == 'negative':
                    op = 'subtract' if op == 'add' else 'add'
                    right = right[1]
            else:
                if op == 'multiply' and isinstance(left, F):
                    left, right = right, left
                if not isinstance(right, F) or isinstance(left, F):
                    raise ValueError('Multiply/divide requires a linear value and scalar')
                if op == 'divide' and not right:
                    raise ValueError('Division by zero')
            return op, left, right
        raise ValueError('Unsupported expression syntax')

    try:
        tree = read(ast.parse(text, mode='eval').body)
    except (SyntaxError, ZeroDivisionError) as ex:
        raise ValueError(f'Invalid expression: {text!r}') from ex
    if isinstance(tree, F):
        raise ValueError('Assignment must be homogeneous linear, not scalar')
    return tree


def parse_slp(path):
    """Parse every semicolon-terminated assignment; reject unrecognized text."""
    text = Path(path).read_text()
    statements = text.split(';')
    if statements.pop().strip():
        raise ValueError('Missing final semicolon')
    prog = []
    for statement in statements:
        match = re.fullmatch(r'\s*([A-Za-z_]\w*)\s*:=\s*(.*?)\s*',
                             statement, re.S)
        if match is None:
            raise ValueError(f'Invalid assignment: {statement!r}')
        target, expr = match.groups()
        if re.fullmatch(r'i\d+', target):
            raise ValueError('Cannot assign to an input')
        _tree(expr)
        prog.append((target, expr))
    return prog


def _outputs(env):
    return {int(name[1:]): value for name, value in env.items()
            if re.fullmatch(r'o\d+', name)}


def eval_map(prog, n_inputs):
    """Evaluate sequential assignments exactly; undefined variables are errors."""
    env = {f'i{k}': {k: F(1)} for k in range(n_inputs)}

    def evaluate(tree):
        if isinstance(tree, str):
            if tree not in env:
                raise ValueError(f'Undefined variable: {tree}')
            return env[tree]
        op, *args = tree
        left = evaluate(args[0])
        if op in ('add', 'subtract'):
            result = left.copy()
            sign = 1 if op == 'add' else -1
            for k, value in evaluate(args[1]).items():
                result[k] = result.get(k, F(0)) + sign * value
        else:
            scale = F(-1) if op == 'negative' else args[1]
            if op == 'divide':
                scale = 1 / scale
            result = {k: scale * value for k, value in left.items()}
        return {k: value for k, value in result.items() if value}

    for target, expr in prog:
        if re.fullmatch(r'i\d+', target):
            raise ValueError('Cannot assign to an input')
        env[target] = evaluate(_tree(expr))
    return _outputs(env)


def count_ops(prog):
    """Count the convention in the module docstring, including dead assignments."""
    counts = dict(adds=0, scalings=0, negations=0)

    def count(tree):
        if isinstance(tree, str):
            return
        op, *args = tree
        count(args[0])
        if op in ('add', 'subtract'):
            counts['adds'] += 1
            count(args[1])
        elif op == 'negative':
            counts['negations'] += 1
        else:
            counts['negations'] += args[1] == -1
            counts['scalings'] += abs(args[1]) != 1

    for _, expr in prog:
        count(_tree(expr))
    return {**counts, 'total': sum(counts.values())}


def _dense(outputs, n_inputs):
    if set(outputs) != set(range(len(outputs))):
        raise ValueError('Output indices must be contiguous from zero')
    return [[outputs[r].get(c, F(0)) for c in range(n_inputs)]
            for r in range(len(outputs))]


def load_progs(Lprog, Rprog, Pprog):
    """Evaluate L, R, P programs to dense exact maps."""
    L = _dense(eval_map(Lprog, 16), 16)
    R = _dense(eval_map(Rprog, 16), 16)
    P = _dense(eval_map(Pprog, len(L)), len(L))
    return L, R, P


def verify_tensor(L, R, P):
    """Check every C[c], A[a], B[b] coefficient; raise on the first mismatch."""
    rank = len(L)
    if (not rank or len(R) != rank or len(P) != 16
            or any(len(row) != 16 for row in L + R)
            or any(len(row) != rank for row in P)):
        raise ValueError('Expected rank x 16, rank x 16, 16 x rank matrices')
    if any(not isinstance(x, (int, F)) or isinstance(x, bool)
           for matrix in (L, R, P) for row in matrix for x in row):
        raise ValueError('Tensor coefficients must be exact integers or Fractions')
    left = [[(a, F(x)) for a, x in enumerate(row) if x] for row in L]
    right = [[(b, F(x)) for b, x in enumerate(row) if x] for row in R]
    for c, row in enumerate(P):
        got = {}
        for k, weight in enumerate(row):
            if weight:
                for a, u in left[k]:
                    for b, v in right[k]:
                        got[a, b] = got.get((a, b), F(0)) + weight * u * v
        for a in range(16):
            for b in range(16):
                want = int(a // 4 == c // 4 and a % 4 == b // 4 and b % 4 == c % 4)
                if got.get((a, b), F(0)) != want:
                    raise ValueError(f'Exact tensor mismatch at C[{c}], A[{a}], B[{b}]')
    return dict(equations_checked=4096, mismatches=0, rank=rank)


def to_dag(Lprog, Rprog, Pprog):
    """Return a JSON-serializable verify_dag dictionary; constants are strings."""
    nodes = []

    def gate(op, args):
        idx = 32 + len(nodes)
        nodes.append(dict(id=idx, op=op, args=args))
        return idx

    def emit(prog, inputs):
        env = {f'i{k}': idx for k, idx in enumerate(inputs)}

        def expression(tree):
            if isinstance(tree, str):
                if tree not in env:
                    raise ValueError(f'Undefined variable: {tree}')
                return env[tree]
            op, *args = tree
            left = expression(args[0])
            if op in ('multiply', 'divide'):
                if args[1] == 1:
                    return left
                if args[1] == -1:
                    return gate('negative', [left])
                return gate(op, [left, str(args[1])])
            if op == 'negative':
                return gate(op, [left])
            return gate(op, [left, expression(args[1])])

        for target, expr in prog:
            if re.fullmatch(r'i\d+', target):
                raise ValueError('Cannot assign to an input')
            env[target] = expression(_tree(expr))
        outputs = _outputs(env)
        if set(outputs) != set(range(len(outputs))):
            raise ValueError('Output indices must be contiguous from zero')
        return [outputs[k] for k in range(len(outputs))]

    left = emit(Lprog, range(16))
    right = emit(Rprog, range(16, 32))
    if not left or len(left) != len(right):
        raise ValueError('L and R must have equal nonzero output counts')
    products = [gate('matmul', [a, b]) for a, b in zip(left, right)]
    outputs = emit(Pprog, products)
    if len(outputs) != 16:
        raise ValueError('P must have 16 outputs')
    return dict(block_size=4, input_b_offset=16, nodes=nodes, outputs=outputs)


def verify_candidate(prefix_or_progs):
    """Single admission path: exact tensor, typed DAG, and recount.

    Accept a file prefix or an iterable of exactly three parsed programs.
    Return reports only; never write files or declare a candidate an improvement.
    """
    from verify_dag import verify_data

    if isinstance(prefix_or_progs, (str, Path)):
        progs = [parse_slp(f'{prefix_or_progs}_{phase}.slp') for phase in 'LRP']
    else:
        progs = list(prefix_or_progs)
    if len(progs) != 3:
        raise ValueError('Expected L, R, P programs')
    tensor = verify_tensor(*load_progs(*progs))
    dag = verify_data(to_dag(*progs))
    counts = [count_ops(prog) for prog in progs]
    total = sum(count['total'] for count in counts)
    if dag['operation_counts']['linear_ops'] != total:
        raise ValueError('SLP/DAG operation count mismatch')
    return dict(tensor=tensor, dag=dag, counts=dict(zip('LRP', counts)), total=total)


def main():
    from verify_dag import verify_data

    parser = argparse.ArgumentParser(description='Exact verification of an L/R/P SLP triple.')
    parser.add_argument('prefix', type=Path, help='path prefix: PREFIX_L.slp, PREFIX_R.slp, PREFIX_P.slp')
    args = parser.parse_args()
    progs = [parse_slp(f'{args.prefix}_{phase}.slp') for phase in 'LRP']
    counts = [count_ops(prog) for prog in progs]
    for phase, count in zip('LRP', counts):
        print(f'{phase}: {json.dumps(count)}')
    total = sum(count['total'] for count in counts)
    print(f'Total linear ops: {total}')
    expected = re.fullmatch(r'4x4x4_\d+_(\d+)', args.prefix.name)
    if expected and total != int(expected[1]):
        print(f'DISCREPANCY: filename advertises {expected[1]}, counted {total}')
    print('Indices: A, B, C row-major; P output is not transposed; product k = L[k]*R[k]')
    admission = verify_candidate(progs)
    print(f'Tensor: {json.dumps(admission["tensor"])}')
    report = verify_data(to_dag(*progs))
    print(f'DAG: {report["equations_checked"]} identities verified')
    print(f'operation_counts: {json.dumps(report["operation_counts"])}')
    assert report['operation_counts']['linear_ops'] == total


if __name__ == '__main__':
    main()
