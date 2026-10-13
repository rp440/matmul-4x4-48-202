"""Exact, typed DAG admission: validate every operand before tensor contraction.

A, B and product-linear spaces never share coordinates. Unexpected phases are
rejected, not projected away. verify_data is read-only; verify() writes a
<file>.verification.json report only when write_report=True.
"""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import sys


def scalar(x, *, argument=False):
    if isinstance(x, bool) or not isinstance(x, (int, float, str, F)):
        raise ValueError(f'Invalid scalar: {x!r}')
    # In this format integer gate arguments denote node references.
    if argument and isinstance(x, int):
        raise ValueError('Integer scalar/node argument ambiguity')
    try:
        return F(x)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as ex:
        raise ValueError(f'Invalid finite rational scalar: {x!r}') from ex


def operation_counts(data):
    counts = Counter(n['op'] for n in data['nodes'])
    q = data.get('block_size', 4)
    scales = data.get('output_scales', [1] * (q*q))
    result = {'products': counts['matmul'],
              'additions': counts['add'] + counts['subtract'],
              'scalings': counts['multiply'] + counts['divide'],
              'negations': counts['negative'],
              'output_scale_ops': sum(scalar(c) != 1 for c in scales),
              'product_alpha_ops': sum(scalar(n.get('alpha', 1)) != 1
                                       for n in data['nodes'] if n['op'] == 'matmul')}
    result['linear_ops'] = sum(result[k] for k in
        ('additions', 'scalings', 'negations', 'output_scale_ops', 'product_alpha_ops'))
    result['cost_model'] = 'Each explicit add/subtract/scale/negation, nonunit output scale and product alpha costs one; products separate.'
    return result


def verify_data(data):
    q = data.get('block_size', 4)
    if type(q) is not int or not 1 <= q <= 8:
        raise ValueError('block_size must be an integer in [1,8]')
    n = q*q
    off = data.get('input_b_offset', 64 if q == 8 else 16)
    if type(off) is not int or off < n:
        raise ValueError('Overlapping or invalid input_b_offset')
    nodes, outputs = data['nodes'], data['outputs']
    scales = data.get('output_scales', [1]*n)
    if not isinstance(nodes, list) or not isinstance(outputs, list) or not isinstance(scales, list):
        raise ValueError('nodes, outputs and output_scales must be lists')
    if len(outputs) != n or len(scales) != n:
        raise ValueError('Output/output-scale length must equal block_size squared')
    values = {i: ('L', {i: F(1)}) for i in range(n)}
    values.update({off+i: ('R', {i: F(1)}) for i in range(n)})
    depth = {i: 0 for i in values}
    products = []
    phase_metadata = []
    max_intermediate = F(1)
    arities = {'add': 2, 'subtract': 2, 'negative': 1,
               'multiply': 2, 'divide': 2, 'matmul': 2}
    for node in nodes:
        idx, op, args = node['id'], node['op'], node['args']
        if type(idx) is not int or idx < 0 or idx in values:
            raise ValueError(f'Duplicate or invalid node ID: {idx!r}')
        if op not in arities or not isinstance(args, list) or len(args) != arities[op]:
            raise ValueError(f'Invalid operation/arity at {idx}')
        refs = args[:1] if op in ('multiply', 'divide') else args
        if any(type(a) is not int or a not in values for a in refs):
            raise ValueError(f'Unknown, forward or noninteger node reference at {idx}')
        phase, av = values[args[0]]
        if op == 'matmul':
            if phase != 'L' or values[args[1]][0] != 'R':
                raise ValueError(f'Product operands must be A-linear then B-linear at {idx}')
            alpha = scalar(node.get('alpha', 1))
            left = {i: alpha*c for i, c in av.items() if alpha*c}
            products.append((left, values[args[1]][1]))
            phase, value = 'P', {len(products)-1: F(1)}
            expected_phase = 'GEMM'
        else:
            expected_phase = phase
            if op in ('add', 'subtract'):
                other_phase, bv = values[args[1]]
                if phase != other_phase:
                    raise ValueError(f'Cross-phase linear operation at {idx}')
                value = av.copy()
                for i, c in bv.items():
                    value[i] = value.get(i, F(0)) + (c if op == 'add' else -c)
            else:
                c = F(-1) if op == 'negative' else scalar(args[1], argument=True)
                if op == 'divide':
                    if not c:
                        raise ValueError(f'Division by zero at {idx}')
                    c = 1/c
                value = {i: c*v for i, v in av.items()}
            if 'alpha' in node:
                raise ValueError('alpha is valid only on matmul nodes')
        annotation = node.get('phase', expected_phase)
        # Legacy compositions place pure operand scalings in the GEMM
        # scheduling region. Semantic L/R/P types remain operand-derived.
        legacy_gemm_scale = (annotation == 'GEMM'
                             and op in ('multiply', 'divide', 'negative'))
        if legacy_gemm_scale:
            phase_metadata.append({'node': idx, 'declared': annotation, 'semantic': phase})
        if annotation != expected_phase and not legacy_gemm_scale:
            raise ValueError(f'Incorrect phase annotation at {idx}')
        value = {i: c for i, c in value.items() if c}
        values[idx] = phase, value
        depth[idx] = 1 + max(depth[a] for a in refs)
        max_intermediate = max(max_intermediate, max(map(abs, value.values()), default=F(0)))
    growth, coefficients, denominators = [], [], []
    for left, right in products:
        growth.append(sum(map(abs, left.values())) * sum(map(abs, right.values())))
        coefficients.extend(left.values()); coefficients.extend(right.values())
    E = F(0)
    for k, (idx, scale) in enumerate(zip(outputs, scales)):
        if type(idx) is not int or idx not in values or values[idx][0] != 'P':
            raise ValueError(f'Output {k} must reference a product-linear value')
        scale = scalar(scale)
        weights = {i: scale*c for i, c in values[idx][1].items() if scale*c}
        coefficients.extend(weights.values())
        got = {}
        for r, weight in weights.items():
            left, right = products[r]
            for a, u in left.items():
                for b, v in right.items():
                    got[a, b] = got.get((a, b), F(0)) + weight*u*v
        got = {pair: c for pair, c in got.items() if c}
        truth = {(k//q*q+j, j*q+k%q): F(1) for j in range(q)}
        if got != truth:
            raise ValueError(f'Exact tensor mismatch at output {k}')
        E = max(E, sum(abs(c)*growth[r] for r, c in weights.items()))
    return {'arithmetic': 'Typed Fraction spaces; strict schema, topology and phase admission; complete sparse tensor equality without projection',
            'equations_checked': n**3, 'mismatches': 0, 'block_size': q,
            'rank': len(products), 'max_tensor_factor_coefficient': str(max(map(abs, coefficients), default=F(0))),
            'max_intermediate_linear_coefficient': str(max_intermediate), 'E': str(E),
            'max_dag_depth': max(depth.values()),
            'common_denominator': math.lcm(*(c.denominator for c in coefficients)),
            'phase_metadata_exceptions': phase_metadata,
            'operation_counts': operation_counts(data)}


def verify(path, *, write_report=False, quiet=False):
    path = Path(path)
    raw = path.read_bytes()
    report = {'file': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
              **verify_data(json.loads(raw))}
    if write_report:
        Path(str(path)+'.verification.json').write_text(json.dumps(report, indent=2))
    if not quiet:
        print(report)
    return report


if __name__ == '__main__':
    for path in sys.argv[1:]:
        verify(path)
