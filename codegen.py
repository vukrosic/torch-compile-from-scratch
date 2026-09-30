"""Part 4 - codegen. Turn the fused graph into one C loop, compile it, and load it into Python."""
import ctypes, os, subprocess, sys, tempfile
import numpy as np
from graph import Node

C = {'add': '{} + {}', 'sub': '{} - {}', 'mul': '{} * {}', 'div': '{} / {}',
     'neg': '-{}', 'tanh': 'tanh{}({})', 'exp': 'exp{}({})'}
# on a Mac, let clang call Apple's vectorized tanhf/expf instead of one-at-a-time libm
FLAGS = ['-O3', '-ffast-math'] + (['-fveclib=Accelerate', '-framework', 'Accelerate'] if sys.platform == 'darwin' else [])
TYPES = {'float32': ('float', 'f'), 'float64': ('double', '')}  # C type, suffix for literals and math


def codegen(g, n, dtype='float32'):
    T, sfx = TYPES[dtype]
    arg = lambda a: a.name if isinstance(a, Node) else f'{float(a)}{sfx}'
    line = lambda v: C[v.op].format(*([sfx] if v.op in ('tanh', 'exp') else []), *map(arg, v.args))
    params = ', '.join(f'const {T}* in_{x.name}' for x in g.inputs)
    body = [f'{T} {x.name} = in_{x.name}[i];' for x in g.inputs]
    body += [f'{T} {v.name} = {line(v)};' for v in g.nodes if v.op != 'input']
    body += [f'out[i] = {g.output.name};']
    return ('#include <math.h>\n'
            f'void kernel({params}, {T}* out) {{\n'
            f'    for (long i = 0; i < {n}; i++) {{\n'
            + ''.join(f'        {s}\n' for s in body) +
            '    }\n}\n')


def build(src):
    folder = tempfile.mkdtemp(prefix='tinycompile-')
    c, so = os.path.join(folder, 'kernel.c'), os.path.join(folder, 'kernel.so')
    open(c, 'w').write(src)
    subprocess.run(['cc', *FLAGS, '-shared', '-fPIC', c, '-o', so], check=True)
    return ctypes.CDLL(so).kernel


def compile_graph(g, n, dtype):
    kernel = build(codegen(g, n, dtype))

    def run(*xs):
        out = np.empty(n, dtype)
        kernel(*[ctypes.c_void_p(x.ctypes.data) for x in xs], ctypes.c_void_p(out.ctypes.data))
        return out
    return run


def codegen_unfused(g, n, dtype='float32'):
    """Same C and compiler, but one loop per op, each writing a full array: what eager does."""
    T, sfx = TYPES[dtype]
    arg = lambda a: f'{a.name}[i]' if isinstance(a, Node) else f'{float(a)}{sfx}'
    line = lambda v: C[v.op].format(*([sfx] if v.op in ('tanh', 'exp') else []), *map(arg, v.args))
    ops = [v for v in g.nodes if v.op != 'input']
    src = '#include <math.h>\n#include <stdlib.h>\n'
    src += f'void kernel(const {T}* {g.inputs[0].name}, {T}* out) {{\n'
    for v in ops:
        name = 'out' if v is g.output else v.name
        if v is not g.output:
            src += f'    {T}* {v.name} = malloc({n} * sizeof({T}));\n'
        src += f'    for (long i = 0; i < {n}; i++) {name}[i] = {line(v)};\n'
    src += ''.join(f'    free({v.name});\n' for v in ops if v is not g.output)
    return src + '}\n'
