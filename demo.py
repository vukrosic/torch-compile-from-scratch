"""Run every part on GPT-2's GELU and time it against eager numpy."""
import json, math, time
import numpy as np
from graph import trace, tanh
from lower import lower
from fuse import fuse, traffic
import ctypes
from codegen import codegen, codegen_unfused, build
from tinycompile import tiny_compile


def gelu(x):  # GPT-2's tanh GELU (minGPT's NewGELU), written with our tiny "torch"
    return 0.5 * x * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def best(f, x, reps=20):
    f(x)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(x); ts.append(time.perf_counter() - t)
    return min(ts) * 1000


out = {}
g = trace(gelu)
print('== 1. captured graph ==')
print(g)
lo = lower(g)
print('\n== 2. lowered graph (pow -> mul) ==')
print(lo)
ks = fuse(lo)
eager_passes, fused_passes = traffic(g)
print(f'\n== 3. fusion: {len(ks)} kernel(s) from {sum(map(len, ks))} ops ==')
print(f'whole-array reads+writes: eager {eager_passes}, fused {fused_passes}')

N = 10_000_000
print('\n== 4. generated C ==')
print(codegen(lo, N))

x = np.random.default_rng(0).standard_normal(N).astype(np.float32)
fast = tiny_compile(gelu)
t = time.perf_counter(); y = fast(x); first = (time.perf_counter() - t) * 1000
err = float(np.abs(y - gelu(x)).max())
unfused_k = build(codegen_unfused(lo, N))
buf = np.empty(N, np.float32)
unfused = lambda x: unfused_k(ctypes.c_void_p(x.ctypes.data), ctypes.c_void_p(buf.ctypes.data))
unfused(x)
err_u = float(np.abs(buf - gelu(x)).max())
te, tu, tc = best(gelu, x), best(unfused, x), best(fast, x)
print(f'== 5. {N:,} floats ==')
print(f'first call (trace+compile): {first:.0f} ms')
print(f'eager numpy:          {te:7.2f} ms')
print(f'C, one loop per op:   {tu:7.2f} ms   (max error {err_u:.2e})')
print(f'C, fused (tiny_compile): {tc:4.2f} ms   (max error {err:.2e})')
print(f'speedup vs eager {te / tc:.1f}x, fusion alone {tu / tc:.1f}x')

print('\n== 6. guards ==')
f1, shapes = tiny_compile(gelu), []
for n in [8, 16, 8, 32]:
    f1(x[:n]); shapes.append(f1.compiles); print(f'batch {n:2d} -> compiles so far: {f1.compiles}')
fast(x); print('same shape again -> compiles so far:', fast.compiles)
fast(x[:1000]); print('new shape (1000,) -> compiles so far:', fast.compiles)
x64 = x.astype(np.float64)
print('new dtype float64 -> compiles so far:', (fast(x64), fast.compiles)[1], ' max error', float(np.abs(fast(x64) - gelu(x64)).max()))


def relu_if(x):
    y = x * 2.0
    if y.sum() > 0:  # needs real numbers to decide
        return y
    return -y


print('\n== 7. graph break ==')
f2 = tiny_compile(relu_if)
f2(x)

out.update(n=N, ops=len([n for n in g.nodes if n.op != 'input']), lowered_ops=len([n for n in lo.nodes if n.op != 'input']),
           kernels=len(ks), eager_passes=eager_passes, fused_passes=fused_passes,
           eager_ms=round(te, 2), unfused_ms=round(tu, 2), compiled_ms=round(tc, 2), speedup=round(te / tc, 1),
           first_call_ms=round(first), shape_compiles=shapes, max_err=err, graph=str(g).split('\n'), lowered=str(lo).split('\n'),
           c_src=codegen(lo, N).split('\n'), c_unfused=codegen_unfused(lo, N).split('\n'))
json.dump(out, open('demo.json', 'w'), indent=1)
