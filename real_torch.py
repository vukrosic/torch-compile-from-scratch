"""The same GELU through the real torch.compile: the FX graph Dynamo captures, graph breaks, and speed.
Run with TORCH_LOGS=output_code to also print the kernel Inductor writes."""
import json, math, time
import torch


def gelu(x):  # GPT-2's tanh GELU (minGPT's NewGELU)
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def relu_if(x):
    y = x * 2.0
    if y.sum() > 0:
        return y
    return -y


def show_graph(gm, example_inputs):  # a "backend" that only prints what Dynamo hands it
    gm.graph.print_tabular()
    return gm.forward


def best(f, x, reps=20):
    f(x)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(x); ts.append(time.perf_counter() - t)
    return min(ts) * 1000


x = torch.randn(10_000_000)
print('== FX graph Dynamo captured ==')
torch.compile(gelu, backend=show_graph)(x)

print('\n== graph breaks ==')
ex = torch._dynamo.explain(relu_if)(x)
print(f'graphs: {ex.graph_count}  graph breaks: {ex.graph_break_count}')
for b in ex.break_reasons:
    print('reason:', b.reason.splitlines()[0])

print('\n== recompiles ==')
from torch._dynamo.testing import CompileCounter
cnt, shapes = CompileCounter(), []
f1 = torch.compile(gelu, backend=cnt)
for n in [8, 16, 8, 32]:
    f1(torch.randn(n)); shapes.append(cnt.frame_count); print(f'batch {n:2d} -> compiles so far: {cnt.frame_count}')

torch._dynamo.reset()
c = torch.compile(gelu)
t = time.perf_counter(); y = c(x); first = (time.perf_counter() - t) * 1000
err = (y - gelu(x)).abs().max().item()
te, tc = best(gelu, x), best(c, x)
print(f'\n== {x.numel():,} floats, torch {torch.__version__}, {torch.get_num_threads()} threads ==')
print(f'first call (compile): {first:.0f} ms')
print(f'eager: {te:.2f} ms   compiled: {tc:.2f} ms   speedup {te / tc:.1f}x   max error {err:.2e}')
json.dump(dict(torch=torch.__version__, threads=torch.get_num_threads(), first_call_ms=round(first), eager_ms=round(te, 2),
               compiled_ms=round(tc, 2), speedup=round(te / tc, 1), max_err=err, graphs=ex.graph_count,
               graph_breaks=ex.graph_break_count, shape_compiles=shapes), open('real_torch.json', 'w'), indent=1)
