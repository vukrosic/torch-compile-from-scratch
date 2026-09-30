# torch.compile from scratch

This repo builds a tiny `torch.compile` in about 200 lines of Python. It turns GPT-2's GELU into one fused C loop and runs it through the same five steps PyTorch uses:

| Part | File | Lines | What it does |
| --- | --- | --- | --- |
| 1. Capture | `graph.py` | 72 | Runs the function once on fake tensors (proxies) and records every operation into a graph. An `if` on tensor values raises `GraphBreak`. |
| 2. Lower | `lower.py` | 23 | Rewrites ops the code generator doesn't know (`pow`, `sigmoid`) into ones it does (`mul`, `div`, `exp`). |
| 3. Fuse | `fuse.py` | 26 | Groups pointwise ops into one kernel: 18 whole-array reads and writes become 2. |
| 4. Codegen | `codegen.py` | 60 | Writes one C loop, compiles it with `cc -O3`, and loads it with `ctypes`. |
| 5. Guard | `tinycompile.py` | 25 | Keys a cache on each input's shape and dtype. A new key compiles again; a graph break falls back to plain Python. |

`demo.py` runs every part and times the result. `real_torch.py` runs the same checks with the real `torch.compile`.

This is the companion code for [Build Torch Compile from Scratch — Full Course](https://youtu.be/Wg9lNKOTJ6g) on [Vuk Rosić’s YouTube channel](https://www.youtube.com/@vukrosic).

The video is currently private; the link will become viewable when it is published.

## Run it

You need Python 3, numpy and a C compiler on your `PATH` as `cc` (clang or gcc).

```bash
python3 demo.py
```

To compare with PyTorch (tested with torch 2.8.0):

```bash
pip install torch
python3 real_torch.py
TORCH_LOGS=output_code python3 real_torch.py   # also prints the kernel Inductor writes
```

## Results in the video

These were measured on an Apple M4 MacBook: 10,000,000 float32 values, best of 20 runs. Timings vary about ±30% from run to run. The full logs are in [`results/`](results/).

| | ms |
| --- | --- |
| eager numpy | 115.61 |
| C, one loop per op | 45.63 |
| C, fused (`tiny_compile`) | 13.61 |
| first call (trace + compile) | 829 |

The fused kernel was 8.5x faster than eager numpy; fusion alone gave 3.4x. Real `torch.compile` (4 threads) measured 27.48 ms eager against 10.66 ms compiled (2.6x), with a first call of 1031 ms.

## Try it yourself

1. **Add an op.** Make `sin` compile and fuse. Add a `sin` function in `graph.py` next to `tanh` and `exp`. Add `'sin'` to `POINTWISE` in `fuse.py`. Add a C template to `C` in `codegen.py`, then add `'sin'` to the `('tanh', 'exp')` tuple so `float` gets `sinf`. `fuse(g)` should still return one kernel.
2. **Break the graph.** Write a function with `if x.sum() > 0:`. Check that `tiny_compile` prints `graph break` and still returns the right answer.
3. **Watch the guards.** Call a compiled function with shapes `(8,)`, `(16,)`, `(8,)`, `(32,)` and print `.compiles` after each call. Predict each count before you run it.

## What real torch.compile adds

TorchDynamo captures graphs from Python bytecode, not proxies, so it can resume after a graph break. AOT Autograd also traces the backward pass. TorchInductor handles reductions, matmuls, strides, dynamic shapes and GPU code (Triton). The pipeline is the same one this repo builds: capture, lower, fuse, generate code, guard. For the full design, see Ansel et al., *PyTorch 2* (ASPLOS 2024).

MIT License.
