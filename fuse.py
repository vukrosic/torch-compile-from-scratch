"""Part 3 - fuse. Pointwise ops touch each element once, so a chain of them can run as one loop."""
from graph import Node

POINTWISE = {'add', 'sub', 'mul', 'div', 'neg', 'tanh', 'exp'}


def fuse(g):
    """Group the graph into kernels: runs of pointwise ops merge, anything else stands alone."""
    kernels, run = [], []
    for n in g.nodes:
        if n.op == 'input':
            continue
        if n.op in POINTWISE:
            run.append(n)
        else:
            kernels += [run, [n]] if run else [[n]]
            run = []
    return kernels + [run] if run else kernels


def traffic(g):
    """Whole-array reads + writes. Eager: every op reads its inputs and writes its result.
    Fused: read each input once, write the output once."""
    eager = sum(1 + sum(isinstance(a, Node) for a in n.args) for n in g.nodes if n.op != 'input')
    fused = len(g.inputs) + 1
    return eager, fused
