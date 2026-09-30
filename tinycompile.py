"""Part 5 - the wrapper. Capture, lower, fuse and generate once, then reuse it while the guards hold."""
from graph import trace, GraphBreak
from lower import lower
from fuse import fuse
from codegen import compile_graph


def tiny_compile(fn):
    cache = {}

    def compiled(*xs):
        guard = tuple((x.shape, x.dtype.name) for x in xs)
        if guard not in cache:
            compiled.compiles += 1
            try:
                g = lower(trace(fn))
                assert len(fuse(g)) == 1, 'one pointwise kernel only'
                cache[guard] = compile_graph(g, xs[0].size, xs[0].dtype.name)
            except GraphBreak as why:
                print('graph break:', why, '-> running eagerly')
                cache[guard] = fn
        return cache[guard](*xs)

    compiled.compiles = 0
    return compiled
