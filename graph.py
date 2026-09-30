"""Part 1 - capture. Run the function once on fake tensors and record every operation into a graph."""
import inspect
import numpy as np


class Node:
    def __init__(self, name, op, args):
        self.name, self.op, self.args = name, op, args  # args: Nodes or plain numbers


class Graph:
    def __init__(self):
        self.nodes, self.inputs, self.output = [], [], None

    def input(self, name):
        node = Node(name, 'input', [])
        self.nodes.append(node)
        self.inputs.append(node)
        return node

    def add(self, op, args):
        node = Node(f't{len(self.nodes) - len(self.inputs)}', op, args)
        self.nodes.append(node)
        return node

    def __str__(self):
        show = lambda a: a.name if isinstance(a, Node) else repr(a)
        rows = [f'{n.name} = {n.op}({", ".join(map(show, n.args))})' for n in self.nodes]
        return '\n'.join(rows + [f'return {self.output.name}'])


class GraphBreak(Exception):
    pass


class Proxy:
    """A fake tensor. It holds no numbers, only the graph node that made it."""
    def __init__(self, graph, node):
        self.graph, self.node = graph, node

    def op(self, name, *args):
        args = [a.node if isinstance(a, Proxy) else a for a in args]
        return Proxy(self.graph, self.graph.add(name, args))

    def __add__(self, o): return self.op('add', self, o)
    def __radd__(self, o): return self.op('add', o, self)
    def __sub__(self, o): return self.op('sub', self, o)
    def __rsub__(self, o): return self.op('sub', o, self)
    def __mul__(self, o): return self.op('mul', self, o)
    def __rmul__(self, o): return self.op('mul', o, self)
    def __truediv__(self, o): return self.op('div', self, o)
    def __rtruediv__(self, o): return self.op('div', o, self)
    def __pow__(self, o): return self.op('pow', self, o)
    def __neg__(self): return self.op('neg', self)
    def __gt__(self, o): return self.op('gt', self, o)
    def sum(self): return self.op('sum', self)

    def __bool__(self):
        raise GraphBreak('an if on tensor values: the graph cannot know which way it goes')


# the few library functions our "torch" has: real math on arrays, a new node on proxies
def tanh(x): return x.op('tanh', x) if isinstance(x, Proxy) else np.tanh(x)
def exp(x): return x.op('exp', x) if isinstance(x, Proxy) else np.exp(x)
def sigmoid(x): return x.op('sigmoid', x) if isinstance(x, Proxy) else 1 / (1 + np.exp(-x))


def trace(fn):
    g = Graph()
    args = [Proxy(g, g.input(name)) for name in inspect.signature(fn).parameters]
    g.output = fn(*args).node
    return g
