"""Part 2 - lower. Rewrite ops the code generator doesn't know into ones it does."""
from graph import Graph, Node, Proxy, exp

DECOMPOSE = {
    'pow': lambda x, k: x * x * x if k == 3 else x * x if k == 2 else None,
    'sigmoid': lambda x: 1.0 / (1.0 + exp(-x)),
}


def lower(g):
    new, env = Graph(), {}
    for n in g.nodes:
        if n.op == 'input':
            env[n] = Proxy(new, new.input(n.name))
            continue
        args = [env[a] if isinstance(a, Node) else a for a in n.args]
        rule = DECOMPOSE.get(n.op)
        out = rule(*args) if rule else None
        if out is None:  # no rule: copy the op as it is
            out = next(a for a in args if isinstance(a, Proxy)).op(n.op, *args)
        env[n] = out
    new.output = env[g.output].node
    return new
