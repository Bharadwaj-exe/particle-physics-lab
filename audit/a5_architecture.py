"""Audit A5: static import graph of particlelab, cycles, layer violations, global state,
duplicated constants, broad exception handlers."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "particlelab"
mods = {}
for f in ROOT.rglob("*.py"):
    rel = f.relative_to(ROOT.parent).with_suffix("")
    name = ".".join(rel.parts).removesuffix(".__init__")
    mods[name] = f


def resolve(mod, node):
    pkg = mod.split(".") if mods[mod].name == "__init__.py" else mod.split(".")[:-1]
    if node.level:
        base = pkg[: len(pkg) - node.level + 1]
        return ".".join(base + ([node.module] if node.module else []))
    return node.module


edges, ext, broad, globals_, consts = {}, {}, [], [], []
for m, f in mods.items():
    tree = ast.parse(f.read_text(encoding="utf-8"))
    deps = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            tgt = resolve(m, node)
            if tgt and tgt.startswith("particlelab"):
                deps.add(tgt)
            elif tgt:
                ext.setdefault(m, set()).add(tgt.split(".")[0])
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("particlelab"):
                    deps.add(a.name)
                else:
                    ext.setdefault(m, set()).add(a.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None or (isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException")):
                broad.append(f"{m}:{node.lineno}")
        elif isinstance(node, ast.Global):
            globals_.append(f"{m}:{node.lineno}")
    edges[m] = sorted(d for d in deps if d != m)
    src = f.read_text(encoding="utf-8")
    for pat in (r"0\.2997", r"0\.3\b\s*\*", r"0\.1056", r"13\.6", r"91\.18", r"3\.0969", r"9\.460"):
        for ln, line in enumerate(src.splitlines(), 1):
            if re.search(pat, line) and not line.strip().startswith("#"):
                consts.append(f"{m}:{ln}: {line.strip()[:90]}")


def layer(m):
    p = m.split(".")
    return p[1] if len(p) > 1 else "root"


# cycles at module level (DFS)
cycles = []
def dfs(n, stack, seen):
    for d in edges.get(n, []):
        d2 = d if d in edges else ".".join(d.split(".")[:-1])
        if d2 in stack:
            cycles.append(stack[stack.index(d2):] + [d2])
        elif d2 not in seen and d2 in edges:
            seen.add(d2); dfs(d2, stack + [d2], seen)
for m in edges:
    dfs(m, [m], {m})

layer_edges = sorted({(layer(a), layer(b)) for a, ds in edges.items() for b in ds if layer(a) != layer(b)})
order = ["units", "domain", "physics", "detector", "analysis", "data", "visualization", "provenance", "app"]
OUT = dict(module_edges=edges, layer_edges=layer_edges,
           external_by_module={k: sorted(v) for k, v in ext.items()},
           qt_or_pyvista_outside_app=[m for m, v in ext.items() if (v & {"PySide6", "pyvista", "pyvistaqt", "pyqtgraph", "vtk"}) and layer(m) != "app"],
           cycles=sorted({" -> ".join(c) for c in cycles}),
           broad_exception_handlers=broad, global_statements=globals_, hardcoded_constants=consts)
Path(__file__).with_name("results").joinpath("a5_architecture.json").write_text(json.dumps(OUT, indent=2))
print(json.dumps({k: v for k, v in OUT.items() if k != "module_edges"}, indent=2))
