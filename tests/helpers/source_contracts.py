"""Compare numerical and execution definitions across namespace relocation."""

import ast
import copy
import hashlib


def definition_digest(source, name, module, routes=None, modules=None, *, removal_delegate=False):
    """Hash syntax, normalizing import identities and omitting documentation."""
    routes, modules = routes or {}, modules or {}
    tree = ast.parse(source)
    bindings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bindings[alias.asname or alias.name] = node.module + "." + alias.name
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bindings[node.name] = module + "." + node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Name):
                        bindings[child.id] = module + "." + child.id

    def canonical(path):
        for key in sorted(routes, key=len, reverse=True):
            if path == key or path.startswith(key + "."):
                if routes[key]:
                    return routes[key] + "." + key.rsplit(".", 1)[1] + path[len(key):]
        for key in sorted(modules, key=len, reverse=True):
            if path == key or path.startswith(key + "."):
                return modules[key] + path[len(key):]
        return path

    class Normalize(ast.NodeTransformer):
        def __init__(self):
            self.locals = []

        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

        def visit_Pass(self, node):
            return None

        def visit_FunctionDef(self, node):
            node.decorator_list = [self.visit(d) for d in node.decorator_list]
            node.args.defaults = [self.visit(d) for d in node.args.defaults]
            node.args.kw_defaults = [self.visit(d) if d else None for d in node.args.kw_defaults]
            for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                if arg.annotation:
                    arg.annotation = self.visit(arg.annotation)
            if node.returns:
                node.returns = self.visit(node.returns)
            local = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            local.update(n.arg for n in ast.walk(node.args) if isinstance(n, ast.arg))
            self.locals.append(local)
            node.body = [self.visit(n) for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.Pass)) and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
            self.locals.pop()
            return node

        def visit_Call(self, node):
            # The sole non-namespace adaptation in the retained stage is its
            # injected selector. Compare that delegation to the original call;
            # retain all arguments, operators, branches and numerical values.
            if (removal_delegate and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self" and node.func.attr == "removable"
                    and len(node.args) == 1 and not node.keywords):
                return ast.Call(
                    func=ast.Name(id="castle_reasoning.data.plans.removable", ctx=ast.Load()),
                    args=[ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()),
                                        attr="plan", ctx=ast.Load()), self.visit(node.args[0])],
                    keywords=[],
                )
            return self.generic_visit(node)

        def visit_Attribute(self, node):
            parts = []
            head = node
            while isinstance(head, ast.Attribute):
                parts.append(head.attr)
                head = head.value
            if isinstance(head, ast.Name) and head.id in bindings and not any(head.id in scope for scope in self.locals):
                return ast.Name(id=canonical(bindings[head.id] + "".join("." + p for p in reversed(parts))), ctx=node.ctx)
            return self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in bindings and not any(node.id in scope for scope in self.locals):
                return ast.Name(id=canonical(bindings[node.id]), ctx=node.ctx)
            return node

    selected = tree
    for part in name.split("."):
        selected = next(n for n in selected.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
    normalized = Normalize().visit(copy.deepcopy(selected))
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()
