from __future__ import annotations

import ast

MODULE_BASE_NAMES = {"Module"}


def _base_is_nn_module(base: ast.expr) -> bool:
    # matches `nn.Module`, `torch.nn.Module`, bare `Module`, or subclassing another
    # class is not resolved statically — direct nn.Module bases only.
    if isinstance(base, ast.Attribute):
        return base.attr in MODULE_BASE_NAMES
    if isinstance(base, ast.Name):
        return base.id in MODULE_BASE_NAMES
    return False


def find_module_classes(files: dict[str, str]) -> list[dict[str, str]]:
    """Statically find nn.Module subclasses across uploaded source files.

    Returns [{"class_name", "file"}], including classes whose base is itself a
    locally-defined nn.Module subclass (one level of transitivity).
    """
    candidates: list[dict[str, str]] = []
    local_module_classes: set[str] = set()

    parsed: dict[str, ast.Module] = {}
    for path, content in files.items():
        try:
            parsed[path] = ast.parse(content)
        except SyntaxError:
            continue

    # first pass: direct nn.Module subclasses
    for path, tree in parsed.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(_base_is_nn_module(base) for base in node.bases):
                candidates.append({"class_name": node.name, "file": path})
                local_module_classes.add(node.name)

    # second pass: classes extending a locally-found module class
    for path, tree in parsed.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(isinstance(base, ast.Name) and base.id in local_module_classes for base in node.bases):
                if not any(c["class_name"] == node.name and c["file"] == path for c in candidates):
                    candidates.append({"class_name": node.name, "file": path})

    return candidates
