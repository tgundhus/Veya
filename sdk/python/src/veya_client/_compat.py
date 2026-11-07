"""Redirect legacy imports without creating duplicate classes or module state."""
import importlib
import importlib.abc
import importlib.util
import sys


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Import machinery assigns the alias spec even when create_module returns
        # an existing module. Keep relative imports and package metadata canonical.
        module.__spec__ = self.canonical_spec

    def get_code(self, fullname):
        spec = importlib.util.find_spec(self.target)
        return spec.loader.get_code(self.target)


class _AliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self, legacy, target):
        self.legacy = legacy
        self.target = target

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self.legacy + "."):
            name = self.target + fullname[len(self.legacy):]
            spec = importlib.util.find_spec(name)
            if spec is not None:
                return importlib.util.spec_from_loader(
                    fullname, _AliasLoader(name), is_package=spec.submodule_search_locations is not None)
        return None


def install_alias(legacy, target):
    sys.meta_path.insert(0, _AliasFinder(legacy, target))
    sys.modules[legacy] = importlib.import_module(target)
