# Compatibility shim: maps the quarantined `lightning` package to
# pytorch-lightning / lightning_fabric (both ship in pytorch-lightning 2.6.1).
#
# The kraken library needs two lightning namespaces at runtime:
#   • lightning.fabric   — used by kraken.lib.vgsl.model (inference path)
#   • lightning.pytorch  — used by kraken.lib.progress (CLI path)
#
# pytorch-lightning ships these as lightning_fabric and pytorch_lightning
# respectively.  The meta path finder below transparently redirects every
# `import lightning.pytorch[.*]` to the corresponding pytorch_lightning module.
import sys
import importlib
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec


class _PLAliasFinder(MetaPathFinder):
    """Redirect `lightning.pytorch[.*]` imports to `pytorch_lightning[.*]`."""

    _PREFIX = "lightning.pytorch"
    _TARGET = "pytorch_lightning"

    def find_spec(self, fullname, path, target=None):
        if fullname != self._PREFIX and not fullname.startswith(self._PREFIX + "."):
            return None
        pl_name = self._TARGET + fullname[len(self._PREFIX):]
        try:
            pl_mod = importlib.import_module(pl_name)
        except ImportError:
            return None
        sys.modules[fullname] = pl_mod
        return ModuleSpec(fullname, _PassthroughLoader())


class _PassthroughLoader(Loader):
    def create_module(self, spec):
        return sys.modules.get(spec.name)

    def exec_module(self, module):
        pass  # module is already the pytorch_lightning one; nothing to do


if not any(isinstance(f, _PLAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _PLAliasFinder())

# lightning.fabric is handled by lightning/fabric/__init__.py (existing shim).
from lightning_fabric.utilities.seed import seed_everything  # noqa: F401
