# Compatibility shim: maps the quarantined `lightning` package to
# pytorch-lightning / lightning_fabric (both ship in pytorch-lightning 2.6.1).
from lightning_fabric.utilities.seed import seed_everything  # noqa: F401
import lightning_fabric as fabric  # noqa: F401
