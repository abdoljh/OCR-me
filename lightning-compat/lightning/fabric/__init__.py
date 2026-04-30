# kraken imports `from lightning.fabric import Fabric` at module load time.
# Delegate to lightning_fabric (shipped inside pytorch-lightning 2.6.1).
from lightning_fabric.fabric import Fabric  # noqa: F401
from lightning_fabric import *  # noqa: F401, F403
