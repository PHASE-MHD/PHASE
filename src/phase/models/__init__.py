"""Neural-operator models shipped with PHASE."""

from .checkpoint_mapping import map_official_tfno_state_dict
from .model_factory import create_model
from .scot_mhd import PoseidonMHDFinetune

try:
    from .tfno import TFNO
except ModuleNotFoundError as exc:
    if exc.name != "tltorch":
        raise
    TFNO = None

__all__ = [
    "PoseidonMHDFinetune",
    "TFNO",
    "create_model",
    "map_official_tfno_state_dict",
]
