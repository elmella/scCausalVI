from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
import numpy as np
from collections.abc import Iterable as IterableClass
from typing import Sequence, Union
from scvi._types import Number


def _get_batch_code_from_category(
        adata_manager: AnnDataManager, category: Sequence[Union[Number, str]]
):
    if not isinstance(category, IterableClass) or isinstance(category, str):
        category = [category]

    batch_mappings = adata_manager.get_state_registry(
        REGISTRY_KEYS.BATCH_KEY
    ).categorical_mapping
    batch_code = []
    for cat in category:
        if cat is None:
            batch_code.append(None)
        elif cat not in batch_mappings:
            raise ValueError(f'"{cat}" not a valid batch category.')
        else:
            batch_loc = np.where(batch_mappings == cat)[0][0]
            batch_code.append(batch_loc)
    return batch_code
