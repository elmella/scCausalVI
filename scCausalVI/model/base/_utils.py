from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
import numpy as np
from collections.abc import Iterable as IterableClass
from typing import Sequence, Union, Dict
from scvi._types import Number

from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    BATCH_KEY: str = "batch"
    LABELS_KEY: str = "labels"
    CONDITION_KEY: str = "condition"
    PROTEIN_EXP_KEY: str = "proteins"
    CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"
    MINIFY_TYPE_KEY: str = "minify_type"
    LATENT_QZM_KEY: str = "latent_qzm"
    LATENT_QZV_KEY: str = "latent_qzv"
    OBSERVED_LIB_SIZE: str = "observed_lib_size"


class _METRIC_KEYS_NT(NamedTuple):
    TRAINING_KEY: str = "training"
    VALIDATION_KEY: str = "validation"
    # classification
    ACCURACY_KEY: str = "accuracy"
    F1_SCORE_KEY: str = "f1_score"
    CALIBRATION_ERROR_KEY: str = "calibration_error"
    AUROC_KEY: str = "auroc"
    CLASSIFICATION_LOSS_KEY: str = "classification_loss"
    TRUE_LABELS_KEY: str = "true_labels"
    LOGITS_KEY: str = "logits"


SCCAUSALVI_REGISTRY_KEYS = _REGISTRY_KEYS_NT()
SCCAUSALVI_METRIC_KEYS = _METRIC_KEYS_NT()


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


def _invert_dict(d: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in d.items()}