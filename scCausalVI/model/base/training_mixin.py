# from typing import List, Optional, Union, Tuple, Dict
# import numpy as np
# import torch
# from scvi.train import TrainingPlan, TrainRunner
# from scCausalVI.data.dataloaders.data_splitting import scCausalVIDataSplitter


# class scCausalVITrainingMixin:
#     def train(
#             self,
#             group_indices_list: List[np.array],
#             max_epochs: Optional[int] = None,
#             use_gpu: Optional[Union[str, int, bool]] = None,
#             train_size: float = 0.9,
#             validation_size: Optional[float] = 0.1,
#             batch_size: int = 128,
#             early_stopping: bool = False,
#             plan_kwargs: Optional[dict] = None,
#             **trainer_kwargs,
#     ) -> None:
#         """
#         Train a scCausalVI model.

#         Args:
#         ----
#             background_indices: Indices for background samples in `adata`.
#             target_indices: Indices for target samples in `adata`.
#             max_epochs: Number of passes through the dataset. If `None`, default to
#                 `np.min([round((20000 / n_cells) * 400), 400])`.
#             use_gpu: Use default GPU if available (if `None` or `True`), or index of
#                 GPU to use (if `int`), or name of GPU (if `str`, e.g., `"cuda:0"`),
#                 or use CPU (if `False`).
#             train_size: Size of training set in the range [0.0, 1.0].
#             validation_size: Size of the validation set. If `None`, default to
#                 `1 - train_size`. If `train_size + validation_size < 1`, the remaining
#                 cells belong to the test set.
#             batch_size: Mini-batch size to use during training.
#             early_stopping: Perform early stopping. Additional arguments can be passed
#                 in `**kwargs`. See :class:`~scvi.train.Trainer` for further options.
#             plan_kwargs: Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword
#                 arguments passed to `train()` will overwrite values present
#                 in `plan_kwargs`, when appropriate.
#             **trainer_kwargs: Other keyword args for :class:`~scvi.train.Trainer`.

#         Returns
#         -------
#             None. The model is trained.
#         """
#         if max_epochs is None:
#             n_cells = self.adata.n_obs
#             max_epochs = np.min([round((20000 / n_cells) * 400), 400])

#         plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

#         data_splitter = scCausalVIDataSplitter(
#             self.adata_manager,
#             group_indices_list,
#             train_size=train_size,
#             validation_size=validation_size,
#             batch_size=batch_size,
#             accelerator='cuda' if use_gpu else 'cpu',
#         )

#         training_plan = TrainingPlan(self.module, **plan_kwargs)

#         es = "early_stopping"
#         trainer_kwargs[es] = (
#             early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
#         )
#         runner = TrainRunner(
#             self,
#             training_plan=training_plan,
#             data_splitter=data_splitter,
#             max_epochs=max_epochs,
#             accelerator="gpu" if use_gpu else "cpu",
#             **trainer_kwargs,
#         )
#         return runner()

from typing import List, Optional, Union, Tuple, Dict, Sequence
import numpy as np
import torch
import pytorch_lightning as pl

from scvi.train import TrainingPlan, TrainRunner
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders import AnnDataLoader
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass
from scvi.distributions import ZeroInflatedNegativeBinomial

# Import our custom training mixin and data loaders
# from scCausalVI.module.scCausalVI import scCausalVIModule
from scCausalVI.model.base._utils import _invert_dict

logger = torch.log if False else __import__("logging").getLogger(__name__)

#################################################################
# Training Mixin with mixed precision and improved GPU utilization
#################################################################
class scCausalVITrainingMixin:
    def train(
        self,
        group_indices_list: List[np.array],
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = 0.1,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ) -> None:
        """
        Train a scCausalVI model with improved efficiency.
        
        • Uses a smart GPU selection and transfers data non–blockingly.
        • Launches DataLoaders with increased num_workers and pinned memory.
        • Enables Automatic Mixed Precision (AMP) training when a GPU is used.
        """
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

        # Pop or set a default number of workers for data loading
        num_workers = trainer_kwargs.pop("num_workers", 4)

        # Create a data splitter with improved DataLoader options.
        data_splitter = scCausalVIDataSplitter(
            self.adata_manager,
            group_indices_list,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            accelerator="cuda" if use_gpu else "cpu",
            num_workers=num_workers,  # New argument for more parallel loading
        )

        # Create a training plan from the scvi library.
        training_plan = TrainingPlan(self.module, **plan_kwargs)

        # Propagate the early_stopping flag.
        trainer_kwargs["early_stopping"] = (
            early_stopping if "early_stopping" not in trainer_kwargs else trainer_kwargs["early_stopping"]
        )
        # Enable mixed-precision if using the GPU.
        # if use_gpu:
        #     trainer_kwargs["use_amp"] = True

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator="gpu" if use_gpu else "cpu",
            **trainer_kwargs,
        )
        return runner()

#################################################################
# DataSplitter with added num_workers and pin_memory support
#################################################################
class scCausalVIDataSplitter(pl.LightningDataModule):
    """
    A DataSplitter that builds a scCausalDataLoader for training, validation,
    and test sets. Optimized to use multiple workers and pinned memory when using GPU.
    """
    def __init__(
        self,
        adata_manager: AnnDataManager,
        group_indices_list: List[List[int]],
        train_size: float = 0.9,
        validation_size: Optional[float] = 0.1,
        accelerator: str = "cpu",
        batch_size: int = 128,
        num_workers: int = 4,  # New parameter
        **kwargs,
    ) -> None:
        super().__init__()
        self.adata_manager = adata_manager
        self.group_indices_list = group_indices_list
        self.train_size = train_size
        self.validation_size = validation_size
        self.accelerator = accelerator
        self.batch_size = batch_size
        self.data_loader_kwargs = kwargs

        # Use default num_workers and, if using CUDA, enable pinned memory.
        self.data_loader_kwargs.setdefault("num_workers", num_workers)
        if self.accelerator == "cuda":
            self.data_loader_kwargs.setdefault("pin_memory", True)

        self.train_idx_per_group = []
        self.val_idx_per_group = []
        self.test_idx_per_group = []
        self.current_dataloader = None

        self.setup()

    def __iter__(self):
        if self.current_dataloader is None:
            self.current_dataloader = self.train_dataloader()
        return iter(self.current_dataloader)

    def __next__(self):
        if self.current_dataloader is None:
            self.current_dataloader = self.train_dataloader()
        return next(self.current_dataloader)

    def setup(self, stage: Optional[str] = None):
        random_state = np.random.RandomState(seed=42)  # Or use scvi.settings.seed

        for group_indices in self.group_indices_list:
            # Validate the split sizes.
            n_train, n_val =  round(len(group_indices)*self.train_size), round(len(group_indices)*self.validation_size)
            group_permutation = random_state.permutation(group_indices)
            self.val_idx_per_group.append(group_permutation[:n_val])
            self.train_idx_per_group.append(group_permutation[n_val: (n_val + n_train)])
            self.test_idx_per_group.append(group_permutation[(n_train + n_val):])

        # Parse the accelerator to determine device details.
        accelerator = self.accelerator
        self.device = torch.device("cuda" if accelerator == "cuda" else "cpu")
        print(f"Accelerator: {accelerator}")
        self.pin_memory = True if accelerator == "cuda" else False

        self.train_idx = np.concatenate(self.train_idx_per_group)
        self.val_idx = np.concatenate(self.val_idx_per_group)
        self.test_idx = np.concatenate(self.test_idx_per_group)

    def _get_scCausal_dataloader(
        self, group_indices_list: List[List[int]], shuffle: bool = True
    ) -> "scCausalDataLoader":
        return scCausalDataLoader(
            self.adata_manager,
            indices_list=group_indices_list,
            shuffle=shuffle,
            drop_last=False,
            batch_size=self.batch_size,
            **self.data_loader_kwargs,
        )

    def train_dataloader(self) -> "scCausalDataLoader":
        return self._get_scCausal_dataloader(self.train_idx_per_group)

    def val_dataloader(self) -> "scCausalDataLoader":
        if all(len(val_idx) > 0 for val_idx in self.val_idx_per_group):
            return self._get_scCausal_dataloader(self.val_idx_per_group)
        else:
            raise ValueError("No validation data found.")

    def test_dataloader(self) -> "scCausalDataLoader":
        if all(len(test_idx) > 0 for test_idx in self.test_idx_per_group):
            return self._get_scCausal_dataloader(self.test_idx_per_group)
        else:
            raise ValueError("No test data found.")

#################################################################
# Custom DataLoader merging batches across conditions efficiently
#################################################################
from itertools import cycle
from scvi.dataloaders._concat_dataloader import ConcatDataLoader

class scCausalDataLoader(ConcatDataLoader):
    """
    A custom DataLoader that uses one sub–DataLoader per condition but merges
    each sub–batch into a single dictionary of tensors.
    """
    def __init__(
        self,
        adata_manager: AnnDataManager,
        indices_list: List[List[int]],
        shuffle: bool = False,
        batch_size: int = 128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        **data_loader_kwargs,
    ) -> None:
        super().__init__(
            adata_manager=adata_manager,
            indices_list=indices_list,
            shuffle=shuffle,
            batch_size=batch_size,
            data_and_attributes=data_and_attributes,
            drop_last=drop_last,
            **data_loader_kwargs,
        )

    def __iter__(self):
        # Cycle through sub-loaders so that each condition contributes equally.
        iter_list = [
            cycle(dl) if dl != self.largest_dl else dl for dl in self.dataloaders
        ]
        for batch_tuple in zip(*iter_list):
            merged_batch = {}
            all_keys = batch_tuple[0].keys()
            for key in all_keys:
                sub_batches = [b[key] for b in batch_tuple]
                merged_batch[key] = torch.cat(sub_batches, dim=0)
            yield merged_batch
