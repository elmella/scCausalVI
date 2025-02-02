from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import torch
from scvi.train import TrainingPlan, TrainRunner
from scCausalVI.data.dataloaders.data_splitting import scCausalVIDataSplitter


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
        Train a scCausalVI model.

        Args:
        ----
            background_indices: Indices for background samples in `adata`.
            target_indices: Indices for target samples in `adata`.
            max_epochs: Number of passes through the dataset. If `None`, default to
                `np.min([round((20000 / n_cells) * 400), 400])`.
            use_gpu: Use default GPU if available (if `None` or `True`), or index of
                GPU to use (if `int`), or name of GPU (if `str`, e.g., `"cuda:0"`),
                or use CPU (if `False`).
            train_size: Size of training set in the range [0.0, 1.0].
            validation_size: Size of the validation set. If `None`, default to
                `1 - train_size`. If `train_size + validation_size < 1`, the remaining
                cells belong to the test set.
            batch_size: Mini-batch size to use during training.
            early_stopping: Perform early stopping. Additional arguments can be passed
                in `**kwargs`. See :class:`~scvi.train.Trainer` for further options.
            plan_kwargs: Keyword args for :class:`~scvi.train.TrainingPlan`. Keyword
                arguments passed to `train()` will overwrite values present
                in `plan_kwargs`, when appropriate.
            **trainer_kwargs: Other keyword args for :class:`~scvi.train.Trainer`.

        Returns
        -------
            None. The model is trained.
        """
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = scCausalVIDataSplitter(
            self.adata_manager,
            group_indices_list,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            accelerator='cuda' if use_gpu else 'cpu',
        )

        training_plan = TrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator="gpu" if use_gpu else "cpu",
            **trainer_kwargs,
        )
        return runner()
