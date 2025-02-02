from itertools import cycle
from typing import List, Optional, Union
import torch

from scvi.data import AnnDataManager
from scvi.dataloaders._concat_dataloader import ConcatDataLoader


class scCausalDataLoader(ConcatDataLoader):
    """
    A custom loader that still uses one sub-DataLoader per condition, but
    merges each sub-batch into a single dictionary, ensuring each group
    contributes exactly `batch_size` samples.

    Args:
    ----
        adata: AnnData object that has been registered via `setup_anndata`.
        indices_list: List where each element is a list of indices in the adata to load
        shuffle: Whether the data should be shuffled.
        batch_size: Mini-batch size to load for background and target data.
        data_and_attributes: Dictionary with keys representing keys in data
            registry (`adata.uns["_scvi"]`) and value equal to desired numpy
            loading type (later made into torch tensor). If `None`, defaults to all
            registered data.
        drop_last: If int, drops the last batch if its length is less than
            `drop_last`. If `drop_last == True`, drops last non-full batch.
            If `drop_last == False`, iterate over all batches.
        **data_loader_kwargs: Keyword arguments for `torch.utils.data.DataLoader`.
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
        """
        Iter method for scCausalVI data loader.

        Will iter over the dataloader with the most data while cycling through
        the data in the other dataloaders. Merge
        """
        iter_list = [
            cycle(dl) if dl != self.largest_dl else dl
            for dl in self.dataloaders
        ]

        for batch_tuple in zip(*iter_list):
            merged_batch = {}
            all_keys = batch_tuple[0].keys()
            for key in all_keys:
                sub_batches = [b[key] for b in batch_tuple]
                merged_batch[key] = torch.cat(sub_batches, dim=0)
                
            yield merged_batch
