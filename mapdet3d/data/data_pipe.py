"""DataPipe wraps datasets to share the prepossessing pipeline."""

from __future__ import annotations

import bisect
import random
from collections.abc import Callable, Iterable

from torch.utils.data import ConcatDataset, Dataset

from .reference import MultiViewDataset
from .transforms.base import TFunctor
from .typing import DictData, DictDataOrList


class DataPipe(ConcatDataset[DictDataOrList]):
    """DataPipe class.

    This class wraps one or multiple instances of a PyTorch Dataset so that the
    preprocessing steps can be shared across those datasets. Composes dataset
    and the preprocessing pipeline.
    """

    def __init__(
        self,
        datasets: Dataset[DictDataOrList] | Iterable[Dataset[DictDataOrList]],
        preprocess_fn: Callable[
            [list[DictData]], list[DictData]
        ] = lambda x: x,
    ):
        """Creates an instance of the class.

        Args:
            datasets (Dataset | Iterable[Dataset]): Dataset(s) to be wrapped by
                this data pipeline.
            preprocess_fn (Callable[[list[DictData]], list[DictData]]):
                Preprocessing function of a single sample. It takes a list of
                samples and returns a list of samples. Defaults to identity
                function.
        """
        if isinstance(datasets, Dataset):
            datasets = [datasets]
        super().__init__(datasets)
        self.preprocess_fn = preprocess_fn

        self.has_reference = any(
            _check_reference(dataset) for dataset in datasets
        )

        if self.has_reference and not all(
            _check_reference(dataset) for dataset in datasets
        ):
            raise ValueError(
                "All datasets must be MultiViewDataset / has reference if "
                + "one of them is."
            )

    def __getitem__(self, idx: int) -> DictDataOrList:
        """Wrap getitem to apply augmentations."""
        samples = super().__getitem__(idx)
        if isinstance(samples, list):
            return self.preprocess_fn(samples)

        return self.preprocess_fn([samples])[0]


class MultiSampleDataPipe(DataPipe):
    """MultiSampleDataPipe class.

    This class wraps DataPipe to support augmentations that require multiple
    images (e.g., Mosaic and Mixup) by sampling additional indices for each
    image. NUM_SAMPLES needs to be defined as a class attribute for transforms
    that require multi-sample augmentation.
    """

    def __init__(
        self,
        datasets: Dataset[DictDataOrList] | Iterable[Dataset[DictDataOrList]],
        preprocess_fn: list[list[TFunctor]],
    ):
        """Creates an instance of the class.

        Args:
            datasets (Dataset | Iterable[Dataset]): Dataset(s) to be wrapped by
                this data pipeline.
            preprocess_fn (list[list[TFunctor]]): Preprocessing functions of a
                single sample. Different than DataPipe, this is a list of lists
                of transformation functions. The inner list is for transforms
                that needs to share the same sampled indices (e.g.,
                GenMosaicParameters and MosaicImages), and the outer list is
                for different transforms.
        """
        super().__init__(datasets)
        self.preprocess_fns = preprocess_fn

    def _sample_indices(self, idx: int, num_samples: int) -> list[int]:
        """Sample additional indices for multi-sample augmentation."""
        indices = [idx]
        for _ in range(1, num_samples):
            indices.append(random.randint(0, len(self) - 1))
        return indices

    def __getitem__(self, idx: int) -> DictDataOrList:
        """Wrap getitem to apply augmentations."""
        samples = super(DataPipe, self).__getitem__(idx)
        if not isinstance(samples, list):
            samples = [samples]
            single_view = True
        else:
            single_view = False

        for preprocess_fn in self.preprocess_fns:
            if hasattr(preprocess_fn[0], "NUM_SAMPLES"):
                num_samples = preprocess_fn[0].NUM_SAMPLES
                aug_inds = self._sample_indices(idx, num_samples)
                add_samples = [
                    super(DataPipe, self).__getitem__(ind)
                    for ind in aug_inds[1:]
                ]
                prep_samples = []
                for i, samp in enumerate(samples):
                    prep_samples.append(samp)
                    prep_samples += [
                        s[i] if isinstance(s, list) else s for s in add_samples
                    ]
            else:
                num_samples = 1
                prep_samples = samples
            for prep_fn in preprocess_fn:
                prep_samples = prep_fn.apply_to_data(prep_samples)  # type: ignore # pylint: disable=line-too-long
            samples = prep_samples[::num_samples]
        return samples[0] if single_view else samples


def _check_reference(dataset: Dataset[DictDataOrList]) -> bool:
    """Check if the datasets have reference."""
    has_reference = (
        dataset.has_reference if hasattr(dataset, "has_reference") else False
    )
    return has_reference or isinstance(dataset, MultiViewDataset)


class TupleDataPipe(ConcatDataset):
    """A custom ConcatDataset that supports indexing with a tuple.

    Modified from VGGT's TupleConcatDataset.

    Standard PyTorch ConcatDataset only accepts an integer index. This class
    extends that functionality to allow passing a tuple like (sample_idx,
    num_images, aspect_ratio), where the first element is used to determine
    which sample to fetch, and the full tuple is passed down to the selected
    dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring
    the provided index. This is useful during training when shuffling the
    entire dataset might cause memory issues due to duplicating dictionaries.
    If doing this, you can set PyTorch's dataloader shuffle to False.
    """

    def __init__(self, datasets, inside_random: bool = False):
        """Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to
                concatenate.
            common_config (dict): Common configuration dict, used to check for
                random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = inside_random

    def __getitem__(self, idx):
        """Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the
                sequence index across the concatenated datasets, and the rest
                are passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have
                exactly three elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]
