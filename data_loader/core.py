r"""Helpers to load datasets and handle them effortlessly."""


# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code


import random
from typing import List, Optional, Union

import numpy
import torch
import torchvision

import lrp.norm

# Turn on multi-process data loading by setting the number of workers
# to a non-zero positive integer value.
# TODO: Set value dynamically.
DATA_LOADER_NUM_WORKERS: int = 2


def _seed_worker(_: int) -> None:
    r"""Seed the worker with a random seed.

    Function imported from PyTorch tutorial:
      https://pytorch.org/docs/stable/notes/randomness.html#dataloader-workers

    :param worker_id: The worker id.
    """
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def _create_mask_for_single_dataset_class(dataset: torch.utils.data.Dataset,
                                          class_idx_or_name: Union[int, str]
                                          ) -> torch.Tensor:
    r"""Create a mask for a specific list of classes, which can be used to filter the dataset.

    :param dataset: The dataset.
    :param class_idx_or_name: The class index or name to be included in the mask.

    :return: Mask with classes marked as True.
    """
    if not hasattr(dataset, 'class_to_idx') or not hasattr(dataset, 'targets'):
        raise ValueError(
            'The dataset does not have a \'class_to_idx\' or a \'targets\' attribute.')

    class_idx: int
    if isinstance(class_idx_or_name, str):
        class_idx = dataset.class_to_idx[class_idx_or_name]
    elif isinstance(class_idx_or_name, int):
        class_idx = class_idx_or_name
    else:
        raise ValueError(
            'The class index or name must be an integer or a string.')

    # Filter which includes only the targets of a certain class in the dataset.
    return torch.Tensor(dataset.targets) == class_idx


def _create_mask_for_dataset_classes(dataset: torch.utils.data.Dataset,
                                     classes: Union[List[str], List[int]]) -> torch.Tensor:
    r"""Create a mask for a specific list of classes, which can be used to filter the dataset.

    :param dataset: The dataset.
    :param classes: The list of classes to be included in the mask.

    :return: Mask with classes marked as True.
    """
    # Filter which includes only the targets of a certain class in the dataset.
    mask: torch.Tensor = _create_mask_for_single_dataset_class(
        dataset, classes[0])

    # Iterate over the remaining classes and add them to the mask.
    for clss in classes[1:]:
        # Create filter for an additional class and add to existing mask.
        mask = mask | _create_mask_for_single_dataset_class(dataset, clss)

    return mask


def _imagenet_dataset(root: str) -> torch.utils.data.Dataset:
    r"""Retrieve the ImageNet dataset.

    :param root: The root directory where the dataset is stored.

    :return: The ImageNet dataset.
    """
    return torchvision.datasets.ImageNet(root=root,
                                         split='val', transform=lrp.norm.ImageNetNorm.transform)


def _data_loader(dataset: torch.utils.data.Dataset,
                 batch_size: int,
                 seed: int = 0) -> torch.utils.data.DataLoader:
    r"""Return a data loader for a specific dataset.

    Args:
        batch_size (int): The batch size.
        seed (int): The seed to use.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """
    # Init generator with specific seed for reproducibility.
    generator: torch.Generator = torch.Generator()
    generator.manual_seed(seed)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=DATA_LOADER_NUM_WORKERS,
                                       worker_init_fn=_seed_worker,
                                       generator=generator,
                                       pin_memory=True)


def imagenet_data_loader(root: str,
                         batch_size: int,
                         classes: Optional[Union[List[str], List[int]]] = None,
                         seed: int = 0
                         ) -> torch.utils.data.DataLoader:
    r"""Return a data loader for the ImageNet dataset.

    :param root: The root directory where the dataset is stored.
    :param batch_size: Number of images to load in each batch.
    :param classes: List of classes to load. If None, all classes are loaded.
    :param seed: Seed to use for reproducibility.

    :return: ImageNet data loader.
    """
    # Dataset with all classes included.
    dataset: torch.utils.data.Dataset = _imagenet_dataset(root)

    # If classes are specified, filter the dataset.
    if classes:
        # Create a mask which includes only the classes specified.
        mask: torch.Tensor = _create_mask_for_dataset_classes(dataset, classes)

        # Filter the dataset to include only the classes of interest.
        dataset = torch.utils.data.Subset(dataset, mask.nonzero())

    return _data_loader(dataset=dataset, batch_size=batch_size, seed=seed)
