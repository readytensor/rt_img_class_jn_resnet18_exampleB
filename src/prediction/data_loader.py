import os
import joblib
from typing import Tuple, Optional, List
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


class DataLoaderFactory:
    """
    A factory class for creating DataLoader instances tailored for image classification
    tasks with PyTorch.

    This class facilitates the creation of DataLoader objects for training and
    (optionally) validation datasets with customizable preprocessing steps, including
    resizing, normalization, and optionally splitting datasets into training and
    validation sets. It supports loading datasets from a directory structure where
    each subdirectory represents a class.
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 6,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        image_size: Tuple[int, int] = (224, 224),
        validation_size: float = 0.0,
    ):
        """
        Initializes a DataLoaderFactory with the specified configurations.

        Args:
            batch_size: The number of samples in each batch.
            num_workers: The number of subprocesses to use for data loading.
            mean: The mean for each channel for normalization.
            std: The standard deviation for each channel for normalization.
            image_size: The target size of the images (width, height).
            validation_size: The fraction of the training set to use as validation.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        self.image_size = tuple(image_size)
        self.validation_size = validation_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.num_classes = None

    def create_data_loaders(
        self,
        data_dir_path: str,
        validation_dir_path: str = None,
        val_size: float = 0.0,
        shuffle: bool = False,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Creates DataLoader objects for training and, if specified, validation datasets.

        This method automatically applies predefined transformations, encapsulates the
        datasets into DataLoader objects for batched processing, and determines whether
        to generate a validation split. A validation DataLoader is created either from
        a provided validation dataset directory or by splitting the training dataset,
        depending on the factory's configuration and the arguments passed.

        Args:
            data_dir_path: The path to the training dataset directory.
            validation_dir_path: Optional; the path to the validation dataset directory. 
                                If provided, the validation dataset is loaded from this
                                directory. Otherwise, a validation split can be created
                                from the training dataset.
            val_size: The proportion of the dataset to allocate to validation when
                      splitting.
            shuffle: Whether to shuffle the training dataset. Validation dataset is
                     not shuffled.

        Returns:
            A tuple of DataLoaders for the training and validation datasets. The 
            validation DataLoader is None if no validation dataset is provided or
            created.
        """
        create_validation = validation_dir_path is None and \
            (val_size or self.validation_size) > 0
        
        dataset = ImageFolder(root=data_dir_path, transform=self.transform)
        self.class_to_idx = dataset.class_to_idx
        self.num_classes = len(dataset.classes)
        self.class_names = dataset.classes

        if validation_dir_path:
            validation_dataset = ImageFolder(
                root=validation_dir_path, transform=self.transform
            )
        elif create_validation and val_size > 0:
            dataset, validation_dataset = \
                DataLoaderFactory.stratified_split_dataset(
                    dataset, val_size=val_size
                )
        else:
            validation_dataset = None

        data_loader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers) if validation_dataset else None

        return data_loader, val_loader

    def create_test_data_loader(
        self,
        data_dir_path: str,
    ) -> Tuple[DataLoader, List[str]]:
        """
        Creates a DataLoader for test data, optionally returning image names.

        This method is tailored for test datasets where shuffling is not required
        and image names might be needed for creating submission files or for reference.

        Args:
            data_dir_path: The directory path of the test data.

        Returns:
            A tuple containing the DataLoader for the test dataset and a list of image
            names.
        """
        dataset = ImageFolder(root=data_dir_path, transform=self.transform)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        image_names = [Path(i[0]).name for i in dataset.imgs]
        return data_loader, image_names

    @staticmethod
    def stratified_split_dataset(
            dataset: ImageFolder, val_size: float, random_state: int = 42
        ) -> Tuple[Subset, Subset]:
        """
        Splits the dataset into stratified training and validation subsets.

        'Stratified' means that the subsets will have the same proportion of examples
        from each class as the original dataset. This is particularly useful for
        maintaining a balanced representation of classes in both training and
        validation sets.

        Args:
            dataset: The dataset to split.
            val_size: The proportion of the dataset to allocate to the validation
                      subset.
            random_state: An optional integer seed for reproducible shuffling.
                          Default is 42.

        Returns:
            A tuple containing Subset instances for training and validation datasets.
        """
        targets = np.array(dataset.targets)
        classes, _ = np.unique(targets, return_counts=True)
        class_indices = [np.where(targets == i)[0] for i in classes]

        np.random.seed(random_state)  # Ensure reproducibility if random_state is provided

        train_indices, val_indices = [], []

        for indices in class_indices:
            np.random.shuffle(indices)
            split = int(len(indices) * val_size)
            val_indices.extend(indices[:split])
            train_indices.extend(indices[split:])

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        return train_subset, val_subset

    def save(self, file_path: str) -> None:
        path = Path(file_path)
        directory_path = path.parent
        os.makedirs(directory_path, exist_ok=True)
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> None:
        return joblib.load(file_path)


def get_data_loader_factory(preprocessing_config: dict) -> DataLoaderFactory:
    """
    Creates an instance of DataLoaderFactory configured with the given preprocessing
    settings.

    This function initializes a DataLoaderFactory with configurations specified in the
    `preprocessing_config` dictionary. The factory can then be used to create
    DataLoader objects for training and validation datasets with applied
    transformations.

    Args:
        preprocessing_config: Dictionary containing the preprocessing configuration.

    Returns:
        An instance of DataLoaderFactory configured as per the provided settings.
    """
    data_loader_factory = DataLoaderFactory(**preprocessing_config)
    return data_loader_factory



def create_training_and_validation_dataloaders(
    loader_factory: DataLoaderFactory,
    train_dir_path: str,
    valid_dir_path: Optional[str] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Creates DataLoader objects for training and, optionally, validation datasets.

    This function utilizes the provided DataLoaderFactory instance to prepare
    DataLoader objects for the training dataset and optionally for the validation
    dataset, depending on the presence of a validation directory path. If a separate
    validation directory is not provided, and if the DataLoaderFactory is configured
    to create a validation split, it will automatically generate a validation
    DataLoader from the training dataset.


    Args:
        loader_factory: An instance of DataLoaderFactory configured for dataset preprocessing.
        train_dir_path: The file path to the training dataset.
        valid_dir_path: Optional; the file path to the validation dataset. If None, and if
                        loader_factory is configured to split the training dataset, a validation
                        split will be created.

    Returns:
        A tuple containing the DataLoader for the training dataset and, if applicable, the DataLoader
        for the validation dataset. The validation DataLoader is None if no validation dataset is
        provided or created from the training dataset.
    """
    train_data_loader, valid_data_loader = loader_factory.create_data_loaders(
        data_dir_path=train_dir_path,
        validation_dir_path=valid_dir_path,
        val_size=loader_factory.validation_size,
        shuffle=True,
    )
    return train_data_loader, valid_data_loader


def save_data_loader_factory(
        data_loader_factory: DataLoaderFactory, file_path: str) -> None:
    """
    Saves the DataLoaderFactory instance to a file.

    This function serializes the DataLoaderFactory instance to the specified file
    path, allowing for its configuration to be persisted and later reloaded.

    Args:
        data_loader_factory: The DataLoaderFactory instance to be saved.
        file_path: The path to the file where the DataLoaderFactory instance will
                   be saved.
    """
    data_loader_factory.save(file_path)


def load_data_loader_factory(file_path: str) -> DataLoaderFactory:
    """
    Loads a DataLoaderFactory instance from a file.

    This function deserializes a DataLoaderFactory instance from the specified file path,
    allowing for the reuse of data loading configurations.

    Args:
        file_path (str): The path to the file from which the DataLoaderFactory instance
                         should be loaded.

    Returns:
        DataLoaderFactory: The deserialized DataLoaderFactory instance.
    """
    try:
        data_loader_factory = joblib.load(file_path)
        return data_loader_factory
    except Exception as exc:
        err_msg = "Failed to load DataLoaderFactory from file."
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc
