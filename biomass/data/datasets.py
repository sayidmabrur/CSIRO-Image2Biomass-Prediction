"""
Dataset classes for the CSIRO Biomass prediction task.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from .config import DataConfig


class Image2BioMassTrainValDataset(Dataset):
    """
    Robust dataset class for training and validation with proper indexing and transform handling.

    This dataset loads plant images and associated features to predict biomass (Dry_Green_g,
    Dry_Dead_g, Dry_Clover_g).

    Features:
    - Categorical: Sampling_Date, State, Species (label encoded)
    - Numeric: Pre_GSHH_NDVI, Height_Ave_cm

    Targets: Dry_Green_g, Dry_Dead_g, Dry_Clover_g
    """

    def __init__(
        self,
        dataset_path,
        indices=None,
        img_transform=None,
        numeric_transform=None,
        target_transform=None,
        df=None,
        label_encoders=None,
        numeric_stats=None,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path: Path to the dataset directory containing train.csv and images
            indices: Optional subset of indices for train/val split
            img_transform: Transform to apply to images
            numeric_transform: Function to normalize numeric features
            target_transform: Transform to apply to targets (e.g., log1p)
            df: Pre-processed DataFrame (for validation set to share encoders)
            label_encoders: Pre-fitted label encoders (for validation set)
            numeric_stats: Pre-computed numeric statistics (for validation set)
        """
        self.dataset_path = dataset_path
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.numeric_transform = numeric_transform

        # If shared df and encoders provided, use them (for val set)
        if df is not None:
            self.df = df
            self.le_date = label_encoders["date"]
            self.le_state = label_encoders["state"]
            self.le_species = label_encoders["species"]
            self.numeric_stats = numeric_stats
        else:
            (
                self.df,
                self.le_date,
                self.le_state,
                self.le_species,
                self.numeric_stats,
            ) = self._process_df()

        # Use subset of indices if provided
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        self.targets = self.df[["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]].values

    def _process_df(self):
        """
        Process the training CSV file.

        Returns:
            Tuple of (processed_df, le_date, le_state, le_species, numeric_stats)
        """
        le_date = LabelEncoder()
        le_state = LabelEncoder()
        le_species = LabelEncoder()

        df = pd.read_csv(os.path.join(self.dataset_path, "train.csv"))
        df["base_sample_id"] = df["sample_id"].str.split("__").str[0]
        df = df.pivot_table(
            index=[
                "base_sample_id",
                "image_path",
                "Sampling_Date",
                "State",
                "Species",
                "Pre_GSHH_NDVI",
                "Height_Ave_cm",
            ],
            columns="target_name",
            values="target",
        ).reset_index()

        df["Sampling_Date"] = le_date.fit_transform(df["Sampling_Date"])
        df["State"] = le_state.fit_transform(df["State"])
        df["Species"] = le_species.fit_transform(df["Species"])

        # Store numeric stats for normalization
        numeric_stats = {
            "Pre_GSHH_NDVI": {
                "min": df["Pre_GSHH_NDVI"].min(),
                "max": df["Pre_GSHH_NDVI"].max(),
            },
            "Height_Ave_cm": {
                "min": df["Height_Ave_cm"].min(),
                "max": df["Height_Ave_cm"].max(),
            },
        }

        return df, le_date, le_state, le_species, numeric_stats

    def __len__(self):
        return len(self.df)

    def get_label_encoders(self):
        """Get the label encoders for categorical features."""
        return {
            "date": self.le_date,
            "state": self.le_state,
            "species": self.le_species,
        }

    def get_cat_vocab_sizes(self):
        """Get vocabulary sizes for categorical features (for embedding layers)."""
        return [
            len(self.le_date.classes_),
            len(self.le_state.classes_),
            len(self.le_species.classes_),
        ]

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            Tuple of (image, combined_features, targets)
            - image: Transformed image tensor
            - combined_features: Concatenated categorical and numeric features
            - targets: Target values (biomass measurements)
        """
        row = self.df.iloc[idx]

        # Load image
        img_path = os.path.join(self.dataset_path, row["image_path"])
        image = decode_image(img_path)

        # Numeric features
        ndvi = row["Pre_GSHH_NDVI"]
        height = row["Height_Ave_cm"]

        # Normalize numeric features
        if self.numeric_transform:
            height = self.numeric_transform(
                height,
                self.numeric_stats["Height_Ave_cm"]["max"],
                self.numeric_stats["Height_Ave_cm"]["min"],
            )

        numeric_features = torch.tensor([ndvi, height], dtype=torch.float32)

        # Categorical features
        categorical_features = torch.tensor(
            [
                row["Sampling_Date"],
                row["State"],
                row["Species"],
            ],
            dtype=torch.long,
        )

        # Apply image transform
        if self.img_transform:
            image = self.img_transform(image)

        # Combine features
        combined_features = torch.cat(
            [categorical_features.float(), numeric_features], dim=0
        )

        # Targets
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        if self.target_transform:
            targets = self.target_transform(targets)

        return image, combined_features, targets


class Image2BioMassTestDataset(Dataset):
    """
    Dataset class for test data (without targets).

    Returns sample_id instead of targets for prediction submission.
    """

    def __init__(
        self,
        dataset_path,
        img_transform=None,
        numeric_transform=None,
        categorical_transform=None,
    ):
        """
        Initialize the test dataset.

        Args:
            dataset_path: Path to the dataset directory containing test.csv and images
            img_transform: Transform to apply to images
            numeric_transform: Function to normalize numeric features (unused for test)
            categorical_transform: Transform for categorical features (unused for test)
        """
        self.df = self.process_df(dataset_path)
        self.dataset_path = dataset_path
        self.img_transform = img_transform
        self.numeric_transform = numeric_transform
        self.categorical_transform = categorical_transform

    def process_df(self, dataset_path):
        """
        Process the test CSV file.

        Returns:
            Processed DataFrame
        """
        self.le_date = LabelEncoder()
        self.le_state = LabelEncoder()
        self.le_species = LabelEncoder()

        df = pd.read_csv(os.path.join(dataset_path, "test.csv"))
        df["base_sample_id"] = df["sample_id"].str.split("__").str[0]
        df = (
            df.assign(_val="")
            .pivot(
                index=["base_sample_id", "image_path"],
                columns="target_name",
                values="_val",
            )
            .reset_index()
        )

        return df

    def __len__(self):
        return len(self.df)

    def get_cat_features(self):
        """Get list of categorical feature names."""
        return ["Sampling_Date", "State", "Species"]

    def get_cat_vocab_sizes(self):
        """Get vocabulary sizes for categorical features."""
        results = []
        for i in self.get_cat_features():
            results.append(len(self.df[i].unique()))
        return results

    def __getitem__(self, idx):
        """
        Get a single test sample.

        Returns:
            Tuple of (image, combined_features, sample_id)
            - image: Transformed image tensor
            - combined_features: Zero tensor placeholder (test data has no features)
            - sample_id: Sample identifier for submission
        """
        img_path = os.path.join(self.dataset_path, self.df.loc[idx, "image_path"])
        image = decode_image(img_path)

        # Use val_transform for test data (no augmentation)
        if self.img_transform:
            image = self.img_transform(image)
        else:
            # Fallback basic transform if no transform provided
            config = DataConfig()
            transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.ToDtype(config.dtype, scale=True),
                    v2.Resize((518, 518)),
                    v2.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
                ]
            )
            image = transform(image)

        # Placeholder features for test data
        combined_features = torch.zeros(5, dtype=torch.float32)
        sample_id = self.df.loc[idx, "base_sample_id"]
        return image, combined_features, sample_id


def create_kfold_datasets(
    dataset_path,
    n_folds=5,
    groups_col_name="Species",
):
    """
    Create K-fold splits using GroupKFold to prevent data leakage.

    Args:
        dataset_path: Path to dataset directory
        n_folds: Number of folds for cross-validation
        groups_col_name: Column name to use for grouping (prevents same group in train and val)

    Returns:
        Tuple of (fold_splits, base_dataset)
    """
    # Create base dataset to get full DataFrame and encoders
    base_dataset = Image2BioMassTrainValDataset(
        dataset_path=dataset_path,
        img_transform=None,
        numeric_transform=None,
        target_transform=None,
    )

    # Get groups for GroupKFold
    groups = base_dataset.df[groups_col_name].values

    # Setup K-Fold Cross Validation with groups
    # GroupKFold doesn't have shuffle or random_state - it splits by groups
    kfold = GroupKFold(n_splits=n_folds)
    fold_splits = list(kfold.split(range(len(base_dataset)), groups=groups))
    return fold_splits, base_dataset


def create_dataloaders(
    train_dataset, val_dataset, batch_size=16, num_workers=4, pin_memory=True
):
    """
    Create DataLoader instances for training and validation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for both dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
