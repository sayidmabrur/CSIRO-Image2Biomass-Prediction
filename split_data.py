import pandas as pd
import os
import shutil
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

# Config
DATASET_PATH = "datasets"
TRAIN_CSV = os.path.join(DATASET_PATH, "train.csv")
TEST_CSV = os.path.join(DATASET_PATH, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "train")
TEST_IMG_DIR = os.path.join(DATASET_PATH, "test")


def main():
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found!")
        return

    print(f"Loading {TRAIN_CSV}...")
    df = pd.read_csv(TRAIN_CSV)
    print(f"Initial Shape: {df.shape}")

    # Backup
    backup_path = TRAIN_CSV + ".bak"
    if not os.path.exists(backup_path):
        print(f"Backing up to {backup_path}")
        shutil.copy2(TRAIN_CSV, backup_path)

    # Extract unique image entries
    unique_df = df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    print(f"Unique Images: {len(unique_df)}")

    # Prepare Splitter
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

    # Groups = Date, Stratify = State
    groups = unique_df["Sampling_Date"]
    y_stratify = unique_df["State"]

    # Get one fold
    train_idx, test_idx = next(sgkf.split(unique_df, y_stratify, groups=groups))

    test_samples_df = unique_df.iloc[test_idx]

    # Verify non-overlapping dates
    train_samples_df = unique_df.iloc[train_idx]
    train_dates = set(train_samples_df["Sampling_Date"])
    test_dates = set(test_samples_df["Sampling_Date"])
    overlap = train_dates.intersection(test_dates)

    if overlap:
        print(f"❌ ERROR: Temporal leakage detected! Overlapping dates: {overlap}")
        return
    else:
        print("✅ No temporal leakage detected.")

    print(f"Test Images: {len(test_samples_df)}")

    # Prepare directory
    if not os.path.exists(TEST_IMG_DIR):
        os.makedirs(TEST_IMG_DIR)
        print(f"Created {TEST_IMG_DIR}")

    # Identify all rows belonging to test images
    test_image_paths = set(test_samples_df["image_path"])

    full_test_df = df[df["image_path"].isin(test_image_paths)].copy()
    full_train_df = df[~df["image_path"].isin(test_image_paths)].copy()

    print(f"Full DataFrame Split:")
    print(f"  Train Rows: {len(full_train_df)}")
    print(f"  Test Rows:  {len(full_test_df)}")

    # Move Files
    print("Moving files...")
    moved_count = 0
    images_to_move = test_samples_df["image_path"].tolist()

    for img_rel_path in images_to_move:
        # img_rel_path is like 'train/img123.jpg'
        src_path = os.path.join(DATASET_PATH, img_rel_path)

        # New relative path will be 'test/filename.jpg'
        filename = os.path.basename(img_rel_path)
        new_rel_path = os.path.join("test", filename)
        dest_path = os.path.join(DATASET_PATH, new_rel_path)

        try:
            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)
                # Update image_path in the dataframe
                # We update it in full_test_df
                # Note: We need to update ALL rows with this image path
                full_test_df.loc[
                    full_test_df["image_path"] == img_rel_path, "image_path"
                ] = new_rel_path.replace("\\", "/")
                moved_count += 1
            else:
                # Check if already moved (idempotency)
                if os.path.exists(dest_path):
                    full_test_df.loc[
                        full_test_df["image_path"] == img_rel_path, "image_path"
                    ] = new_rel_path.replace("\\", "/")
                    moved_count += 1
                else:
                    print(f"Warning: Source file not found: {src_path}")
        except Exception as e:
            print(f"Error moving {src_path}: {e}")

    print(f"Moved {moved_count} images.")

    # SAVE CSVs
    print(f"Saving {TRAIN_CSV}...")
    full_train_df.to_csv(TRAIN_CSV, index=False)

    print(f"Saving {TEST_CSV}...")
    full_test_df.to_csv(TEST_CSV, index=False)

    print("✅ Split Complete!")


if __name__ == "__main__":
    main()
