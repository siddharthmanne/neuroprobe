"""
Demo script: Logistic Regression Baseline for Scene Object Identification
===========================================================================

This script demonstrates a simple logistic regression baseline for identifying
scene objects (number of faces) from brain recordings.

Task: Predict the number of faces in the scene (0, 1, or 2+ faces)
Data: Single subject, single trial for quick demo
Model: Logistic Regression (scikit-learn)

SETUP INSTRUCTIONS:
-------------------
1. Extract the dataset (if not already done):
   python braintreebank_download_extract.py

2. Set the ROOT_DIR_BRAINTREEBANK environment variable:
   export ROOT_DIR_BRAINTREEBANK=/path/to/neuroprobe/braintreebank

   OR run this script with:
   ROOT_DIR_BRAINTREEBANK=/path/to/neuroprobe/braintreebank python demo_scene_object_baseline.py
"""

import numpy as np
import torch
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Set default ROOT_DIR_BRAINTREEBANK if not already set
if 'ROOT_DIR_BRAINTREEBANK' not in os.environ:
    # Try to use the local braintreebank directory
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'braintreebank')
    if os.path.exists(default_path):
        os.environ['ROOT_DIR_BRAINTREEBANK'] = default_path
        print(f"Using default braintreebank directory: {default_path}")
    else:
        print("ERROR: ROOT_DIR_BRAINTREEBANK environment variable not set and ./braintreebank directory not found.")
        print("\nPlease either:")
        print("1. Extract the dataset: python braintreebank_download_extract.py")
        print("2. Set the environment variable: export ROOT_DIR_BRAINTREEBANK=/path/to/braintreebank")
        print("3. Run with: ROOT_DIR_BRAINTREEBANK=/path/to/braintreebank python demo_scene_object_baseline.py")
        sys.exit(1)

from neuroprobe import BrainTreebankSubject, BrainTreebankSubjectTrialBenchmarkDataset
import neuroprobe.train_test_splits as neuroprobe_train_test_splits
import neuroprobe.config as neuroprobe_config


def main():
    print("="*70)
    print("Scene Object Identification Demo - Logistic Regression Baseline")
    print("="*70)
    print()

    # Configuration
    subject_id = 1
    trial_id = 1
    eval_name = 'face_num'  # Scene object task: number of faces (0, 1, 2+)
    seed = 42

    print(f"Configuration:")
    print(f"  - Subject ID: {subject_id}")
    print(f"  - Trial ID: {trial_id}")
    print(f"  - Task: {eval_name} ({neuroprobe_config.NEUROPROBE_TASKS_MAPPING[eval_name]})")
    print(f"  - Model: Logistic Regression")
    print(f"  - Time window: 0-1 second after word onset")
    print(f"  - Random seed: {seed}")
    print()

    # Load subject data
    print("Loading subject data...")
    subject = BrainTreebankSubject(
        subject_id=subject_id,
        cache=True,
        dtype=torch.float32
    )
    print(f"  - Loaded subject {subject_id}")
    print()

    # Generate train/test splits (Within-Session cross-validation)
    print("Generating train/test splits (within-session k-fold)...")
    start_neural_data_before_word_onset = 0
    end_neural_data_after_word_onset = neuroprobe_config.SAMPLING_RATE * 1  # 1 second

    folds = neuroprobe_train_test_splits.generate_splits_within_session(
        subject=subject,
        trial_id=trial_id,
        eval_name=eval_name,
        dtype=torch.float32,
        output_indices=False,
        start_neural_data_before_word_onset=start_neural_data_before_word_onset,
        end_neural_data_after_word_onset=end_neural_data_after_word_onset,
        lite=True  # Use Neuroprobe-Lite (subset of electrodes, max 3500 samples)
    )

    print(f"  - Number of folds: {len(folds)}")

    # Get dataset info from first fold
    first_dataset = folds[0]["train_dataset"].dataset
    print(f"  - Number of electrodes: {len(first_dataset.electrode_labels)}")
    print(f"  - Total samples in dataset: {len(first_dataset)}")
    print()

    # Evaluate on each fold
    all_train_accs = []
    all_test_accs = []

    for fold_idx, fold in enumerate(folds):
        print(f"Fold {fold_idx + 1}/{len(folds)}")
        print("-" * 70)

        train_dataset = fold["train_dataset"]
        test_dataset = fold["test_dataset"]

        print(f"  - Train samples: {len(train_dataset)}")
        print(f"  - Test samples: {len(test_dataset)}")

        # Convert PyTorch dataset to numpy arrays for scikit-learn
        X_train = np.array([item[0].flatten().numpy() for item in train_dataset])
        y_train = np.array([item[1] for item in train_dataset])
        X_test = np.array([item[0].flatten().numpy() for item in test_dataset])
        y_test = np.array([item[1] for item in test_dataset])

        print(f"  - Feature dimension: {X_train.shape[1]}")
        print(f"  - Classes in train: {sorted(np.unique(y_train).tolist())}")
        print(f"  - Classes in test: {sorted(np.unique(y_test).tolist())}")

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train logistic regression
        print(f"  - Training logistic regression...")
        clf = LogisticRegression(
            random_state=seed,
            max_iter=10000,
            tol=1e-3
        )
        clf.fit(X_train_scaled, y_train)

        # Get predictions
        train_preds = clf.predict(X_train_scaled)
        test_preds = clf.predict(X_test_scaled)

        # Calculate accuracy
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        print(f"  - Train Accuracy: {train_acc:.4f}")
        print(f"  - Test Accuracy:  {test_acc:.4f}")

        # Show classification report for first fold only (to avoid clutter)
        if fold_idx == 0:
            print()
            print("  Classification Report (Test Set):")
            report_lines = classification_report(y_test, test_preds, zero_division=0).split('\n')
            for line in report_lines:
                if line.strip():
                    print(f"    {line}")

        all_train_accs.append(train_acc)
        all_test_accs.append(test_acc)

        print()

    # Summary statistics
    print("="*70)
    print("Summary Across All Folds")
    print("="*70)
    print(f"Train Accuracy: {np.mean(all_train_accs):.4f} ± {np.std(all_train_accs):.4f}")
    print(f"Test Accuracy:  {np.mean(all_test_accs):.4f} ± {np.std(all_test_accs):.4f}")
    print()
    print("Demo completed successfully!")
    print()


if __name__ == "__main__":
    main()
