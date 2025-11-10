import os
os.environ['ROOT_DIR_BRAINTREEBANK'] = '/Users/Siddharth/neuroprobe/braintreebank/'

import torch
import neuroprobe.config as neuroprobe_config

from neuroprobe import BrainTreebankSubject
from neuroprobe import BrainTreebankSubjectTrialBenchmarkDataset

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

################### 1) LOADING IN A SUBJECT AS BRAINTREEBANKSUBJECT ######################
subject_id=1
coordinates_type = "cortical"
subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=False, dtype=torch.float32, coordinates_type=coordinates_type)

print("Loaded subject", subject_id)
print("Electrode labels (first 10): ", subject.electrode_labels[:10])

# subject.set_electrode_subset(['', '', '']) 

# LOADING ELECTRODE DATA FROM SPECIFIC TRIAL (a movie the subject watched)
trial_id = 1
subject.load_neural_data(trial_id)
window_from = None # This is the index into the neural data array from where to start loading the data.
window_to = None # if None, the whole trial will be loaded

#  To get the data for a specific electrode, use subject.get_electrode_data(trial_id, electrode_label)
all_neural_data = subject.get_all_electrode_data(trial_id, window_from=window_from, window_to=window_to)

print("All neural data shape:")
print(all_neural_data.shape) # (n_electrodes, n_samples), could be something like (129 electrodes x 21,401,009 timesteps)



############# 2) CREATE DATASET ##########################

# CHOOSE EVALUATION METRIC
eval_name = "speech"

# if True, the dataset will output the indices of the samples in the neural data in a tuple: (index_from, index_to); 
# if False, the dataset will output the neural data directly
output_indices = False

start_neural_data_before_word_onset = 0 # the number of samples to start the neural data before each word onset
end_neural_data_after_word_onset = neuroprobe_config.SAMPLING_RATE * 1 # cuts brain data into 1 second windows around each word. the number of samples to end the neural data after each word onset -- here we use 1 second

#CREATE A DATASET WITH SUBJECTS
dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=torch.float32, eval_name=eval_name, output_indices=output_indices,
                                                    start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                    lite=True)
data_electrode_labels = dataset.electrode_labels# NOTE: this is different from the subject.electrode_labels! Neuroprobe uses a special subset of electrodes in this exact order.
data_electrode_coordinates = dataset.electrode_coordinates

print("Items in the dataset:", len(dataset), "\n")
print(f"The first item: (shape = {dataset[0][0].shape})", dataset[0][0], f"label = {dataset[0][1]}", sep="\n")
print("")
print(f"Electrode labels in the data above in the following order ({len(data_electrode_labels)} electrodes):", data_electrode_labels)
print(f"Electrode coordinates in the data above in the following order ({len(data_electrode_coordinates)} electrodes):", data_electrode_coordinates)


############# 3) TRAIN/TEST SPLITS ##########################
import neuroprobe.train_test_splits as neuroprobe_train_test_splits

# Makes 2 folds for cross validation. 
# Fold 1: train on half, test the other. 
# Fold 2: reversed
folds = neuroprobe_train_test_splits.generate_splits_within_session(subject, trial_id, eval_name, dtype=torch.float32, 
                                                                                # Put the dataset parameters here
                                                                                output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                                                                                lite=True)
print("len(folds) = k_folds =", len(folds))



############# 4) TRAIN MODEL ##########################

for fold_idx, fold in enumerate(folds):
    print(f"Fold {fold_idx+1} of {len(folds)}")
    train_dataset = fold['train_dataset']
    test_dataset = fold['test_dataset']

    # Convert PyTorch dataset to numpy arrays for scikit-learn
    X_train = np.array([item[0].flatten() for item in train_dataset]) # Flattening (num_electrodes, timesteps) into 1 array
    y_train = np.array([item[1] for item in train_dataset])
    X_test = np.array([item[0].flatten() for item in test_dataset])
    y_test = np.array([item[1] for item in test_dataset])

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(random_state=42, max_iter=1000, tol=1e-3)
    clf.fit(X_train, y_train)

    # Evaluate model
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"\t Train accuracy: {train_score:.3f} | Test accuracy: {test_score:.3f}")
