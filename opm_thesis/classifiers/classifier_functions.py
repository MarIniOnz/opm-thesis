import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from opm_thesis.classifiers.classifier import (
    DeepConvNet,
    MyDataset,
    LF_CNN,
)


def calculate_accuracy(
    freq, labels_to_use, classifier_chosen, num_epochs=100, num_splits=5
):
    # Define the path to the file
    DATA_DIR = "./data/digits_epochs/freq_bands/"
    FILENAME = DATA_DIR + freq + "_all_epochs_decimated.pkl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=num_splits, shuffle=True, random_state=0)

    with open(FILENAME, "rb") as file:
        epochs = pickle.load(file)

    epochs = epochs.pick("mag", exclude="bads")

    valid_epochs_mask = np.isin(epochs.events[:, -1], labels_to_use)
    data = np.real(epochs.get_data())[valid_epochs_mask]
    data -= np.mean(data, axis=(0, 2), keepdims=True)
    data /= np.std(data, axis=(0, 2), keepdims=True)

    label_mapping = {label: idx for idx, label in enumerate(labels_to_use)}
    labels = np.array(
        [
            label_mapping[label]
            for label in epochs.events[:, -1]
            if label in label_mapping
        ]
    )

    # Store accuracies for each fold
    accuracies = []

    for train_index, test_index in kf.split(data):
        # Split data into training and test sets for this fold
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # Create PyTorch datasets and loaders
        train_dataset = MyDataset(train_data, train_labels)
        test_dataset = MyDataset(test_data, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Initialize and train the model
        num_channels = train_data.shape[1]
        num_samples = train_data.shape[2]
        classifier = classifier_chosen(
            num_channels, num_samples, len(labels_to_use), device
        ).to(device)
        classifier.train_model(
            train_loader,
            test_loader,
            num_epochs=num_epochs,
            learning_rate=1e-3,
        )

        # Evaluate the model
        accuracy = classifier.evaluate(test_loader) * 100
        accuracies.append(accuracy)
        print(f"Accuracy for fold: {accuracy:.2f}")

    # Calculate and print the mean accuracy
    mean_accuracy = np.mean(accuracies)
    # Print the mean accuracy rounded to 2 decimal places
    print(f"Mean accuracy for freq {freq}: {mean_accuracy:.2f}")
    return accuracies


ids = [2**i for i in range(3, 8)]

accuracies = calculate_accuracy("all_data", ids, LF_CNN, 200)

with open("./data/classifier_accuracies/all_data_lf_cnn_all_fingers.pkl", "wb") as file:
    pickle.dump(accuracies, file)

# accuracies = calculate_accuracy("beta", ids, LF_CNN, 300)

# with open("./data/classifier_accuracies/beta_lf_cnn_all_fingers.pkl", "wb") as file:
#     pickle.dump(accuracies, file)


# id_pairs = [[2**i, 2**j] for i in range(3, 8) for j in range(i + 1, 8)]
# accuracies = []

# for id_pair in id_pairs:
#     print(id_pair)
#     accuracies.append(calculate_accuracy("beta", id_pair, LF_CNN, 2000))

# with open("./data/classifier_accuracies/beta_lfcnn_maxpool2.pkl", "wb") as file:
#     pickle.dump(accuracies, file)

# accuracies = []

# for id_pair in id_pairs:
#     print(id_pair)
#     accuracies.append(calculate_accuracy("all_data", id_pair, LF_CNN, 2000))

# with open("./data/classifier_accuracies/all_data_lfcnn_maxpool2.pkl", "wb") as file:
#     pickle.dump(accuracies, file)
