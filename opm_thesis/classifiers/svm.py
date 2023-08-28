import mne
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_save = r"C:\Users\user\Desktop\MasterThesis\opm-thesis\data\data_nottingham_preprocessed\epochs"
frequency_bands = [
    "alpha",
    "beta",
    "low_gamma",
    "low_mid_gamma",
    "mid_gamma",
    "high_gamma",
    "all_gamma",
]
decimate = True

for frequency_band in frequency_bands:
    with open(data_save + "\\hilbert_" + frequency_band + "_all_epochs.pkl", "rb") as f:
        epochs = pickle.load(f)

    picks = mne.pick_types(epochs.info, meg=True, exclude="bads")

    # Extract the epoch data for the selected channels
    epoch_data = epochs.get_data()[:, picks]
    if decimate:
        epoch_data = epoch_data[:, :, ::10]

    x = epoch_data.reshape(epoch_data.shape[0], -1)
    y = (np.log2(epochs.events[:, 2]) - 2).astype(int)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=50
    )

    # Instantiate the SVM classifier
    svm_classifier = svm.SVC()

    # Train the SVM classifier
    svm_classifier.fit(x_train, y_train)

    # Predict labels on the test set
    predicted_labels = svm_classifier.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_labels)
    print(
        frequency_band
        + " "
        + str(int(decimate))
        + f" SVM Test Accuracy: {accuracy:.4f}"
    )
