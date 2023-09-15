""" This script is used to train and evaluate the classifier. """
import pickle
import mne
import numpy as np
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from opm_thesis.classifiers.classifier import Classifier, MyDataset

DATA_DIR = (
    r"/Users/martin.iniguez/Desktop/master_thesis/"
    r"opm-thesis/data/data_nottingham_preprocessed"
)
FREQUENCY_BAND = "low_gamma"
DECIMATE = True

with open(DATA_DIR + "/hilbert_" + FREQUENCY_BAND + "_all_epochs.pkl", "rb") as f:
    epochs = pickle.load(f)

picks = mne.pick_types(epochs.info, meg=True, exclude="bads")

# Extract the epoch data for the selected channels
x = epochs.get_data()[:, picks]
y = (np.log2(epochs.events[:, 2]) - 2).astype(int)

if DECIMATE:
    x = x[:, :, ::10]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=50
)

dataset_train = MyDataset(x_train, y_train)
dataset_test = MyDataset(x_test, y_test)

# Define batch size for training
BATCH_SIZE = 16  # You can adjust this based on your available memory

# Create a DataLoader for your dataset
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

classifier = Classifier()

# Train the classifier using your training data
# train_loader should be a DataLoader containing your training data
classifier.train(train_loader, num_epochs=10, learning_rate=0.001)

classifier.evaluate(test_loader)
