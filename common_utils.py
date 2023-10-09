### THIS FILE CONTAINS COMMON FUNCTIONS, CLASSSES

from tqdm import tqdm
import os
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def split_dataset(df, columns_to_drop, test_size, random_state):
    # Step 1: Encode the labels in the 'label' column as integers
    label_encoder = preprocessing.LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    # Step 2: Split the dataset into training and testing sets
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # Step 3: Drop the specified columns from the training set and separate the labels
    # Drop filename and label else the model will learn to predict the label from the train data, and overfit
    df_train2 = df_train.drop(columns_to_drop, axis=1)
    y_train2 = df_train["label"].to_numpy()

    # Step 4: Drop the specified columns from the test set and separate the labels
    df_test2 = df_test.drop(columns_to_drop, axis=1)
    y_test2 = df_test["label"].to_numpy()

    return df_train2, y_train2, df_test2, y_test2


def preprocess_dataset(df_train, df_test):
    """
    Used to scale the features in the dataset to zero mean and unit variance, which helps in training deep neural networks.
    The training data's scaling parameters are applied to the test data to ensure consistency.
    """

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled


def set_seed(seed=0):
    """
    This sets the seed for multiple libraries to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # Set PYTHONHASHSEED for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.benchmark = False  # Disable non-deterministic algorithms
    torch.backends.cudnn.deterministic = True


# This is an implementation of early stopping, a technique to stop training the model once it stops improving on a validation set to prevent overfitting.
# The class checks if the validation loss has stopped decreasing (or increased beyond a certain threshold) for a number of epochs specified by patience. If it has, the training process can be stopped.
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# -------------------------------------------------------------------------#
# --------------------------FOR QUESTION A2 -------------------------------#
# CONTAINS THE FOLLOWING:                                                  #
# - network (MLP defined in QA1)                                           #
# - torch datasets (CustomDataset defined in QA1)                          #
# - loss function (loss_fn defined in QA1)                                 #
# - train_one_epoch function                                               #
# - evaluate_model function                                                #
# -------------------------------------------------------------------------#


class MLP(nn.Module):
    # Initialize the model parameters
    def __init__(self, no_features, no_hidden, no_labels):
        super().__init__()  # Call the parent class constructor

        # Define the neural network layers and their sequence
        self.mlp_stack = nn.Sequential(
            # First hidden layer
            nn.Linear(
                no_features, no_hidden
            ),  # Linear layer with 'no_features' inputs and 'no_hidden' outputs
            nn.ReLU(),  # Rectified Linear Unit (ReLU) activation function
            nn.Dropout(
                0.2
            ),  # Dropout layer to prevent overfitting, drops 20% of the neurons
            # Second hidden layer
            nn.Linear(
                no_hidden, 128
            ),  # Another linear layer with 'no_hidden' inputs and 128 outputs
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(0.2),  # Dropout layer with 20% dropout rate
            # Third hidden layer
            nn.Linear(128, 128),  # Third linear layer with 128 inputs and 128 outputs
            nn.ReLU(),  # ReLU activation function
            nn.Dropout(0.2),  # Dropout layer with 20% dropout rate
            # Output layer
            nn.Linear(
                128, no_labels
            ),  # Linear layer to produce final output. 128 inputs and 'no_labels' outputs.
            nn.Sigmoid(),  # Sigmoid activation function for binary classification
        )

    # Define the forward pass
    def forward(self, x):
        # Pass the input through the mlp_stack to get the output
        logits = self.mlp_stack(x)
        return logits


class CustomDataset(Dataset):
    # Constructor: called when an object is created from the class
    def __init__(self, X, y):
        # Convert the input arrays to PyTorch tensors
        self.X = torch.tensor(
            X, dtype=torch.float32
        )  # Converts input data to float32 dtype tensor
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Returns the length of the dataset
    def __len__(self):
        return len(self.y)

    # Allows the dataset to be indexed so that it can work with PyTorch's DataLoader
    # For the given index 'idx', it returns the input data and its corresponding label
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Loss Function
loss_fn = nn.BCELoss()


# Helper function for training the model for one epoch
def train_one_epoch(model, data_loader, optimizer, loss_function):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    for inputs, targets in data_loader:  # Iterate through the dataset
        optimizer.zero_grad()  # Reset the gradients
        outputs = model(inputs)  # Forward pass
        loss = loss_function(outputs, targets)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        total_loss += loss.item() * inputs.size(0)  # Accumulate loss
    return total_loss / len(data_loader.dataset)  # Return average loss


# Helper function for evaluating the model
def evaluate_model(model, data_loader, loss_function):
    model.eval()
    correct_predictions = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            correct_predictions += (predicted == targets).sum().item()
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct_predictions / len(data_loader.dataset)
    return accuracy, avg_loss


# ----------------------------------------------------------------------------------------------------------#
# ------------------------------------------ FOR QUESTION A3 -----------------------------------------------#
# CONTAINS THE FOLLOWING:                                                                                   #
# - generate_cv_folds_for_num_neurons function to generate 5-fold cross-validation folds                    #
# - generate_cv_dataloaders function to generate dataloaders for each fold                                  #
# ----------------------------------------------------------------------------------------------------------#


def generate_cv_folds_for_num_neurons(num_neurons, X_train, y_train):
    set_seed()
    # Dictionary Initializations
    X_train_scaled_dict = {}  # Dictionary to store scaled training data
    X_val_scaled_dict = {}  # Dictionary to store scaled validation data
    y_train_dict = {}  # Dictionary to store training labels
    y_val_dict = {}  # Dictionary to store validation labels

    # Create a 5-fold cross-validation object with shuffling
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    # Loop through different batch sizes
    for num_neuron in num_neurons:
        X_train_scaled_folds = []  # List to store scaled training data for each fold
        X_val_scaled_folds = []  # List to store scaled validation data for each fold
        y_train_folds = []  # List to store training labels for each fold
        y_val_folds = []  # List to store validation labels for each fold

        # Loop through each fold
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

            # Scaling the data using StandardScaler
            scaler = preprocessing.StandardScaler()
            X_train_fold_scaled = scaler.fit_transform(X_train_fold)
            X_val_fold_scaled = scaler.transform(X_val_fold)

            # Append data to respective lists
            X_train_scaled_folds.append(X_train_fold_scaled)
            X_val_scaled_folds.append(X_val_fold_scaled)
            y_train_folds.append(y_train_fold)
            y_val_folds.append(y_val_fold)

        # Store data for this batch size in dictionaries
        X_train_scaled_dict[num_neuron] = X_train_scaled_folds
        X_val_scaled_dict[num_neuron] = X_val_scaled_folds
        y_train_dict[num_neuron] = y_train_folds
        y_val_dict[num_neuron] = y_val_folds

    # Return the dictionaries containing data for different batch sizes
    return X_train_scaled_dict, X_val_scaled_dict, y_train_dict, y_val_dict


def generate_cv_dataloaders(X_train_scaled, y_train2, X_val_scaled, y_val2, batch_size):
    set_seed()

    # Create a PyTorch Dataset for the training data for the current fold. The CustomDataset will convert our data into a format that PyTorch can work with.
    train_dataset = CustomDataset(X_train_scaled, y_train2)

    # Similarly, create a PyTorch Dataset for the validation data for the current fold.
    val_dataset = CustomDataset(X_val_scaled, y_val2)

    # Create a DataLoader for the training dataset. This DataLoader will allow us to efficiently iterate over data in mini-batches of size 'batch_size'.
    # We shuffle the training data to ensure the order is different each epoch.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create a DataLoader for the validation dataset. There's no need to shuffle validation data as we typically just evaluate performance on it.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ----------------------------------------------------------------------------------------------------------#
# ------------------------------------------ FOR QUESTION A4 -----------------------------------------------#
# CONTAINS THE FOLLOWING:                                                                                   #
# - preprocess_A4 for preprocessing                                                                         #
# - get_scaler_A4 for obtaining the scaler                                                                  #
# - get_trained_model_A4 for obtaining the model trained with hyperparameters obtained in A3                #
# ----------------------------------------------------------------------------------------------------------#


def preprocess_A4(df):
    # Dropping Unnecessary Columns and Splitting the Dataset:
    # The dataset column called filename is likely a unique identifier for each sample and not a feature we would use for modeling. Hence, we should remove it before feeding the data to our model.
    columns_to_drop = ["filename", "label"]
    df["label"] = df["filename"].str.split("_").str[-2]
    # Splitting the dataset into training and testing sets:
    # The split is done in a 70:30 ratio, and the random_state ensures reproducibility.
    df_train, y_train, df_test, y_test = split_dataset(
        df, columns_to_drop, test_size=0.3, random_state=0
    )

    return df_train, y_train, df_test, y_test


def get_scaler_A4(df):
    scaler = preprocessing.StandardScaler()
    scaler.fit(df)
    return scaler


def get_trained_model_A4(model_path, input_size):
    # Ensure the model architecture is defined (in this case, MLP)
    # Create an instance of your model
    no_input_features = 77
    no_hidden = 256
    no_labels = 1
    model = MLP(no_input_features, no_hidden, no_labels)

    # Load the saved state dictionary into this model instance
    model.load_state_dict(torch.load("a3_model.pth"))

    # Set the model to evaluation mode
    model.eval()

    return model

