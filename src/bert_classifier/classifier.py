import datetime
import os
import pathlib
import sys

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from bert_classifier.constants import OUTPUT_DIR


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Binary classification

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train(
    cls_embeddings,
    data_df,
    device,
    max_epochs,
    optimizer_string="adam",
    learning_rate=.01,
    momentum=.9,
    random_seed=42,
    _patience=3
):
    print("Commencing training...")
    print(f"\tOptimizer: {optimizer_string}")
    print(f"\tPatience: {_patience}")
    print(f"\tLearning Rate: {learning_rate}")
    if optimizer_string == "sdg":
        print(f"\tMomentum: {momentum}")
    print("---" * 30)

    now = datetime.datetime.now()

    # Split data into training and testing sets
    X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(
        cls_embeddings, data_df['Category'], test_size=0.4, random_state=random_seed
    )
    
    X_val_tmp, X_test, y_val_tmp, y_test = train_test_split(
        X_test_tmp, y_test_tmp, test_size=.75, random_state=random_seed
    )

    # Create test validation for early stopping checking.
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_val_tmp, y_val_tmp, test_size=.5, random_state=random_seed
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).to(device)

    model = LogisticRegressionModel(input_dim=X_train.shape[1]).to(device)

    criterion = nn.BCELoss()  # Binary cross-entropy loss

    optimizer = None
    if optimizer_string == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_string == "sdg":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    X_train_val = torch.tensor(X_train_val, dtype=torch.float32).to(device)  # move the validation set to the GPU

    best_accuracy = 0.0
    model_output_dir = pathlib.Path(OUTPUT_DIR, "models")

    try:
        if not model_output_dir.is_dir():
            os.makedirs(model_output_dir)
    except Exception as e:
        sys.exit(f"{e}")

    best_model_path = pathlib.Path(model_output_dir, "best_model.pth")
    patience = _patience
    wait = 0
    last_accuracy = 0.0

    current_epoch = 1
    for epoch in range(max_epochs):  # Training loop
        print(f"\rTraining... Epoch {current_epoch}/{max_epochs}", end="", flush=True)
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Begin early stopping code
        # Checking accuracy
        model.eval()  # Set the model to evaluation mode for testing accuracy
        with torch.no_grad():
            # Compute predictions on validation data
            y_train_val_pred = model(X_train_val).squeeze().round().detach().cpu().numpy()
            accuracy = accuracy_score(y_train_val, y_train_val_pred)

        print(f" - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        # Did accuracy improve?
        if accuracy > best_accuracy:
            wait = 0
            print(f"New best accuracy: {accuracy:.4f}. Saving model checkpoint...")
            best_accuracy = accuracy  # update our best accuracy

            # Save the best model to disk
            torch.save(model.state_dict(), best_model_path)
        else:
            if last_accuracy < accuracy:
                wait = 0
            elif wait < patience:
                wait += 1
            else:
                print(f"Accuracy worsened: {accuracy:.4f}. Stopping training.")
                break  # Early stopping in the case that accuracy worsens

            model.train()
        # End early stopping code

        current_epoch += 1

    print()  # Adding a newline

    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_pred = model(X_val).squeeze().round().detach().cpu().numpy()


    print("Accuracy:", accuracy_score(y_val, y_pred))

    return model

