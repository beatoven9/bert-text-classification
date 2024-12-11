import datetime
import math
import os
import pathlib
import pickle
import uuid

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import torch
from transformers import BertTokenizer, BertModel

from bert_classifier.constants import OUTPUT_DIR


def tokenize_messages(texts, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )


def extract_features(tokenized_inputs, device, batch_size=8):
    print("---" * 30)
    print("Extracting features. This can take a while...")
    now = datetime.datetime.now()

    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(device)  # Move the model over to the GPU if available.
    model.eval()  # Put the model in evaluation mode

    all_embeddings = []
    total_samples = tokenized_inputs['input_ids'].shape[0]
    total_batches = math.ceil(total_samples / batch_size)

    current_batch = 1

    # Extract features using the model
    with torch.no_grad():
        for start_idx in range(0, total_samples, batch_size):
            print(f"\rOperating over batch {current_batch}/{total_batches}", end="", flush=True)
            end_idx = min(start_idx + batch_size, total_samples)  # End of the current batch
            batch = {
                key: val[start_idx:end_idx].to(device)  # Slice and move to device
                for key, val in tokenized_inputs.items()
            }
            # Forward pass for the batch
            outputs = model(**batch)

            # Extract CLS embeddings (first token in last_hidden_state)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move to CPU
            all_embeddings.append(cls_embeddings)  # Store batch embeddings
            current_batch += 1

    print()  # This is to get a newline after the batch update line.
    # Concatenate embeddings.
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    just_now = datetime.datetime.now()
    time_delta = just_now - now
    print("Feature extraction elapsed time: ", time_delta)
    print("---" * 30)

    return all_embeddings


def get_misclassified_examples(data_df, y_val, y_pred):
    val_data = pd.DataFrame({
        'text': data_df.iloc[y_val.index]['Message'],
        'true_label': y_val,
        'predicted_label': y_pred
    })

    false_predictions = val_data[val_data['true_label'] != val_data['predicted_label']]

    return false_predictions


def get_embeddings(data_df, device, feature_embeddings_path=None):
    if feature_embeddings_path is not None:
        try:
            with open(feature_embeddings_path, "rb") as pickle_file:
                cls_embeddings = pickle.load(pickle_file)
                return cls_embeddings
        except Exception as e:
            sys.exit(f"Exception encountered while loading pickle object: {feature_embeddings_path}.\n{e}")
    else:
        tokenized_inputs = tokenize_messages(
            data_df['Message'].tolist(),
        )
        cls_embeddings = extract_features(tokenized_inputs, device)

        print("---" * 30)
        user_response = input("Would you like to save the features for later use? [y/n]")
        if user_response in ["y", "Y", "yes", "Yes", "YES"]:
            output_dir = pathlib.Path(OUTPUT_DIR, "features")
            random_uuid = uuid.uuid4()
            print("---" * 30)
            output_pickle_filename = str(random_uuid) + ".pkl"
            output_path = pathlib.Path(output_dir, output_pickle_filename)
            try:
                if not output_dir.is_dir():
                    os.makedirs(output_dir)
                with open(output_path, "wb") as pickle_file:
                    pickle.dump(cls_embeddings, pickle_file)
                    print(f"Saved featureset to file named '{output_path}'")
            except Exception as e:
                sys.exit(f"Exception encountered while saving pickle object to {output_path}\n{e}")

        return cls_embeddings
