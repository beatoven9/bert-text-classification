import argparse
import pathlib
import sys

import torch

from bert_classifier.classifier import train
from bert_classifier.constants import OUTPUT_DIR
from bert_classifier.csv_loader import load_csv
from bert_classifier.feature_extraction import get_embeddings


def main():
    """
    Main entrypoint to the model trainer.
    """
    parser = argparse.ArgumentParser(
        prog='Bert Classifier',
        description='This program takes CSV files of the format [Text, Category]',
        epilog='For more information on how to run this program, refer to the README file.'
    )


    parser.add_argument(
        '-i',
        '--input_csv',
        help="The csv file for training and testing. The program itself will handle splitting for training and validation."
    )
    parser.add_argument(
        '-e',
        '--max_epochs',
        type=int,
        default=100,
        help="The maximum epochs the program should train the model for."
    )

    parser.add_argument(
        '-f',
        '--features',
        default=None,
        help="This is the path to a pickle object of the feature embeddings from a previous run. It can speed up fine tuning iterations to use this parameter."
    )

    parser.add_argument(
        '-o',
        '--optimizer',
        choices=["adam", "sdg"],
        default="adam",
        help="This is the optimizer to be used."
    )

    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        default=0.01,
        help="This is the learning rate for the optimizer. A good starting value is .01 which is the default."
    )

    parser.add_argument(
        '-m',
        '--momentum',
        type=float,
        default=0.9,
        help="This is the momentum used for the SDG optimizer. A good starting value is .09 which is the default."
    )

    parser.add_argument(
        '-p',
        '--patience',
        type=int,
        default=3,
        help="Training checks at the end of each epoch if accuracy has worsened. If it has worsened for a certain amount of iterations in a row, it will stop and load the best recorded model. This parameter, patience, sets the number of epochs in a row the model is allowed to get worse."
    )

    parser.add_argument(
        '-r',
        '--random_seed',
        type=int,
        default=42,
        help="The random seed of the program. This is for reproducibility. If you play around with this parameter, make sure to note which seed you used so that you can replicate it."
    )

    args = parser.parse_args()
    filepath = pathlib.Path(args.input_csv)
    max_epochs = args.max_epochs
    feature_embeddings_path = args.features
    optimizer_string = args.optimizer
    learning_rate = args.learning_rate
    momentum = args.momentum
    patience = args.patience
    random_seed = args.random_seed

    device = get_torch_device()

    data_df = load_csv(filepath)
    cls_embeddings = get_embeddings(
        feature_embeddings_path=feature_embeddings_path,
        data_df=data_df,
        device=device,
    )

    model = train(
        cls_embeddings=cls_embeddings,
        device=device,
        data_df=data_df, 
        max_epochs=max_epochs,
        optimizer_string=optimizer_string,
        learning_rate=learning_rate,
        momentum=momentum,
        _patience=patience,
        random_seed=random_seed
    )

    print("---" * 30)
    user_response = input("Would you like to save this model? [y/n]")
    if user_response in ["y", "Y", "yes", "Yes", "YES"]:
        print("---" * 30)
        output_dir = pathlib.Path(OUTPUT_DIR, "models")
        model_filename = input("What would you like to call the file? ('.pth' is the most common file extension.)\n\tfile name: ")
        output_path = pathlib.Path(output_dir, model_filename)
        torch.save(model.state_dict(), output_path)
        print(f"Model saved to: {output_path}")
    elif user_response in ["n", "N", "no", "No", "NO"]:
        print("---" * 30)
        sys.exit("User exited program.")
    else:
        print("---" * 30)
        sys.exit(f"{user_response} is not a valid response. Exiting now.")


def get_torch_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("---" * 30)
    user_response = input(f"Torch will operate on a {device} device. Proceed? [y/n]")
    if user_response in ["y", "Y", "yes", "Yes", "YES"]:
        print("---" * 30)
        return device
    elif user_response in ["n", "N", "no", "No", "NO"]:
        print("---" * 30)
        sys.exit("User exited program.")
    else:
        print("---" * 30)
        sys.exit(f"{user_response} is not a valid response. Exiting now.")


if __name__=="__main__":
    main()
