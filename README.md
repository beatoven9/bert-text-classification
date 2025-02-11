# Bert Text Classification

## Description
The main purpose of this was to build an outline for a project that I'm doing on a secure system with restricted access to the internet. As such, development on the system is much more tedious since vanilla vim/emacs (no plugin installations) are the only text editors available on the system. So I built this so that I could move a starting point to the computer. The network in this example is extremely simple, but it will be expanded upon on the secure system.

This is a Binary Classification model trained using the dataset from a kaggle [dataset](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification).

The features of the text are extracted using bert-base-uncased from the [transformers](https://pytorch.org/hub/huggingface_pytorch-transformers/) module provided by pytorch.

Extracting the features from the dataset is by far the most computationally expensive step. Therefore, once the features have been extracted, the user is asked whether or not they'd like to save the features for future use. If they answer "yes", they can provide the file as a commandline option for future runs of the program.

A handful of options are made available to the user via the commandline for finetuning. These command line options are viewable via the program's help message.

## Installation
To install this package, simply git clone the repository, navigate to the directory where the setup.py file lives, and run `pip install -e .` while in a virtual environment.

## Use
Once installed, you can view the help message via the following command: `python -m bert_classifier --help`

The output of the help message has been provided below for your convenience.

```
usage: Bert Classifier [-h] [-i INPUT_CSV] [-e MAX_EPOCHS] [-f FEATURES] [-o {adam,sdg}] [-l LEARNING_RATE] [-m MOMENTUM] [-p PATIENCE] [-r RANDOM_SEED]

This program takes CSV files of the format [Text, Category]

options:
  -h, --help            show this help message and exit
  -i INPUT_CSV, --input_csv INPUT_CSV
                        The csv file for training and testing. The program itself will handle splitting for training and validation.
  -e MAX_EPOCHS, --max_epochs MAX_EPOCHS
                        The maximum epochs the program should train the model for.
  -f FEATURES, --features FEATURES
                        This is the path to a pickle object of the feature embeddings from a previous run. It can speed up fine tuning iterations to use this parameter.
  -o {adam,sdg}, --optimizer {adam,sdg}
                        This is the optimizer to be used.
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        This is the learning rate for the optimizer. A good starting value is .01 which is the default.
  -m MOMENTUM, --momentum MOMENTUM
                        This is the momentum used for the SDG optimizer. A good starting value is .09 which is the default.
  -p PATIENCE, --patience PATIENCE
                        Training checks at the end of each epoch if accuracy has worsened. If it has worsened for a certain amount of iterations in a row, it will stop and load the best recorded model. This parameter, patience, sets the number of
                        epochs in a row the model is allowed to get worse.
  -r RANDOM_SEED, --random_seed RANDOM_SEED
                        The random seed of the program. This is for reproducibility. If you play around with this parameter, make sure to note which seed you used so that you can replicate it.
```

An example run of how to use the program is provided below:

`python -m bert_classifier -i ./data/spam_text_message_dataset.csv -e 200 -o adam -l .01 -p 3 -r 41 -f features/80370f26-eae3-4129-8069-0dc615224ca6.pkl`
