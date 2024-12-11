import pandas as pd
import pathlib
import sys


def load_csv(filepath):
    data_df = pd.read_csv(filepath)

    data_df["Category"] = data_df["Category"].map({'ham': 0, 'spam': 1})

    print("---" * 30)
    print("DataFrame header: \n", data_df.head())
    print("\n\n")
    print("Example count: ", data_df.shape[0])
    print("---" * 30)
    user_response = input("Does this look right? Shall we proceed? [y,n]")
    if user_response in ["y", "Y", "yes", "Yes", "YES"]:
        print("---" * 30)
        return data_df
    elif user_response in ["n", "N", "no", "No", "NO"]:
        print("---" * 30)
        sys.exit("User exited program.")
    else:
        print("---" * 30)
        sys.exit(f"{user_response} is not a valid response. Exiting now.")
