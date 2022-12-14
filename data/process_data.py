# import modules
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv files

    Arguments:
        messages_filepath: path to disaster_messages.csv
        categories_filepath: path to disaster_categories.csv
    Output:
        df: a dataframe containing features
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on=["id"])
    # return merged dataset
    return df


def clean_data(df):
    """
    Clean the dataset by converting the feature values to
    binary classifications, removing irrelevant columns.

    Arguments:
        df: a dataframe containing features
    Output:
        df: a dataframe containing cleaned features
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # extract a list of new column names for categories.
    category_colnames = [x[:-2] for x in list(row)]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(columns=["categories"], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(subset=["message"], inplace=True)
    # remove category 'child alone' which has only zero values
    df = df.drop(["child_alone"], axis=1)
    # some values in category 'related' are two, these are set to one
    df["related"] = df["related"].map(lambda x: 1 if x == 2 else x)
    # return cleaned dataset
    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataset in a SQLite database.

    Arguments:
        df: a dataframe containing features
        database_filename: path to the to be saved database
    Output:
        None
    """
    # Save the clean dataset into an sqlite database
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("cleaned_dataset", engine, index=False, if_exists="replace")
    return None


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
