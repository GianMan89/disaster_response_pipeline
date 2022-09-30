# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # create a dataframe of the 36 individual category columns
    categories = categories["categories"].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # extract a list of new column names for categories.
    category_colnames = [x[:-2] for x in list(row)]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # merge datasets
    df = pd.merge(messages, categories, on=["id"])
    # return merged dataset
    return df


def clean_data(df):
    # drop duplicates
    df.drop_duplicates(subset=["message"], inplace=True)
    # return cleaned dataset
    return df


def save_data(df, database_filename):
    # Save the clean dataset into an sqlite database
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("clean_dataset", engine, index=False)
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