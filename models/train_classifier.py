# import modules
import sys
import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Load data from SQLite database

    Arguments:
        database_filepath: path to the SQLite database
    Output:
        X: a dataframe containing features
        Y: a dataframe containing labels for multiple categories
        category_names: list of category names
    """
    # load data from database
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("cleaned_dataset", engine)
    # put the messages that are used for classification in X
    X = np.array(df["message"])
    # put the multiclass classification labels in Y
    Y = np.array(df[df.columns[4:]])
    return (X, Y, df.columns[4:])


def build_model():
    """
    Build a pipeline for data preprocessing and classification
    using sklearn's CountVectorizer, TfidfTransformer, and
    RandomForestClassifier

    Arguments:
        None
    Output:
        model: the build sklearn pipeline
    """
    model = make_pipeline(
        CountVectorizer(),
        TfidfTransformer(),
        MultiOutputClassifier(RandomForestClassifier(n_estimators=10)),
    )
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the pipeline on the test set.

    Arguments:
        model: the build sklearn pipeline
        X_test: a dataframe containing features for the test set
        Y_test: a dataframe containing labels for multiple categories
        for the test set
        category_names: list of category names
    Output:
        None
    """
    # predict the classification labels for test set
    Y_pred = model.predict(X_test)
    # calculate the accuracy over all categories
    accuracy = (Y_pred == Y_test).mean().mean()
    print("Average overall accuracy {0:.2f}%".format(accuracy * 100))
    # print the classification report per category
    for i in range(Y_test.shape[1]):
        print("Model performance for category '{}':".format(category_names[i]))
        print(classification_report(Y_test[:, i], Y_pred[:, i]))
    return None


def save_model(model, model_filepath):
    """
    Export the trained model to a pickle file.

    Arguments:
        model: the build sklearn pipeline
        model_filepath: path to the pickle file
    Output:
        None
    """
    pickle.dump(model, open(model_filepath, "wb"))
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
