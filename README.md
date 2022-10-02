# disaster_response_pipeline
Udacity Data Scientist Nanodegree: analyze disaster data from Appen to build a model for an API that classifies disaster messages.

# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Files in the Repository](#files)
3. [Instructions](#instructions)
	1. [Dependencies](#dependencies)
	2. [Executing Program](#execution)
4. [Licensing, Authors, and Acknowledgements](#licensing)

<a name="description"></a>
## 1. Description:

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Appen.
The aim is to analyze disaster data and to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. A machine learning pipeline was created to categorize these events so that they can be send to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

<a name="files"></a>
## 2. Files in the Repository:

The repository consists of the following folders and files:

***app***: contains HTML and Flask files that build and run the web-based API that classifies disaster messages.
***data***: contains two csv files: disaster_categories.csv (class labels) and disaster_messages.csv (to be classified disaster messages and tweets), a SQLite database of the cleaned input data: Disasterresponse.db, and a Python script that loads, cleans, and saves the disaster messages and categories into the database, process_data.py.
***models***: a Pickle file of the trained classification model, classifier.pkl, and a Python script that loads, preprocesses, trains, tests, tunes, and saves the classification model. The model consists of a sklearn pipeline for data preprocessing and classification using sklearn CountVectorizer, TfidfTransformer, RandomForestClassifier, and GridSearchCV.

The repository is structured as follows:

app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # labels of the message data
|- disaster_messages.csv # disaster message data
|- process_data.py # load, clean, and save data to a SQLite database
|- DisasterResponse.db # database to save clean data to
models
|- train_classifier.py # build, train, test, and save a classification pipeline
|- classifier.pkl # saved model
LICENSE # MIT license for this repository
poetry.lock # log file of all installed Python packages
pyproject.toml # package summary and dependencies
README.md

<a name="instructions"></a>
## 3. Instructions:
<a name="dependencies"></a>
### 1. Dependencies:

All relevant modules, versions and dependencies can be found in the 
poetry.lock and pyproject.toml

<a name="execution"></a>
### 2. Executing Program:

    1. Run the following commands in the project's root directory to set up your database and model.

        - To run ETL pipeline that cleans data and stores in database
            `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        - To run ML pipeline that trains and evaluates classifier and saves
            `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
        Note: this repository already includes both DisasterResponse.db and classifier.pkl. Delete them if necessary.

    2. Go to `app` directory: `cd app`

    3. Run your web app: `python run.py`

    4. Open http://127.0.0.1:3000/ in your browser.

<a name="licensing"></a>
## 4. Licensing, Authors, and Acknowledgements:

Code is under MIT license (see file).

Author: Gianluca Manca

Acknowledgements to:
* [Udacity](https://www.udacity.com/) for the Data Science Nanodegree program.
* [Appen](https://appen.com/) for the disaster data.