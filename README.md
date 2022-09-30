# disaster_response_pipeline
Udacity Data Scientist Nanodegree: analyze disaster data from Appen to build a model for an API that classifies disaster messages.

# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Instructions](#instructions)
	1. [Dependencies](#dependencies)
	2. [Executing Program](#execution)
3. [Licensing, Authors, and Acknowledgements](#licensing)

<a name="description"></a>
## 1. Description:

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Appen.
The aim is to analyze disaster data and to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. A machine learning pipeline was created to categorize these events so that they can be send to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

<a name="instructions"></a>
## 2. Instructions:
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
## 3. Licensing, Authors, and Acknowledgements:

Code is under MIT license (see file).

Author: Gianluca Manca

Acknowledgements to:
* [Udacity](https://www.udacity.com/) for the Data Science Nanodegree program.
* [Appen](https://appen.com/) for the disaster data.