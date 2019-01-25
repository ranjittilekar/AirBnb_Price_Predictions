import os
from data import make_dataset
from features import build_features
import parameters
from models import M1_CatBoost

import pandas as pd


class Program:
    def __init__(self):
        self.rootDirectory = str(os.path.abspath('..'))

# Main function call in Program (Run all workflow)
def main():

    # Run load data when selected at parameter
    if parameters.Parameters().run_load_data == True:
        fx_make_dataset()
        fx_build_features()
        preprocess_dataset()

    # Run the Catboost data pipeline
    catboost_model()


# Helper Function called through main function or notebooks
def fx_make_dataset():
    #Unzip data files
    make_dataset.unzipfile(rootDirectory + "/data/raw/SourceFiles.zip", rootDirectory + "/data/raw")
    # Make Data Set
    make_dataset.main("{}{}".format(rootDirectory, "/data/raw"), "{}{}".format(rootDirectory, "/data/interim"))
    # Make External Zillow Data
    make_dataset.import_external("{}{}".format(rootDirectory, "/data/external"), "{}{}".format(rootDirectory, "/data/interim"))


def fx_build_features():
    # Features
    Test_Train_Append = pd.read_csv(rootDirectory + "/data/interim/Train_Test_Append.csv")
    zillow = pd.read_csv(rootDirectory + "/data/interim/zillow_ready.csv")

    Test_Train_Append_Features =  build_features.feature_engineering(Test_Train_Append, zillow)
    build_features.output_fetaured_data(Test_Train_Append_Features, rootDirectory + "/data/interim/Train_Test_Append_Featured.csv")


def preprocess_dataset():
    # Load data from CSV
    test_train_append_features = pd.read_csv(rootDirectory + "/data/interim/Train_Test_Append_Featured.csv")
    # Transform Data
    preprocessed_data = build_features.preprocessing(test_train_append_features)
    # Split Data to XTrain, YTrain, XTest
    build_features.split_data(test_train_append_features, preprocessed_data, rootDirectory + "/data/processed")


def catboost_model():

    # Load data from CSV
    X_train = pd.read_csv(rootDirectory + "/data/processed/X_train.csv")
    y_train = pd.read_csv(rootDirectory + "/data/processed/y_train.csv")
    X_test = pd.read_csv(rootDirectory + "/data/processed/X_test.csv")

    # Load Test Data to Pull ID
    Original_Test = pd.read_csv(rootDirectory + "/data/raw/test.csv")
    test_Id = Original_Test['id']

    # Call CatBoost Model
    catmodel = M1_CatBoost.CatBoost_Model(X_train, y_train, X_test)


    if parameters.Parameters().run_train == True:
        # Perform Train Test Split Sequence
        catmodel.train_test_split()
        catmodel.train_pre_model()
        catmodel.predict_train_data()
        catmodel.cross_validated_catboost()
        catmodel.log_train_results()

    if parameters.Parameters().run_test == True:
        # Perform Full Train and Final Test Sequence
        catmodel.train_final_model()
        catmodel.predict_final_model()
        catmodel.output_submission(test_Id)


    # Perform Hyper-tuning (Work in Progress)
    # catmodel.cross_validated_catboost()




if __name__ == '__main__':
    # Initate variable to create Root Path
    rootDirectory = os.path.abspath('..')
    rootDirectory = str(rootDirectory)
    print(rootDirectory)

    # Set Panda Display options to show all columns
    pd.set_option('display.max_columns', None)

    # Run Main command
    main()
    print("Run in Main Function: program.py")