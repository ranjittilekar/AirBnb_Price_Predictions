import os
import pandas as pd
import numpy as np
import program
import zipfile


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    #Define file name
    train_filename = '/train.csv'
    test_filename = '/test.csv'
    sample_filename = '/sample_submission.csv'

    #Read CSV file
    train_detail =  pd.read_csv(input_filepath + train_filename)
    test_detail =  pd.read_csv(input_filepath + test_filename)
    sample_detail=  pd.read_csv(input_filepath + sample_filename)

    #Create Source Indicator
    train_detail['Source'] = "Train"
    print("Train Shape: {}".format(train_detail.shape))

    test_detail['Source'] = "Test"
    print("Test Shape: {}".format(test_detail.shape))

    # Append Test and Train Data For Feature Engineering
    Train_Test_Append = pd.concat([train_detail, test_detail], axis=0)
    print("Train Test Append Shape: {}".format(Train_Test_Append.shape))

    Train_Test_Append.to_csv(output_filepath + '/Train_Test_Append.csv', index=False, encoding='utf-8')

    print("Make Dataset Complete")


def import_external(input_filepath, output_filepath):

    # Zillow data import
    zillow_filename = '/Zip_Zhvi_Summary_AllHomes.csv'
    zillow = pd.read_csv(input_filepath + zillow_filename)
    zillow = zillow[['RegionName','City','Zhvi']]
    zillow = zillow[(zillow.City =='Boston') | (zillow.City =='Chicago') | (zillow.City =='San Francisco') |
        (zillow.City =='Los Angeles') | (zillow.City =='Washington') | (zillow.City =='New York') ]

    # Rename cities name to match the Zillow dataset and train dataset
    name_map = {'New York':'NYC','San Francisco':'SF','Los Angeles':'LA','Washington':'DC','Chicago':'Chicago','Boston':'Boston'}

    zillow['City'] = zillow['City'].apply(lambda x:name_map[x])

    # Apply log to the Zillow home value index
    zillow['Zhvi'] = zillow['Zhvi'].apply(lambda x:np.log(x))

    # Rename column to be able to merge
    zillow.columns = ['zipcode','city','zhvi']
    # Change data type to be able to merge
    zillow['zipcode']= zillow['zipcode'].astype('str')

    print("Zillow data Shape: {}".format(zillow.shape))

    zillow.to_csv(output_filepath + '/zillow_ready.csv', index=False, encoding='utf-8')

    print("Make Zillow Dataset Complete")


def unzipfile(path_to_zip_file, directory_to_extract_to):
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()



if __name__ == '__main__':
    print("Run in make_dataset.py. To execute full script, run program.py.")
