import pandas as pd
from datetime import datetime
import program

# Python ML Parameters

class Parameters:

    def __init__(self):

        # Log parameter run time
        self.date = datetime.now().strftime('%Y%m%d_%H%M%S')


        ''' 
        -------------------------
        Select Test Variables
        -------------------------
        '''

        # List of continuous variables
        self.continuous_variables =     [
                                        'accommodates',
                                        'bathrooms',
                                        # 'first_review',
                                        # 'host_response_rate',
                                        # 'host_since',
                                        # 'last_review',
                                        'number_of_reviews',
                                        # 'review_scores_rating',
                                        'bedrooms',
                                        # 'beds',
                                        # 'amenities_count',
                                        'description_length',
                                        'latitude',
                                        'longitude',
                                        'zhvi'
                                        ]

        # List of categorical variables
        self.categorical_variables =    [
                                        'property_type',
                                        'room_type',
                                        # 'bed_type',
                                        # 'cancellation_policy',
                                        # 'cleaning_fee',
                                        # 'city',
                                        # 'host_since',
                                        # 'description',
                                        # 'host_has_profile_pic',
                                        # 'host_identity_verified',
                                        # 'instant_bookable',
                                        # 'name',
                                        'neighbourhood',
                                        # 'thumbnail_url',
                                        #  'zipcode'
                                        ]


        ''' 
        -------------------------
        Preprocessing Parameters
        -------------------------
        '''

        # Normalize Continuous Parameter (1: Min Max Scaler , 2: Z-Score Standard Scaler)
        self.numeric_scaler_option = 2

        # Normalize Categorical Parameter
        self.one_hot_encode = False         # Catboosting does not require one hot encoding - True to turn on, False to turn off
        self.drop_first = False             # True: to drop First one hot ecode, False: to not drop First

        #Modeling Parameters: Train test split for repeatability
        self.random_state = 420
        self.test_size = 0.20

        ''' 
        -------------------------
        CatBoost Parameters
        -------------------------
        '''

        # Run load data and feature engineering
        self.run_load_data = True

        # Run Train - True turn train on. False turn train off
        self.run_train = True

        # Run Test - True turn test on. False turn test off
        self.run_test = True


        # General Parameters
        self.output_submission_file = True      # True creates csv file, False skips csv file
        self.CatBoost_save_model_name = "CatBoostModel"
        self.CatBoost_submission_name = program.Program().rootDirectory + "/models/submission/Sub_CatBoost {}.csv".format(self.date)
        self.use_best_model = True
        self.plot = True
        self.verbose = True

        # Modeling Parameter: Catboost Parameters (Final Prediction)
        self.iterations = 1000              # Default 1000
        self.depth = 5                      # Default 5
        self.learning_rate = 0.3            # Default 0.3
        self.loss_function = 'RMSE'
        self.logging_level = 'Silent'

        # Cross Validation Parameters
        self.fold_count = 3
        self.shuffle = True
        self.params = {'iterations': self.iterations,
                       'depth': self.depth,
                       'learning_rate': self.learning_rate,
                       'loss_function': self.loss_function,
                       'logging_level': self.logging_level}



        # # Modeling Parameter: Catboost Parameters (Grid Search)
        # self.gs_iterations = [250, 500, 750, 1000, 1250, 1500, 2000]
        # self.gs_depth = [3,5,7,10]
        # self.gs_learning_rate = [0.15, 0.3, 0.45, 0.6, 1, 1.5]
        # self.gs_loss_function = 'RMSE'
        # self.gs_logging_level = 'Silent'




    if __name__ == '__main__':
        print("Run in parameters.py. To execute full script, run program.py.")