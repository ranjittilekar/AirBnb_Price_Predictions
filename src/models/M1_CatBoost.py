import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import parameters
import program
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

class CatBoost_Model:


    def __init__(self, X_train, y_train, X_test ):
        # Load Parameter
        self.input = parameters.Parameters()

        # Modeling Values
        self.cat_model = None
        self.cat_final_model = None
        self.X_train_model = None
        self.X_test_model = None
        self.y_train_model = None
        self.y_test_model = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = None
        self.train_R2 = None
        self.test_R2 = None
        self.train_RMSE = None
        self.test_RMSE = None
        self.cv_train_RMSE = None
        self.cv_test_RMSE = None
        self.best_score = None
        self.best_iteration = None
        self.categorical_features_indices = None

    def train_test_split(self):

        print("Run CatBoost Train Test Split")
        ## train_features and train_target
        self.X_train_model, self.X_test_model, self.y_train_model, self.y_test_model = train_test_split(
                                        self.X_train,
                                        self.y_train,
                                        test_size=self.input.test_size,
                                        random_state=self.input.random_state)

        # Show the results of the split
        print("Training_model set has {} rows and {} columns.".format(self.X_train_model.shape[0], self.X_train_model.shape[1]))
        print("Testing_model set has {} samples and {} columns.".format(self.X_test_model.shape[0], self.X_test_model.shape[1]))


    def train_pre_model(self):

        # self.cat_model = CatBoostRegressor(iterations=self.input.iterations,
        #                                 depth=self.input.depth,
        #                                 learning_rate=self.input.learning_rate,
        #                                 loss_function=self.input.loss_function,
        #                                 logging_level=self.input.logging_level)

        self.cat_model = CatBoostRegressor(**self.input.params, use_best_model = self.input.use_best_model)

        # Create np list of categorical variables
        # categorical_features_indices = self.input.categorical_variables
        self.categorical_features_indices = np.where(self.X_train.dtypes == np.object)[0]

        print(self.categorical_features_indices)

        self.cat_model.fit(self.X_train_model,
                           self.y_train_model,
                           cat_features=self.categorical_features_indices,
                           verbose=self.input.verbose,
                           eval_set=(self.X_test_model, self.y_test_model)
                           )

        print ("Train CatBoost Pre-Train Model Complete")


    def  predict_train_data(self):

        # Run prediction on train data only
        train_prediction = self.cat_model.predict(self.X_train_model)
        self.train_R2 = r2_score(self.y_train_model, train_prediction)
        print("Train R2 Score: {}".format(self.train_R2))
        self.train_RMSE = np.sqrt(mean_squared_error(self.y_train_model,train_prediction))
        print("Train RMSE Score: {}".format(self.train_RMSE))


        # Run prediction on test data only
        test_prediction = self.cat_model.predict(self.X_test_model)
        self.test_R2 = r2_score(self.y_test_model, test_prediction)
        print("Test R2 Score: {}".format(self.test_R2))
        self.test_RMSE = np.sqrt(mean_squared_error(self.y_test_model,test_prediction))
        print("Test RMSE Score: {}".format(self.test_RMSE))



    def train_final_model(self):

        # self.cat_final_model = CatBoostRegressor(iterations=self.input.iterations,
        #                                 depth=self.input.depth,
        #                                 learning_rate=self.input.learning_rate,
        #                                 loss_function=self.input.loss_function,
        #                                 logging_level=self.input.logging_level)

        self.cat_final_model = CatBoostRegressor(**self.input.params)

        # Create np list of categorical variables
        # categorical_features_indices = np.where(self.X_train.dtypes != np.float)[0]
        categorical_features_indices = np.where(self.X_train.dtypes == np.object)[0]
        print(categorical_features_indices)

        # Train model with full data
        self.cat_final_model.fit(self.X_train,
                           self.y_train,
                           cat_features=categorical_features_indices,
                           verbose=self.input.verbose
                           )

        print ("Train CatBoost Final Model Complete")


    def predict_final_model(self):
        # Make final prediction of test data
        self.y_test = self.cat_final_model.predict(self.X_test)
        print("Final Prediction Results")
        print(self.y_test)



    def output_submission(self, Test_Id):

        # Transform to 1d array
        self.y_test = np.ravel(self.y_test)

        # Output table
        output = pd.DataFrame({'id': Test_Id,
                               'log_price': self.y_test}).set_index('id')

        print("Final Output Shape: {}".format(output.shape))
        print("Final Output Head: {}".format(output.head()))

        # Export CSV
        if self.input.output_submission_file == True:
            output.to_csv('{}'.format(self.input.CatBoost_submission_name), index=True)
            print ("Output CSV Complete: {}".format(self.input.CatBoost_submission_name))
        else:
            print ("Output CSV Parameter Turned Off. Update Parameter to extract CSV")



    def log_train_results(self):

        #Log results for Train Test Split
        model_log = pd.read_csv(program.Program().rootDirectory + "/models/log/catboost_model_log.csv")

        new_data = {"model": "CatBoost",
                    "run_time": self.input.date,
                    "random_state": self.input.random_state,
                    "test_size": self.input.test_size,
                    "iterations": self.input.iterations,
                    "depth": self.input.depth,
                    "learning_rate": self.input.learning_rate,
                    "loss_function": self.input.loss_function,
                    "logging_level": self.input.logging_level,
                    "continuous_variables": self.input.continuous_variables,
                    "categorical_variables": self.input.categorical_variables,
                    "submission_filename": self.input.CatBoost_submission_name,
                    "best_score": self.best_score,
                    "best_iteration": self.best_iteration,
                    "train_R2": self.train_R2,
                    "test_R2": self.test_R2,
                    "train_RMSE": self.train_RMSE,
                    "test_RMSE": self.test_RMSE,
                    "cv_train_RMSE": self.cv_train_RMSE,
                    "cv_test_RMSE": self.cv_test_RMSE
                    }

        # Log results
        model_log = model_log.append(new_data, ignore_index=True)

        # Update log file
        model_log.to_csv(program.Program().rootDirectory + "/models/log/catboost_model_log.csv", index=False)

        print ("Log Train Results Complete. Saved to: {}".format(program.Program().rootDirectory + "/models/log/catboost_model_log.csv"))


    def cross_validated_catboost(self):

        kf = KFold(n_splits=self.input.fold_count, shuffle=self.input.shuffle)

        cv_train_result = []
        cv_validation_result = []

        train_set = self.X_train_model
        train_label = self.y_train_model
        validation_set = self.X_test_model
        validation_label =self.y_test_model
        params = self.input.params

        for train_index, test_index in kf.split(train_set):
            print("-------- CV Loop ---------")

            train = train_set.iloc[train_index, :]
            test = train_set.iloc[test_index, :]

            labels = train_label.iloc[train_index, :]
            test_labels = train_label.iloc[test_index, :]

            clf = CatBoostRegressor(**params)

            clf.fit(train, np.ravel(labels), cat_features = self.categorical_features_indices, verbose=self.input.verbose)

            # Predict on CV Train Population
            cv_train_pred = clf.predict(test)

            # Run on Train CV
            cv_train_value = np.sqrt(mean_squared_error(cv_train_pred, np.ravel(test_labels)))
            cv_train_result.append(cv_train_value)

            # Predict on Validation Population
            cv_validation_pred = clf.predict(validation_set)

            # Run on Test CV
            cv_validation_value = np.sqrt(mean_squared_error(cv_validation_pred, np.ravel(validation_label)))
            cv_validation_result.append(cv_validation_value)

        self.cv_train_RMSE = np.mean(cv_train_result)
        print("Cross Validation train result: {}".format(cv_train_result))
        print("Cross Validation train average: {}".format(self.cv_train_RMSE))

        self.cv_test_RMSE = np.mean(cv_validation_result)
        print("Cross Validation test result: {}".format(cv_validation_result))
        print("Cross Validation test average: {}".format(self.cv_test_RMSE))



if __name__ == '__main__':
    print("Run in M1_CatBoost.py. To execute full script, run program.py.")