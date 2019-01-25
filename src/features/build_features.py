import pandas as pd
import parameters
from sklearn.preprocessing import Imputer
import numpy as np
from scipy.stats import skew

def feature_engineering(data, zillow):

    # ---------------------
    # Fill NA
    # ---------------------

    # continuous_eng_features = []
    # data[continuous_eng_features] = data[continuous_eng_features].astype(float).fillna(data[continuous_eng_features].median())
    #
    # categorical_eng_features = ['amenities','description', 'cleaning_fee']
    # data[categorical_eng_features] = data[categorical_eng_features].astype(object).fillna("Blank")


    # ---------------------
    # Engineer Features
    # ---------------------

    #-------- original -------#


    # To deal with large right and left skews, log normal all numeric values
    # numeric_feats = data.dtypes[data.dtypes != "object"].index
    # numeric_to_exlude = ['id','latitude','longitude','log_price']
    # numeric_feats_final = list(set(numeric_feats) - set(numeric_to_exlude))
    # print("Numeric Fields For Log Normal Considerations: {}".format(numeric_feats_final))
    #
    # skewed_feats = data[numeric_feats_final].apply(lambda x: skew(x.dropna())) #compute skewness
    # skewed_feats = skewed_feats[skewed_feats > 0.75]
    # skewed_feats = skewed_feats.index
    #
    # data[skewed_feats] = np.log1p(data[skewed_feats])


    # # Count the number of aminities
    # data['amenities_count'] = data['amenities'].apply(lambda x: len(x.split()))
    #
    # # Count the number of words in the description
    # data['description_length'] = data['description'].apply(lambda x: len(x.split()))
    #
    # # Replace True and False with 1 and 0
    # data['cleaning_fee'] = data['cleaning_fee'].apply(lambda x: 1 if x == True else 0)
    #
    # # Replcase existing thumbnail by 1 and missig thumbnail by 0
    # data['thumbnail_url'].loc[data['thumbnail_url'].notnull()] = 1
    # data['thumbnail_url'].loc[data['thumbnail_url'].isnull()] = 0
    #
    # # Extract year from first_review field. Missing first_review date replaces by 9999
    # # For this variable we could create a dummy variable
    # data['first_review'].loc[data['first_review'].isnull()] = "9999"
    # data['first_review'] = data['first_review'].apply(lambda x: x[:4])
    #
    # # Initializing the imputer
    # imp_most_frequent = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    #
    # # convert formatting for host_response_rate, removing % signs
    # data['host_response_rate'] = (data['host_response_rate'].str.replace(r'[^-+\d.]', '').astype(float))
    # data['host_response_rate'] = imp_most_frequent.fit_transform(data[['host_response_rate']])

    #-------- original -------#


    ## Initializing the imputer
    imp_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp_median = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp_most_frequent = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

    ## Imputing Host_response_rate
    # convert formatting for host_response_rate, removing % signs
    data['host_response_rate'] = (data['host_response_rate'].str.replace(r'[^-+\d.]', '').astype(float))
    data['host_response_rate'] = imp_median.fit_transform(data[['host_response_rate']])

    ## Imputing review_scores_rating
    data['review_scores_rating'] = imp_mean.fit_transform(data[['review_scores_rating']]).astype(int)

    ## replace NaN by 'NO Date' and treat the missing values correctly
    data['neighbourhood'] = data['neighbourhood'].replace(np.NaN, 'not provided')
    data['zipcode'] = data['zipcode'].replace(np.NaN, 'not provided')

    ## Imputing Bathrooms and bedrooms variables
    data['bathrooms'] = imp_median.fit_transform(data[['bathrooms']]).astype(int)
    data['bedrooms'] = imp_median.fit_transform(data[['bedrooms']]).astype(int)
    data['beds'] = imp_median.fit_transform(data[['beds']]).astype(int)

    ## Most frequent
    data['host_identity_verified'] = data['host_identity_verified'].replace(['t', 'f'], [1, 0])
    host_identity_verified_median = data["host_identity_verified"].median()
    data['host_identity_verified'] = data['host_identity_verified'].replace([np.NaN],
                                                                            [host_identity_verified_median]).astype(int)

    data['host_has_profile_pic'] = data['host_has_profile_pic'].replace(['t', 'f'], [1, 0])
    host_has_profile_pic_median = data["host_has_profile_pic"].median()
    data['host_has_profile_pic'] = data['host_has_profile_pic'].replace([np.NaN], [host_has_profile_pic_median]).astype(
        int)

    # Count the number of aminities
    data['amenities_count'] = data['amenities'].apply(lambda x: len(x.split()))

    data['name'] = data['name'].replace([np.NaN], ["Not Provided"])

    # Count the number of words in the description
    data['description'] = data['description'].replace([np.NaN], ["Not Provided"])
    data['description_length'] = data['description'].apply(lambda x: len(x.split()))

    # Replace True and False with 1 and 0
    data['cleaning_fee'] = data['cleaning_fee'].apply(lambda x: 1 if x == True else 0)

    # Replcase existing thumbnail by 1 and missig thumbnail by 0
    data['thumbnail_url'].loc[data['thumbnail_url'].notnull()] = 1
    data['thumbnail_url'].loc[data['thumbnail_url'].isnull()] = 0

    # Extract year from first_review field. Missing first_review date replaces by 9999
    # For this variable we could create a dummy variable
    data['first_review'].loc[data['first_review'].isnull()] = "9999"


    data['last_review'].loc[data['last_review'].isnull()] = "9999"


    data['host_since'].loc[data['host_since'].isnull()] = "9999"


    # Original Zillow Engineer

    # Merge Zillow external data
    zillow['zipcode']= zillow['zipcode'].astype('str')
    data = data.merge(zillow,how='left', on = ['zipcode','city'])
    # Fill missing with mean by city
    data["zhvi"] = data[['city','zhvi']].groupby("city").transform(lambda x: x.fillna(x.mean()))


    print("List of Final Headers")
    print(data.dtypes)

    print("Feature Engineering Complete")

    return data

#
#
# def one_hot_encode_amenities(data):
#
#     amenities_list = []
#     for i in range(len(train_detail)):
#         amenities_list += train_detail.amenities[i][1:-1].split(',')
#     amenities_list = list(set(amenities_list))
#     amenities_list_dict_train = {}
#     amenities_list_dict_test = {}
#
#     for a in amenities_list:
#         amenities_list_dict_train[a] = []
#         amenities_list_dict_test[a] = []
#
#     # amenities_list_dict
#
#     for i in range(len(train_detail)):
#         l = train_detail.amenities[i][1:-1].split(',')
#         for a in amenities_list_dict_train.keys():
#             if a in l:
#                 amenities_list_dict_train[a].append(1)
#             else:
#                 amenities_list_dict_train[a].append(0)
#
#     amenities_table_train = pd.DataFrame(amenities_list_dict_train)
#     amenities_table_train = amenities_table_train.drop([''], axis=1)
#
#     for i in range(len(test_detail)):
#         l = test_detail.amenities[i][1:-1].split(',')
#         for a in amenities_list_dict.keys():
#             if a in l:
#                 amenities_list_dict_test[a].append(1)
#             else:
#                 amenities_list_dict_test[a].append(0)
#
#     amenities_table_test = pd.DataFrame(amenities_list_dict_test)
#     amenities_table_test = amenities_table_test.drop([''], axis=1)
#
#     train_detail = pd.concat([train_detail, amenities_table_train], axis=1)
#
#     test_detail = pd.concat([test_detail, amenities_table_test], axis=1)





def output_fetaured_data(data, filepath):
    data.to_csv(filepath, index=False, encoding='utf-8')
    print("Saved Train Test Append Featured CSV")


def preprocessing(data):

    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    input = parameters.Parameters()

    data = pd.DataFrame(data)

    # Pull Selected Continuous Variables from Parameters
    continuous_variables = input.continuous_variables
    categorical_variables = input.categorical_variables

    print("Continuous Variable")
    print(continuous_variables)
    print("Categorical Variable")
    print(categorical_variables)

    # Select Train and test data, select features
    source_col = ['Source']
    process_col = continuous_variables + categorical_variables

    # Reduce to in scope columns
    data_source = data.loc[:, source_col]
    data = data.loc[:, process_col]

    # print(data.head())

    # Standardize the data type, Fill Numeric Blank with Median
    data[continuous_variables] = data[continuous_variables].astype(float).fillna(data[continuous_variables].median())
    data[categorical_variables] = data[categorical_variables].astype(object).fillna("Blank")

    #Find Missing Values
    # Searching by detail for missing values
    missing_features = data.isnull().sum(axis=0).reset_index()
    missing_features.columns = ['column_name', 'count']
    missing_features = missing_features.sort_values(by='count')
    print("Find Missing Features")
    print(missing_features)


    # Continuous variable scaler option
    if input.numeric_scaler_option == 1:
        minmax_scaler = MinMaxScaler() # default=(0, 1)
        data[continuous_variables] = minmax_scaler.fit_transform(data[continuous_variables])
    elif input.numeric_scaler_option == 2:
        std_scaler = StandardScaler()
        data[continuous_variables] = std_scaler.fit_transform(data[continuous_variables]).round(3)
    else:
        print("Cannot Detect numeric scaler parameter")


    # One-Hot-Encoding: Create array of dummies
    if input.one_hot_encode == True:
        data = pd.get_dummies(data, drop_first = input.drop_first, dummy_na=False)


    data = pd.merge(data, data_source, left_index=True, right_index=True)

    # Display shape
    print("Complete: The preprocessed data shape: {}".format(data.shape))

    return data



def split_data(main_data, processed_data, filepath):

    input = parameters.Parameters()

    # Select headers
    train_col = input.continuous_variables + input.categorical_variables
    test_col = ['log_price']

    #X_Train
    X_train = processed_data.loc[(processed_data['Source']== "Train")]
    X_train = X_train.drop('Source', axis=1)

    # y_Train
    y_train = main_data[test_col].loc[(main_data['Source']== "Train")]

    # X_Test
    X_test = processed_data.loc[(processed_data['Source']== "Test")]
    X_test = X_test.drop('Source', axis=1)

    # Display shape
    print ("X_train Shape: {}".format(X_train.shape))
    print ("y_train Shape: {}".format(y_train.shape))
    print ("X_test Shape: {}".format(X_test.shape))

    #Output Data
    X_train.to_csv(filepath + '/X_train.csv', index=False, encoding='utf-8')
    y_train.to_csv(filepath + '/y_train.csv', index=False, encoding='utf-8')
    X_test.to_csv(filepath + '/X_test.csv', index=False, encoding='utf-8')

    print("Split Data Complete.")


if __name__ == '__main__':
    print("Run in build_features.py. To execute full script, run program.py.")
