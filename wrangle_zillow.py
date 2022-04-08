#Acquire and prep function for the zillow dataset - clustering module#

#imports
import numpy as np
import pandas as pd
import os
from env import get_db_url
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
###################################################################################################

#acquire zillow function
def acquire_zillow():
    '''
    This function checks for a copy of the dataset in the local directory 
    and pulls a new copy and saves it if there is not one,
    it then cleans the data by removing significant outliers then
    removing the rows with null values for 'yearbuilt'
    '''
    #assign the file name
    filename = 'zillow_clustering.csv'
    #check if the file exists in the current directory and read it if it is
    if os.path.exists(filename):
        print('Reading from csv file...')
        #read the local .csv into the notebook
        df = pd.read_csv(filename)
        return df
    #assign the sql query to a variable for use in pulling a new copy of the dataset from the database
    query = '''
    SELECT 
    prop_2017.*,
    log.logerror,
    log.transactiondate,
    airconditioningtype.airconditioningdesc,
    architecturalstyletype.architecturalstyledesc,
    buildingclasstype.buildingclassdesc,
    heatingorsystemtype.heatingorsystemdesc,
    propertylandusetype.propertylandusedesc,
    storytype.storydesc,
    typeconstructiontype.typeconstructiondesc
    FROM properties_2017 AS prop_2017
    JOIN (SELECT parcelid, MAX(transactiondate) AS max FROM predictions_2017 GROUP BY parcelid) AS pred_2017 USING(parcelid)
    LEFT JOIN (SELECT * FROM predictions_2017) AS log ON log.parcelid = pred_2017.parcelid AND log.transactiondate = pred_2017.max
    LEFT JOIN airconditioningtype USING(airconditioningtypeid) 
    LEFT JOIN architecturalstyletype USING(architecturalstyletypeid) 
    LEFT JOIN buildingclasstype USING(buildingclasstypeid) 
    LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid) 
    LEFT JOIN propertylandusetype USING(propertylandusetypeid) 
    LEFT JOIN storytype USING(storytypeid)
    LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)
    WHERE prop_2017.latitude IS NOT NULL;
    '''
    #if needed pull a fresh copy of the dataset from the database
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, get_db_url('zillow'))
    #save a copy of the dataset to the local directory as a .csv file
    df.to_csv(filename, index=False)
    return df

#############################################################################

#Missing values by column
def missing_col_values(df):
    '''this function returns a dataframe that has the count of null values by column
    '''
    cols_df = pd.DataFrame({'count' : df.isna().sum(), 'percent' : df.isna().mean()})
    return cols_df

#############################################################################

#missing values by row
def missing_row_values(df):
    '''this function returns a dataframe that has the count of null values by row
    '''
    rows_df = pd.concat([
    df.isna().sum(axis=1).rename('num_cols_missing'),
    df.isna().mean(axis=1).rename('pct_cols_missing'),
    ], axis=1).value_counts().to_frame(name='num_cols').sort_index().reset_index()
    return rows_df

#############################################################################

#find single unit properties function
def single_unit_properties(df):
    '''this function filters the dataframe to include only properties that are
    single family residences, mobile homes, manufactures homes, and cluster homes 
    and have a unit count of 1
    '''
    type_values = [261.0, 263.0, 275.0, 265.0]
    df = df[df.propertylandusetypeid.isin(type_values) == True]
    unit_values = [2.0, 3.0]
    df = df[df.unitcnt.isin(unit_values) == False]
    return df

##############################################################################

#remove columns you want to drop
def remove_columns(df, cols_to_remove): 
    '''this function removes identified columns to remove 
    and is used in the below data_prep function
    ''' 
    df = df.drop(columns=cols_to_remove)
    return df

#handle missing values by missing value percentage by columns then rows
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    '''this function removes columns with more than 50 percent null values 
    then removes remaing rows with 75 or more percent null values 
    and is used in the below prep_data function
    '''
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

#finish pre prep by dealing with remaining null values
def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    '''this function runs the above remove_columns and handle missing values functions 
    then fills in missing unit counts with 1 and calculates missing structure values
    '''
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    df['unitcnt'].fillna(1, inplace=True)
    df['structuretaxvaluedollarcnt'].fillna(df.taxvaluedollarcnt - df.landtaxvaluedollarcnt, inplace=True)
    df = df.dropna()
    return df

###################################################################################

#split the data
def split_data(df):
    '''
    this function takes the full dataset and splits it into three parts (train, validate, test) 
    and returns the resulting dataframes
    '''
    train_val, test = train_test_split(df, train_size = 0.8, random_state=123)
    train, validate = train_test_split(train_val, train_size = 0.7, random_state=123)
    return train, validate, test

####################################################################################

#remove outliers
def remove_outliers(df, k, col_list):
    ''' this function will remove outliers from a list of columns in a dataframe 
        and return that dataframe. A list of columns with significant outliers is 
        assigned to a variable in the below wrangle function and can be modified if needed
    '''
    #loop throught the columns in the list
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)] 
    return df

###################################################################################

#split dataset by region function
def split_by_region(df):
    ''' This function splits the original dataframe into three regional dataframes
     using the fips codes which represent the county where the property is located
    '''
    df1 = df[df['fips']==6037.0]
    df2 = df[df['fips']==6059.0]
    df3 = df[df['fips']==6111.0]
    return df1, df2, df3

##############################################################################

#scale the data
def scale_data(df):
    ''' This function takes in a dataframe and returns a dataframe 
    with the values scaled using a min-max scaler
    '''
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled_df = scaler.transform(df)
    df = pd.DataFrame(scaled_df, columns=df.columns, index=df.index)
    return df

#############################################################################

#prep for linear regression modeling
def lr_model_prep(cluster_df, original_df):
    ''' this function prepares the dataset for modeling with linear regression. 
    it takes in the two dataframes from the clustering phase and concatinates 
    the cluster columns to the scaled variable dataframe then it encodes the cluster columns 
    so all the columns are ready for regression modeling
    '''
    add_cols = original_df[['bedbathsqft_cluster', 'latlong_cluster', 'dist_cluster']]
    cluster_df = pd.concat([cluster_df, add_cols], axis=1)
    encode_cols = ['bedbathsqft_cluster', 'latlong_cluster', 'dist_cluster']
    for col in encode_cols:
        dummie_df = pd.get_dummies(cluster_df[col], prefix = cluster_df[col].name, drop_first = True)
        cluster_df = pd.concat([cluster_df, dummie_df], axis=1)
    return cluster_df

###############################################################################

#prep for clustering models
def cluster_model_prep(df):
    '''This function takes in the dataframe and creates a new dataframe containing
    the feature columns with scaled values so they are ready for clustering
    '''
    scale_cols = df[['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'latitude', 'longitude', 'dist_lat', 'dist_long']]
    scaler = MinMaxScaler()
    scaler.fit(scale_cols)
    scaled_df = scaler.transform(scale_cols)
    scaled_cols_df = pd.DataFrame(scaled_df, columns=scale_cols.columns, index=scale_cols.index)
    return scaled_cols_df

###############################################################################