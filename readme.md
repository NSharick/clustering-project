# Clustering Project - Zillow Dataset

### Project Description:
- This project is a continuation of the regression project focused on analyzing the Zillow dataset. For this project the target was analyzing divers of logerror (error in the zestimate home price prediction) using visualizations, statistical testing, clustering machinging learning models and regression machine learning models.

### Project Goals:
- Goal 1: Identify features that are related to logerror using visualizations and statistical testing

- Goal 2: Identify feature relationships and feature groups using clustering machine learning models

- Goal 3: Use features and feature clusters identified during exploration to develop a regression machine learning model to predict the logerror for a property

### Initial Questions:
- Question 1: Is the square footage of the home related to the logerror for the property?

- Question 2: Is the number of bathrooms the home has related to the logerror for the property?

- Question 3: Is the number of bedrooms a home has related to the logerror for the property?

- Question 4: Is the location of the property (latitude) related to the logerror for the property?

- Question 5: Is the location of the property (longitude) related to the logerror for the property?

### ### Data Dictionary:

| Variable | Meaning |
|----------|---------|
|calculatedfinishedsquarefeet|Square footage of the home|
|bedroomcnt|Number of bedrooms in the home|
|bathroomcnt|Number of bathrooms in the home|
|fips|Federal information processing standards - for this project specifically relating to location of the property with the first two numbers representing the state and the last three numbers representing the county code. In this data set the values in the fips column are missing a leading '0' for the state code of '06' representing the state of California|
|Latitude|The latitude value for the location of the property|
|Longitude|The longitude value for the location of the property|
|Dist_lat|The distance in latitude of the property from the regions main/most popular beach|
|Dist_long|The distance in longitude of the property from the regions main/most popular beach|

### Project Planning:

- First, write a function to pull the correct dataset from the database and save it as a csv in the local directory.

- Save the data acquision function in a seperate wrangle_zillow.py file for future use

- Then, write a function that prepares the data by dealing with missing values, removing unneeded columns, and encoding categorical variables and scaling data as needed for use in machine learning models

- The data should also be split the dataset into train, validate, and test sets for modeling

- Save the data preparation functions in a wrangle_zillow.py file for later use

- Document specific questions that will be asked of the data to guide the data exploration

- Explore the data by visualizing key features related to the questions and how they relate to customer churn

- Continue to explore the data by running statistical tests to verify statistical significance of the relationships between the variables

- Document initial takeaways from the data exploration

- Explore the dataset further using clustering models to look at relationships and groupings of data within the dataset

- Document takeaways from exploration with clustering

- Develop initial regression machine learning models using the features identified in the exploration phase

- Refine those models using the train dataset by adjusting feature input and hyperparameter values

- Document the models performance on the train dataset

- Choose the three best performing models to validate using the validate dataset

- Document the performance of the models on the validate dataset

- Choose the model that performed the best and best fit the needs of the buisness question and test it using the test dataset

- Document key findings, recomendations, and next steps

### How to Reproduce this Project and Findings:

To reproduce my findings on this project you will need:

- An env file with you own credentials (hostname, username, password) to access the database and pull the zillow dataset

- The wrangle_zillow.py file in this repository that contains all the functions used to acquire, prep, split and wrangle the dataset

- The jupyter notebook in this repository named "final_report_clustering_project" which contains the code used to produce the project including the random_state identifiers to make sure and randomization of the data is consistent.

- Libraries used are numpy, pandas, seaborn, sklearn, scipy, and matplotlib. All imports are included at the top of the notebook.

### Summary:
- Goal 1 was to find home features that were related to the property's logerror. It was found that bathroom count, bedroom count, square footage of the home and the home's location in latitude and longitude were related to its logerror value

- Goal 2 was to explore feature relationships and feature groups using clustering models. It was found that there were useful data groups in bedroom count, bathroom count, and square footage as well as in property location using latitude and longitude. While visualizing clusters by the properties distance from the beach provided useful insight for further exploration since the range of logerror increased as the property got closer it did not provided additional usage for machine learning models since it was the same as the property's location when the data was scaled.

- Goal 3 was to use features and feature clusters identified during data exploration to develop a regression model for predicting the logerror for a property. While multiple models were developed with different combinations of features and hyperparameters, the best performing model was only able to perform with a root mean squared error of 0.000002 less than baseline with the validate data and only on one of the regional datasets. That same model also did not perform better than baseline with the test dataset.

### Recomendations:
- My recomendation is to attempt to gather more information on the properties in this dataset or to gather updated data that includes new properties within the same regions to continue exploring and for developing and refining predicitive models.

### Next Steps:
- If new or upodated data was obtained I would continue to explore the data loking for new relationships between features and between features and the target variable

- I would continue to develop and refine the predictive models to improve their performance

- I would further investigate the relationship between the property's distance from the beach and it's logerror and would possibly find cutoffs to appropriately bin the distance values so that they could be used in conjunction with latitude and longitude of the property to hopefully improve the model's predictive performance.

