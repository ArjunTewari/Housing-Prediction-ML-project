import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Gathering the data :
Housing_California = pd.read_csv('C:/Users/hp/OneDrive/Desktop/Learning/ML_PROJECT/Housing_data/housing1.csv')
# print(Housing_California.head())

# Know your data : This will give you all the columns.
#print(Housing_California.info())

# Removing the missing data points :
for x in ['total_rooms']:
    Housing_California.dropna(subset=[x], how='all', inplace=True)

# Ploting the histograms :
# plt.hist(Housing_California['housing_median_age'],50)
# plt.show()

#Splitting the data set :
np.random.seed(42)

# def training_test_split(data,test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(test_ratio * len(data))
#     test_indices = shuffled_indices[:test_set_size]
#     training_indices = shuffled_indices[test_set_size:]
#     return data.iloc[test_indices],data.iloc[training_indices]


a, b = train_test_split(Housing_California, test_size=0.20, random_state=42)

# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.hist(a['housing_median_age'], bins=50, color='blue', alpha=0.7)
# plt.title('Training Set: Housing Median Age')
# plt.xlabel('Housing Median Age')
# plt.ylabel('Frequency')
#
# plt.subplot(1, 2, 2)
# plt.hist(b['housing_median_age'], bins=50, color='green', alpha=0.7)
# plt.title('Test Set: Housing Median Age')
# plt.xlabel('Housing Median Age')
# plt.ylabel('Frequency')
#
# plt.tight_layout()
# plt.show()


#Trying stratification
Housing_California["median_income"] /= 1.5
Housing_California["income_cat"] = pd.cut(Housing_California["median_income"], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                          labels=[1, 2, 3, 4, 5])

from sklearn.model_selection import StratifiedShuffleSplit

sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train, test in sp.split(Housing_California, Housing_California["income_cat"]):
    strat_train = Housing_California.loc[train]
    strat_test = Housing_California.loc[test]

#Removing the income cat -
for set in (strat_train, strat_test):
    set.drop(["income_cat"], axis=1, inplace=True)
# Option - 2 -
#strat_train, strat_test = [df.drop("income_cat", axis=1) for df in (strat_train, strat_test)]

housing = strat_train.copy()


# Plotting the graph -
# housing.info()
# housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4, label="Population", c ='median_house_value', cmap = plt.get_cmap("jet"), s = housing["population"]/100, sharex=False)
# plt.legend()
# plt.show()

#Finding correlation -
# corr_matrix = housing.select_dtypes(include=['number']).corr()
# corr_matrix["median_house_value"].sort_values(ascending = False)
# print(corr_matrix["median_house_value"].sort_values(ascending = False))

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

# corr_matrix = housing.select_dtypes(include=['number']).corr()
# corr_matrix["median_house_value"].sort_values(ascending = False)
# print(corr_matrix["median_house_value"].sort_values(ascending = False))

#Now we will drop the target value :
housing = strat_train.drop("median_house_value", axis=1)
housing_test = strat_test.drop("median_house_value", axis=1)
housing_labels = strat_train["median_house_value"].copy()
housing_labels_test = strat_test["median_house_value"].copy()

housing_num = housing.drop("ocean_proximity", axis=1)


#Now we will create a pipeline for automating the entire process :
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Separate numerical and categorical columns
num_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attributes = ["ocean_proximity"]

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder())
])

# Combine pipelines using ColumnTransformer
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes),
])

# Display the prepared data
housing_prepared = full_pipeline.fit_transform(housing)

from sklearn.ensemble import RandomForestRegressor

some_data = housing_test.iloc[:5]
some_labels = housing_labels_test.iloc[:5]
some_data_prep = full_pipeline.transform(some_data)

# Fine tuning the model
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# Define the base estimators
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('ridge', Ridge())
]

st = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
st.fit(housing_prepared, housing_labels)

print("predicted values :", st.predict(some_data_prep))
print("Actual values : ", list(some_labels))

hous_pred = st.predict(housing_prepared)
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(housing_labels, hous_pred)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)











