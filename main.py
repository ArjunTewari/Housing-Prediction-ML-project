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



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns)

# Handling the categorical values
mean_target = housing.groupby('ocean_proximity')['median_income'].mean()
housing_encode = housing['ocean_proximity'].map(mean_target)

# Combining the encoded categorical column with the numerical column


# Normalizing the numerical dataframe :
# from sklearn.preprocessing import minmax_scaling
# minmax_scaling(housing_num, columns=housing_num.columns)
scaler = MinMaxScaler()

# Apply the scaler to your data
housing_num_scaled = scaler.fit_transform(housing_num)
housing_num_scaled = pd.DataFrame(housing_num_scaled, columns=housing_num.columns)

housing_prepared = pd.concat([housing_num, housing_encode], axis=1)

#Now we will create a pipeline for automating the entire process :
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

num_attributes = list(housing.drop("ocean_proximity", axis=1))
cat_attribute = ["ocean_proximity"]

# Custom transformer to select numerical or categorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


# Custom transformer to encode categorical data using mean encoding
class MeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column

    def fit(self, X, y=None):
        self.mean_target = X.groupby('ocean_proximity')[self.target_column].mean()
        return self

    def transform(self, X):
        # Perform the mean encoding
        encoded_column = X['ocean_proximity'].map(self.mean_target)
        return encoded_column.values.reshape(-1, 1)

# Numerical pipeline
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attributes)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Categorical pipeline
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(['ocean_proximity', 'median_income'])),
    ('mean_encoder', MeanEncoder(target_column="median_income"))
])

# Combine both pipelines
from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer(transformers=[
    ("num_pipeline", num_pipeline, num_attributes),
    ("cat_pipeline", cat_pipeline, ["ocean_proximity", "median_income"]),
])

housing_prepared_test = full_pipeline.fit_transform(housing_test)
# Prepare the data
housing_prepared = full_pipeline.fit_transform(housing)
# Convert to DataFrame for easier handling
housing_prepared_df = pd.DataFrame(housing_prepared, columns=num_attributes + ["ocean_proximity_encoded"])
# Display the prepared data

from sklearn.ensemble import RandomForestRegressor
tree_reg= RandomForestRegressor(random_state=42)
tree_reg.fit(housing_prepared,housing_labels)

some_data = housing_test.iloc[:5]
some_labels = housing_labels_test.iloc[:5]
some_data_prep = full_pipeline.transform(some_data)

print("predicted values :", tree_reg.predict(some_data_prep))
print("Actual values : ", list(some_labels))

from sklearn.metrics import mean_squared_error
hous_pred = tree_reg.predict(housing_prepared_test)
lin_mse = mean_squared_error(housing_labels_test, hous_pred)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)


# Fine tuning the model
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators' : randint(low = 1, high = 200),
    'max_features' : randint(low = 1, high = 8)
}

rnd_search = RandomizedSearchCV(tree_reg, param_distributions=param_dist, n_iter=10, cv = 5, scoring='neg_mean_squared_error',random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

# some_data = housing_test.iloc[:5]
# some_labels = housing_labels_test.iloc[:5]
# some_data_prep = full_pipeline.transform(some_data)
#
# print("predicted values :", tree_reg.predict(some_data_prep))
# print("Actual values : ", list(some_labels))











