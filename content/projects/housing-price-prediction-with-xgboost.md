---
title: "Housing Price Prediction with XGBoost"
tags: ["XGBoost", "Kaggle", "House Prices", "Regression", "Machine Learning", "Feature Engineering", "Data Cleaning", "Mutual Information"]
---
This is my [Kaggle notebook](https://www.kaggle.com/code/houmingyi/housing-price-prediction-with-xgboost) for the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition on Kaggle. The goal of this competition is to predict the sale price of homes in Ames, Iowa, based on various features of the homes.

I will be using various techniques learned in the course to predict the sale price of homes. The techniques include: preprocessing the data, feature engineering, and using different machine learning models to make predictions. I will also be using cross-validation to evaluate the performance of the models and to avoid overfitting.

## Exploratory Data Analysis (EDA)

### Load Libraries & Data


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Set plot styles
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
data_dir = Path("./house-prices-advanced-regression-techniques")
df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

```

    Train shape: (1460, 80)
    Test shape: (1459, 79)


### Understand the Target Variable


```python
# Distribution of target
sns.histplot(df_train["SalePrice"], kde=True)
plt.title("SalePrice Distribution")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.savefig("saleprice_distribution.png", bbox_inches="tight")
plt.show()

# Log-transform target to check skew
sns.histplot(np.log1p(df_train["SalePrice"]), kde=True)
plt.title("Log-Transformed SalePrice")
plt.xlabel("Log(SalePrice + 1)")
plt.ylabel("Frequency")
plt.savefig("log_saleprice_distribution.png", bbox_inches="tight")
plt.show()

# Summary stats
df_train["SalePrice"].describe()
```


    
![png](/images/housing-price-prediction-with-xgboost_files/housing-price-prediction-with-xgboost_5_0.png)
    



    
![png](/images/housing-price-prediction-with-xgboost_files/housing-price-prediction-with-xgboost_5_1.png)
    





    count      1460.000000
    mean     180921.195890
    std       79442.502883
    min       34900.000000
    25%      129975.000000
    50%      163000.000000
    75%      214000.000000
    max      755000.000000
    Name: SalePrice, dtype: float64



### Overview of the Dataset


```python
df = pd.concat([df_train, df_test], axis=0)
# df.info()
# df.describe()

# Missing value heatmap
import missingno as msno
msno.matrix(df)
```




    <Axes: >




    
![png](/images/housing-price-prediction-with-xgboost_files/housing-price-prediction-with-xgboost_7_1.png)
    


### Data Cleaning


```python
def clean_data(df):
    # Clean
    to_category = ['MSSubClass', 'MoSold', 'YrSold', 'GarageYrBlt', 'YearBuilt', 'YearRemodAdd']

    for col in to_category:
        df[col] = df[col].astype(str)

    df['Functional'] = df['Functional'].fillna('Typ')
    df["Electrical"] = df["Electrical"].fillna('SBrkr')
    df["KitchenQual"] = df["KitchenQual"].fillna('TA')

    df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])
    df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])
    df["SaleType"] = df["SaleType"].fillna(df["SaleType"].mode()[0])

    # Impute
    # Fill missing values in object columns with "None"
    objects = []
    for i in df.columns:
        if df[i].dtype == object:
            objects.append(i)
    df.update(df[objects].fillna('None'))

    # Fill missing values in numeric columns with 0
    numerics = []
    for i in df.columns:
        if df[i].dtype != object:
            numerics.append(i)
    df.update(df[numerics].fillna(0))

    return df
```


```python
def load_data():
    data_dir = Path("./house-prices-advanced-regression-techniques")
    df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
    df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")

    # Clean data
    df = pd.concat([df_train, df_test], axis=0)
    df = clean_data(df)

    # Split back into train and test
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :].drop(columns=["SalePrice"])
    return df_train, df_test
```


```python
df_train, df_test = load_data()
# Check the cleaned data
# train_missing = df_train.isnull().sum()
# print(train_missing[train_missing > 0])
# test_missing = df_test.isnull().sum()
# print(test_missing[test_missing > 0])
# df_train.info()
```

### Correlation with Target (Numerical Features)


```python
# Compute correlation matrix
corr_matrix = df_train.corr(numeric_only=True)

# Get top 15 features correlated with SalePrice
top_corr = corr_matrix["SalePrice"].abs().sort_values(ascending=False).head(15)

# Visualize
sns.heatmap(df_train[top_corr.index].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Top Correlated Features with SalePrice")
plt.savefig("top_correlated_features.png", bbox_inches="tight")
plt.show()

```


    
![png](/images/housing-price-prediction-with-xgboost_files/housing-price-prediction-with-xgboost_13_0.png)
    


### Categorical Features Preview


```python
categoricals = df_train.select_dtypes(include="object").columns
print(f"Categorical features: {len(categoricals)}")
# print(categoricals.tolist())

# Example: Visualize average SalePrice by a few important categorical features
important_cats = ["Neighborhood", "ExterQual", "GarageFinish", "KitchenQual"]

for col in important_cats:
    sns.boxplot(data=df_train, x=col, y="SalePrice")
    plt.title(f"SalePrice by {col}")
    plt.xticks(rotation=45)
    plt.savefig(f"saleprice_by_{col}.png", bbox_inches="tight")
    plt.show()

```

    Categorical features: 49



    
![png](/images/housing-price-prediction-with-xgboost_files/housing-price-prediction-with-xgboost_15_1.png)
    



    
![png](/images/housing-price-prediction-with-xgboost_files/housing-price-prediction-with-xgboost_15_2.png)
    



    
![png](/images/housing-price-prediction-with-xgboost_files/housing-price-prediction-with-xgboost_15_3.png)
    



    
![png](/images/housing-price-prediction-with-xgboost_files/housing-price-prediction-with-xgboost_15_4.png)
    


### Time-Related Patterns


```python
sns.boxplot(x="YrSold", y="SalePrice", data=df_train)
plt.title("SalePrice by Year Sold")
plt.show()

sns.scatterplot(x="YearBuilt", y="SalePrice", data=df_train)
plt.title("SalePrice vs Year Built")
plt.show()

```


    
![png](/images/housing-price-prediction-with-xgboost_files/housing-price-prediction-with-xgboost_17_0.png)
    



    
![png](/images/housing-price-prediction-with-xgboost_files/housing-price-prediction-with-xgboost_17_1.png)
    


It does not look like there is any time related feature.

## Feature Engineering


```python
X = df_train.copy()
y = X.pop("SalePrice")
# X.info()
```

### Separate Categorical and Numerical Features


```python
# cat_cols = X.select_dtypes(include=['object']).columns.tolist()
# num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
# print(f"Categorical columns: {len(cat_cols)}")
# print(f"Numerical columns: {len(num_cols)}")
```

### Encoding


```python
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder

class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, ordered_levels):
        self.ordered_levels = ordered_levels
        self.low_cardinality = []
        self.high_cardinality = []
        self.te = None

    def fit(self, X, y):
        X = X.copy()
        self.ordered_levels = {k: ["None"] + v for k, v in self.ordered_levels.items()}

        for col, order in self.ordered_levels.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=order, ordered=True).codes

        ordinal_cols = list(self.ordered_levels.keys())
        nominal_cols = [col for col in X.select_dtypes(include='object').columns if col not in ordinal_cols]

        self.low_cardinality = [col for col in nominal_cols if X[col].nunique() <= 10]
        self.high_cardinality = [col for col in nominal_cols if X[col].nunique() > 10]

        for col in self.low_cardinality:
            X[col] = X[col].astype("category").cat.codes

        self.te = TargetEncoder()
        self.te.fit(X[self.high_cardinality], y)

        return self

    def transform(self, X):
        X = X.copy()
        for col, order in self.ordered_levels.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=order, ordered=True).codes

        for col in self.low_cardinality:
            if col in X.columns:
                X[col] = X[col].astype("category").cat.codes

        if self.te:
            X[self.high_cardinality] = self.te.transform(X[self.high_cardinality])
        return X

```


```python
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(1, 11)) # 1 - 10 is the correct range!

ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}
```


```python
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

basepipe = Pipeline([
    ("encode", Encoder(ordered_levels=ordered_levels)),
    ("xgb", XGBRegressor(random_state=42))
])
```


```python
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.model_selection import cross_val_score
import numpy as np

# Custom RMSLE scorer (greater_is_better=False because lower is better)
rmsle_scorer = make_scorer(
    lambda y_true, y_pred: np.sqrt(mean_squared_log_error(np.exp(y_true), np.exp(y_pred))),
    greater_is_better=False
)
y_log = np.log(y) # We train on the log transformed target
# Then use it with cross_val_score
score = cross_val_score(basepipe, X, y_log, cv=5, scoring=rmsle_scorer)
print(f"RMSLE scores: {-score}")  # Negate to get actual RMSLE values
print(f"Mean RMSLE: {-score.mean():.5f}")
```

    RMSLE scores: [0.1397286  0.15149179 0.14633292 0.1268186  0.14509794]
    Mean RMSLE: 0.14189


### Transform Skewed Numerical Features
This transform is not used in the end. 


```python
class SkewedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        skewness = X.skew().abs()
        self.skewed_cols = skewness[skewness > self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.skewed_cols:
            X[col] = np.log1p(X[col])
        return X

skew_transformer = SkewedFeatureTransformer()

```

### Add New Features


```python
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
```


```python
def add_custom_features(df):
    df = df.copy()
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Total_sqr_footage'] = df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Total_Bathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df['Total_porch_sf'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

    df['haspool'] = (df['PoolArea'] > 0).astype(int)
    df['has2ndfloor'] = (df['2ndFlrSF'] > 0).astype(int)
    df['hasgarage'] = (df['GarageArea'] > 0).astype(int)
    df['hasbsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['hasfireplace'] = (df['Fireplaces'] > 0).astype(int)
    return df

custom_feature_step = FunctionTransformer(add_custom_features)
```

### Mutual Information
From my experiment, this seems to be the most useful tool.


```python
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Select top k features based on MI
mi_selector = SelectKBest(score_func=mutual_info_regression, k=50)  # or 'k="all"' to get scores
```

### PCA
This is not used in the end.


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)  # You can tune this
```

### Model-Based Feature Selection
Again, this is not used in the end. Mutual information already turns out to be very effective.


```python
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor

# Use a fitted model to select features with importance above threshold
model_selector = SelectFromModel(XGBRegressor(n_estimators=100), threshold="median")
```

## Model Evaluation


```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression


pipeline = Pipeline([
    ("custom_features", custom_feature_step),
    ("encoder", Encoder(ordered_levels=ordered_levels)),  
    ("feature_select", SelectKBest(score_func=mutual_info_regression, k=60)),
    ("model", XGBRegressor())  
])
```


```python
# Then use it with cross_val_score
score = cross_val_score(pipeline, X, y_log, cv=5, scoring=rmsle_scorer)
print(f"RMSLE scores: {-score}")  # Negate to get actual RMSLE values
print(f"Mean RMSLE: {-score.mean():.5f}")
```

    RMSLE scores: [0.13717768 0.15245099 0.1511411  0.12531991 0.13995213]
    Mean RMSLE: 0.14121


We can see a slight improvement compared to the baseline model above.

## Hyperparameter Tuning and Final Predictions


```python
# import optuna
# from sklearn.model_selection import cross_val_score

# def objective(trial):
#     xgb_params = dict(
#         max_depth=trial.suggest_int("max_depth", 2, 10),
#         learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
#         n_estimators=trial.suggest_int("n_estimators", 1000, 8000),
#         min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
#         colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
#         subsample=trial.suggest_float("subsample", 0.2, 1.0),
#         reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
#         reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
#     )
#     pipeline.set_params(**{f"model__{key}": val for key, val in xgb_params.items()})
#     score = cross_val_score(pipeline, X, y_log, cv=5, scoring=rmsle_scorer)
#     return -score.mean()  # Minimize RMSLE

# # Run Optuna optimization
# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=10)
```


```python
# This is the set of parameters for my submission
xgb_params = {'max_depth': 7, 
              'learning_rate': 0.004565565417769295, 
              'n_estimators': 4701, 
              'min_child_weight': 9, 
              'colsample_bytree': 0.5911157102802619, 
              'subsample': 0.265969852467484, 
              'reg_alpha': 0.030977607695995966, 
              'reg_lambda': 0.168357167207514}
```


```python
from sklearn.metrics import mean_squared_log_error

# Train the pipeline on the entire training set
X = df_train.copy()
y = X.pop('SalePrice')
y_log = np.log(y)

# Initialize the model
pipeline = Pipeline([
    ("custom_features", custom_feature_step),
    ("encoder", Encoder(ordered_levels=ordered_levels)),
    ("feature_select", SelectKBest(score_func=mutual_info_regression, k=60)),
    ("model", XGBRegressor())
])

# Properly prefix parameter names with "model__"
# best_params_prefixed = {f"model__{key}": val for key, val in study.best_params.items()}
best_params_prefixed = {f"model__{key}": val for key, val in xgb_params.items()}
# Set the best parameters to the pipeline
pipeline.set_params(**best_params_prefixed)
# Train the model
pipeline.fit(X, y_log)
```

```python
# Predict on the test set
test_predictions = pipeline.predict(df_test)
predictions = np.exp(test_predictions) 
```


```python
# Save the final result
output = pd.DataFrame({'Id': df_test.index, 'SalePrice': predictions})
output.to_csv('my_submission.csv', index=False)
```