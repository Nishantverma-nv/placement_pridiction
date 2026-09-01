from pyexpat import model

import numpy as np
from sklearn.linear_model import (
LinearRegression,
Ridge,
Lasso,
ElasticNet
)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.data.load_data import load_data

from src.data.preprocess import (split_data,identity_features,handle_missing_values,standardize_data,one_hot_encode_data,ordinal_encode_data)

def create_models():
    models = {
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(alpha=1.0),
        "LassoRegression": Lasso(alpha=0.01),
        "ElasticNetRegression": ElasticNet(apha=0.01, l1_ratio=0.5),
    }
    return models
def train_models(models, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def evalute_model(y_test,y_pred):
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse= mse ** 0.5
    r2 = r2_score(y_test,y_pred)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

def main():
    df = load_data()

    print("Original DataSet Shape:")
    print(df.shape)

    X_train, X_test, y_train, y_test = split_data(df,target_column="Salary Package",drop_columns=[
        "StudentID",
        "PlacementStatus",
        "IsAnomaly"
    ])
    print("Trained DataSet Shape:")
    print(X_train.shape)
    print("Test DataSet Shape:")
    print(X_test.shape)

    numerical_features, categorical_features = ( identity_features(X_train) )
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)


    one_hot_features = [
    'Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs'
    ]
    ordinal_features = [
    'CollegeTier', 'CGPA_Tier'
    ]
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    print("Missing values handled.")

    X_train, X_test, scaler = standardize_data(X_train, X_test, numerical_features)
    print("Standardization completed.")


    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    print("One-hot encoding completed.")

    # Ordinal encode ordered categorical features
    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    print("Ordinal encoding completed.")

    



