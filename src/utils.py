import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
# from fbprophet import Prophet

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def load_data(path):
    """Load data from a local file
    Args: path (string): path to the data
    Returns: pandas dataframe
    """
    data = pd.read_csv(path, sep=',')
    return data

def format_date(df):
    """Format the date into the correct format and sorted values
    Args: df (pandas dataframe)
    Returns: pandas dataframe: dataframe with formatted date
    """
    # Change data type of date column. ( object to datetime)
    df['date'] = pd.to_datetime(df["date"], format='%Y-%m-%d')
    # Sort data by date column.
    df.sort_values(by=['date'], inplace=True)
    return df

def add_columns(df):
    """Add columns extracted from the date to help with the prediction
    Args: df (pandas dataframe)
    Returns: pandas dataframe: dataframe with added temporal columns
    """
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['season'] = (df['date'].dt.month % 12 + 3) // 3
    return df

def split_train_test(df, test_period):
    """Split the data into train and test sets based on the test periods
    Args: 
        df (pandas dataframe): data to split
        test period(integer): the testing period length
    Returns:
        X_train (pandas dataframe): Train features
        X_test (pandas dataframe): Test features
        y_train (pandas series): Train target
        y_test (pandas series): Test target
    """
    test = df[-test_period:]
    train = df[:-test_period]

    X_train = train[["day_of_week", "day_of_month", "month", "week_of_year", "season"]]
    y_train = train[["target"]]

    X_test = test[["day_of_week", "day_of_month", "month", "week_of_year", "season"]]
    y_test = test[["target"]]

    return X_train, X_test, y_train, y_test

def model_predict(model_name, X_train, X_test, y_train):
    """Call the model and make predictions
    Args:
        model_name (string): name of the model to make predictions
        X_train (pandas dataframe): Train features
        X_test (pandas dataframe): Test features
        y_train (pandas series): Train target

    Returns:
        array: predictions of the model over the test period
    """
    if model_name == "Linear Regression":
        lr = LinearRegression()
        lr.fit(X_train, y_train) # Fit the model
        lr_pred = lr.predict(X_test) # Get predictions

        return lr_pred
    
    elif model_name == "XGB Regression":
        xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
        xgb.fit(X_train, y_train) # Fit the model
        xgb_pred = xgb.predict(X_test) # Get predictions

        return xgb_pred
    
    elif model_name=="LGBM Regression":
        # fit scaler on training data
        norm = MinMaxScaler().fit(X_train)
        # transform training data
        x_train_normm3 = pd.DataFrame(norm.transform(X_train))
        # transform testing data
        x_test_normm3 = pd.DataFrame(norm.transform(X_test))

        lgb_tune = LGBMRegressor(learning_rate=0.1,
                                max_depth=2,
                                min_child_samples=25,
                                n_estimators=100,
                                num_leaves=31)
        lgb_tune.fit(x_train_normm3, y_train) # Fit the model
        lgb_pred = lgb_tune.predict(x_test_normm3) # Get predictions
        
        return lgb_pred
    
def report_metric(pred, test, model_name):
    """Function to evaluate tyhe predictions vs actual values on the testing period
    Args:
        pred (pandas series): predicted values over the test period
        test (pandas series): actual values over the test period
        model_name (string): name of the model that made the predictions
    Returns:
        pandas dataframe: metrics values based on the predictions that have been made
    """
    mape = mean_absolute_percentage_error(pred, test)
    mae = mean_absolute_error(pred, test)
    mse = mean_squared_error(pred, test)
    rmse = np.sqrt(mse)
    r2 = r2_score(test, pred)

    metric_data = {
        'Metric': ['MAPE', 'MAE', 'RMSE', 'R2'],
        model_name: [mape, mae, rmse, r2]
        }
    metric_df = pd.DataFrame(metric_data)

    return metric_df

def plot_preds(data, pred, mode="whole period", test_period=365):
    """Plot predictions along with the actual data
    Args:
        data (pandas dataframe): whole data 
        pred (pandas series): predicted values over the test period
        mode (str, optional): mode of the plot, wheteher to plot over the whole data period or only over the test period. Defaults to "whole period".
    """
    fig = plt.figure(figsize=(20,10))
    test_dates = data[-test_period:]["date"]
    if mode=="whole period":
        data_dates = data["date"]
        target = data["target"]
    elif mode=="test period":
        data_dates = data[-test_period:]["date"]
        target = data[-test_period:]["target"]
    plt.plot(data_dates, target, label = 'Actual')
    plt.plot(test_dates, pred, label = 'Predicted')
    plt.legend()
    st.pyplot(fig)