import pandas as pd
import streamlit as st

from utils import *

TEST_PERIOD = 365

# Read data from csv.
data = load_data('../data/streamlit_data.csv')
# Format the date
data = format_date(data)
# Add relevant temporal columns
data = add_columns(data)
# Split the data into train and test datasets
X_train, X_test, y_train, y_test = split_train_test(data, test_period=TEST_PERIOD)


# Create a page dropdown
page = st.sidebar.selectbox("""
Please select the predicting model""", ["Main Page", "Linear Regression", "XGB Regression", "LGBM Regression", "Compare Models"])

#-------------------------------------------------


if page == "Main Page":

    ### INFO
    st.title("Hello, welcome to sales predictor!")
    st.write("""

    This application predicts sales for the last year with different models
    
    # Sales drivers used in prediction: 
    
    - date: date format time feature
    - target: target variable to be predicted
      
    """)


    st.write("Let's plot sales data!")
    st.line_chart(data[["date", "target"]].set_index("date"))


#-------------------------------------------------


elif page in ["Linear Regression", "XGB Regression", "LGBM Regression"]:

    # Models

    st.title(page)
    st.write("Used features: day of week, day of month, month, week of year, season")

    # Call the model and make predictions
    predictions = model_predict(page, X_train, X_test, y_train)
    # Evaluate the model
    metric = report_metric(predictions, y_test, page)

    st.write(metric)
    

    """
    ### Real vs Pred. Plot
    """
    plot_preds(data, predictions)

#-------------------------------------------------

elif page == "Compare Models":

    # Compare models.
    st.title("Compare Models: ")

    all_metrics = pd.DataFrame({
        'Metric': ['MAPE', 'MAE', 'RMSE', 'R2']
        })
    # Calculate metrics for all models
    for model_name in ["Linear Regression", "XGB Regression", "LGBM Regression"]:
        predictions = model_predict(model_name, X_train, X_test, y_train)
        metric = report_metric(predictions, y_test, model_name)
        all_metrics[model_name] = metric[model_name].copy()
    
    st.write(all_metrics)


    #--------------------------------------------------

    # Get the best model with the least MAPE value

    # Set 'metric' column as a filter for each metric
    mape_filter = all_metrics['Metric'] == 'MAPE'
    # Find the model with the least MAPE for each metric
    min_mape_models = all_metrics.loc[mape_filter, ['Linear Regression', 'XGB Regression', 'LGBM Regression']].idxmin(axis=1)

    # Best Model
    st.title(f"Best Model : {min_mape_models[0]}")
    st.write("Let's plot the best model predictions in detail.")

    # Plot best model results.
    plot_preds(data, predictions, mode="test period", test_period=TEST_PERIOD)

link='Made by [Niez Gharbi](https://github.com/Niez-Gharbi)'
st.markdown(link, unsafe_allow_html=True)



