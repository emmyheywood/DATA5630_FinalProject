# Author: 
# Date: 


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def manual_train_test_split(y, train_size):
    split_point = int(len(y) * train_size)
    return y[:split_point], y[split_point:]

def DecisionTree(y_train, y_test, fh, **params):
    #Prepping the data for decision trees
    #Creating dataframe for y_train, y_test to allow for lag columns
    y_train = y_train.to_frame('data_value')
    y_test = y_test.to_frame('data_value')

    #Creating lag variables for train
    y_train['Lag_1'] = y_train['data_value'].shift(1)
    y_train['Lag_2'] = y_train['data_value'].shift(2)
    y_train['Lag_3'] = y_train['data_value'].shift(3)
    #Dropping nulls
    y_train.dropna(inplace=True)

    #Creating lag variables for test
    y_test['Lag_1'] = y_test['data_value'].shift(1)
    y_test['Lag_2'] = y_test['data_value'].shift(2)
    y_test['Lag_3'] = y_test['data_value'].shift(3)
    #Dropping nulls
    y_test.dropna(inplace=True)

    X_train = y_train[['Lag_1','Lag_2','Lag_3']]
    y_train = y_train['data_value']

    X_test = y_test[['Lag_1','Lag_2','Lag_3']]
    y_test = y_test['data_value']

    
    dt_model = DecisionTreeRegressor(**params)
    dt_model.fit(X_train, y_train)

    
    return dt_model, X_test, y_test, X_train

def random_forest_time_series(data, target_col, n_lags=3, test_size=0.2, n_estimators=100):
    """
    Train a Random Forest model for time-series prediction with customizable parameters.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the time-series data.
        target_col (str): The column name of the target variable.
        n_lags (int): Number of lagged features to create.
        test_size (float): Proportion of the data to use for testing (0 to 1).
        n_estimators (int): Number of trees in the Random Forest.

    Returns:
        model (RandomForestRegressor): Trained Random Forest model.
        y_test (pd.Series): True target values for the test set.
        y_pred (np.ndarray): Predicted target values for the test set.
    """
    # Create lagged features
    for lag in range(1, n_lags + 1):
        data[f'lag_{lag}'] = data[target_col].shift(lag)

    # Drop rows with NaN values resulting from lagging
    data = data.dropna()

    # Define predictors and target
    X = data[[f'lag_{i}' for i in range(1, n_lags + 1)]]
    y = data[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    return model, y_test, y_pred

def forecast_future(model, data, fh):
    # Initialize the list of forecasted values
    forecasted_values = []

    # Start with the last known lag values
    last_known_lags = data.iloc[-1, -3:].values  # Get the last 3 lags from the data

    for step in range(fh):
        # Convert the current lags into a DataFrame with appropriate column names
        current_lags_df = pd.DataFrame([last_known_lags], columns=[f'Lag_{i}' for i in range(1, 4)])

        # Predict the next value using the model
        next_value = model.predict(current_lags_df)[0]

        # Append the forecasted value
        forecasted_values.append(next_value)

        # Update the lag values for the next prediction
        last_known_lags = np.roll(last_known_lags, -1)  # Shift the lag values
        last_known_lags[-1] = next_value  # Replace the last lag with the predicted value

    return forecasted_values

@st.cache_data
def run_forecast(y_train, y_test, model, fh, **kwargs):
    if model == 'ETS':
        forecaster = AutoETS(**kwargs)
    elif model == 'ARIMA':
        forecaster = AutoARIMA(**kwargs)

    elif model == "Decision Tree":
        forecaster, X_test, y_test, X_train = DecisionTree(y_train, y_test, fh)
        #Keeping track of the index
        test_indices = X_test.index
        y_pred = forecaster.predict(X_test)
        #Combining the predicted values with the date index
        y_pred = pd.DataFrame({
            'Prediction': y_pred
        }, index = test_indices)
        #Getting the forecast
        last_date = y_test.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
        future_horizon = ForecastingHorizon(future_dates, is_relative=False)
        forecasts = forecast_future(forecaster, X_test,fh)
        y_forecast = pd.DataFrame({
                "Forecasts": forecasts
        }, index=future_dates)

    elif model == "Random Forest":
        forecaster, y_test, y_pred = random_forest_time_series(
        data=df_ph_clean,
        target_col='data_value',
        n_lags=2,
        test_size=0.2,
        n_estimators=150
        )

    else:
        raise ValueError("Unsupported model")
    
    if model != "Decision Tree":
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
        
        last_date = y_test.index[-1]
        future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
        future_horizon = ForecastingHorizon(future_dates, is_relative=False)
        y_forecast = forecaster.predict(fh=future_horizon)
    
    return forecaster, y_pred, y_forecast

def plot_time_series(y_train, y_test, y_pred, y_forecast, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_train.index.to_timestamp(), y_train.values, label="Train")
    ax.plot(y_test.index.to_timestamp(), y_test.values, label="Test")
    ax.plot(y_pred.index.to_timestamp(), y_pred.values, label="Test Predictions")
    ax.plot(y_forecast.index.to_timestamp(), y_forecast.values, label="Forecast")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    return fig

def plot_model_comparison(models, visible_models, test_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot test data
    ax.plot(test_data.index.to_timestamp(), 
            test_data.values, 
            label="Actual Test Data",
            color='black',
            linestyle='--')
    
    # Plot model predictions
    for model in models:
        if visible_models[model['name']]:
            ax.plot(model['predictions'].index.to_timestamp(), 
                   model['predictions'].values, 
                   label=f"{model['name']}")
    
    plt.legend()
    plt.title("Model Comparison")
    plt.xlabel("Date")
    plt.ylabel("Value")
    return fig

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'MSE': round(mse,4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE (%)': round(mape, 4)
    }

@st.cache_data
def load_and_process_data(uploaded_file, selected_freq):
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    df = df.set_index('date')
    df = df.sort_index()
    df.index = df.index.to_period(selected_freq)
    df = df.loc[df.index.notnull()]
    return df

def main():
    st.set_page_config(layout="wide")

    st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #f2f8fc;
        }
        
        /* Selectbox background */
        div[data-baseweb="select"] {
            background-color: white !important;
        }
        
        /* Selectbox border */
        div[data-baseweb="select"] > div {
            background-color: white !important;
            border-color: #cccccc !important;
        }
                    
        /* File uploader - both drag&drop and button */
        [data-testid="stFileUploader"] {
            background-color: white !important;
        }
        
        /* Number input background and buttons */
        [data-testid="stNumberInput"] > div > div > div {
            background-color: white !important;
        }
        
        /* Number input step buttons */
        [data-testid="stNumberInput"] button {
            background-color: white !important;
        }
        
        /* Number input step buttons on hover */
        [data-testid="stNumberInput"] button:hover {
            background-color: #f0f0f0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #4682B4; padding: 50px; margin-bottom: 50px; border-radius: 0px'>
        <h1 style='color: white; text-align: center; margin: 0; font-family: "Helvetica Neue", Arial, sans-serif; font-weight: 200; font-size: 70px;'>Time Series Forecasting App</h1>
    </div>
    """, unsafe_allow_html=True)
    #st.title("Time Series Forecasting App")

    tab1, tab2 = st.tabs(["Single Model Forecast", "Model Comparison"])

     # Initialize session state for model history
    if 'model_history' not in st.session_state:
        st.session_state.model_history = []
    
    with tab1:

        col1, col2, col3 = st.columns([1.5, 3.5, 5])

        with col1:
            st.header("Model Assumptions")
            model_choice = st.selectbox("Select a model", ["ETS", "ARIMA", "Decision Tree", "Random Forest"])
            train_size = st.slider("Train size (%)", 50, 95, 80) / 100

            if model_choice == "ETS":
                error = st.selectbox("Error type", ["add", "mul"])
                trend = st.selectbox("Trend type", ["add", "mul", None])
                seasonal = st.selectbox("Seasonal type", ["add", "mul", None])
                damped_trend = st.checkbox("Damped trend", value=False)
                seasonal_periods = st.number_input("Seasonal periods", min_value=1, value=1)
                model_params = {
                    "error": error,
                    "trend": trend,
                    "seasonal": seasonal,
                    "damped_trend": damped_trend,
                    "sp": seasonal_periods,
                }
            elif model_choice == "ARIMA":
                st.subheader("Non-seasonal")
                start_p = st.number_input("Min p", min_value=0, value=0)
                max_p = st.number_input("Max p", min_value=0, value=5)
                start_q = st.number_input("Min q", min_value=0, value=0)
                max_q = st.number_input("Max q", min_value=0, value=5)
                d = st.number_input("d", min_value=0, value=1)
                
                st.subheader("Seasonal")
                seasonal = st.checkbox("Seasonal", value=True)
                if seasonal:
                    start_P = st.number_input("Min P", min_value=0, value=0)
                    max_P = st.number_input("Max P", min_value=0, value=2)
                    start_Q = st.number_input("Min Q", min_value=0, value=0)
                    max_Q = st.number_input("Max Q", min_value=0, value=2)
                    D = st.number_input("D", min_value=0, value=1)
                    sp = st.number_input("Periods", min_value=1, value=12)
                
                model_params = {
                    "start_p": start_p,
                    "max_p": max_p,
                    "start_q": start_q,
                    "max_q": max_q,
                    "d": d,
                    "seasonal": seasonal,
                }
                if seasonal:
                    model_params.update({
                        "start_P": start_P,
                        "max_P": max_P,
                        "start_Q": start_Q,
                        "max_Q": max_Q,
                        "D": D,
                        "sp": sp
                    })

            elif model_choice == "Decision Tree":

                criterion = st.selectbox("Criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"])
                splitter = st.selectbox("Splitter", ["best", "random"])
                max_depth = st.number_input("Maximum Depth", min_value=1, max_value=50, value=15)
                min_samples_split = st.number_input("Minimum Samples Split", min_value=2, max_value=20, value=5)
                min_samples_leaf = st.number_input("Minimum Samples Leaf", min_value=1, max_value=10, value=2)
                max_features_type = st.selectbox("Maximum Features Type", ["sqrt", "log2", "float", None])
                if max_features_type == "float":
                    max_features = st.slider("Maximum Features (fraction)", 0.1, 1.0, 1.0, 0.1)
                else:
                    max_features = max_features_type
                random_state = st.number_input("Random State", value=42)

                model_params = {
                    "criterion":criterion,
                    "splitter":splitter,
                    "max_depth":max_depth,
                    "min_samples_split":min_samples_split,
                    "min_samples_leaf":min_samples_leaf }
                
            elif model_choice == "Random Forest":

                n_lags = st.number_input("Lags", min_value=1, value=2) 
                n_estimators = st.number_input("Number of Trees", min_value = 1, value =150)

                model_params = {
                    "n_lags": n_lags,
                    "n_estimators": n_estimators
                }

        with col2:
            st.header("Data Handling")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    # Allow user to select the frequency
                    freq_options = ['D', 'W', 'M', 'Q', 'Y']
                    selected_freq = st.selectbox("Select the data frequency", freq_options)

                    # Use cached function
                    df = load_and_process_data(uploaded_file, selected_freq)

                    st.subheader("Data Preview")
                    st.write(df.head())

                    # Filter out non-numeric columns
                    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if not numeric_columns:
                        st.error("No numeric columns found in the uploaded data. Please ensure your CSV contains numeric data for forecasting.")
                    else:
                        target_variable = st.selectbox("Select your target variable", numeric_columns)

                        # Plot the time series of the selected target variable
                        st.subheader(f"Time Series Plot: {target_variable}")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(df.index.to_timestamp(), df[target_variable])
                        plt.title(f"{target_variable} Time Series")
                        plt.xlabel("Date")
                        plt.ylabel("Value")
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"An error occurred while processing the file: {str(e)}")
                    st.error("Please ensure your CSV file is properly formatted with a date column and numeric data for forecasting.")

        with col3:
            st.header("Forecast Results")
            fh = st.number_input("Number of periods to forecast", min_value=1, value=10)
            run_forecast_button = st.button("Run Forecast")
            
            if run_forecast_button:
                if 'df' in locals() and 'target_variable' in locals():
                    try:
                        y = df[target_variable]
                        y_train, y_test = manual_train_test_split(y, train_size)
                        forecaster, y_pred, y_forecast = run_forecast(y_train, y_test, model_choice, fh, **model_params)
                        
                        # Store model results in session state
                        model_results = {
                            'name': f"{model_choice} Model {len(st.session_state.model_history) + 1}",
                            'params': model_params,
                            'predictions': y_pred,
                            'forecast': y_forecast,
                             'test_data': y_test
                        }
                        
                        if len(st.session_state.model_history) >= 4:
                            st.session_state.model_history.pop(0)
                        st.session_state.model_history.append(model_results)
                        
                        # Display single model results
                        fig = plot_time_series(y_train, y_test, y_pred, y_forecast, 
                                            f"{model_choice} Forecast for {target_variable}")
                        st.pyplot(fig)
                        
                    # Create two columns for predictions and forecast
                        pred_col, forecast_col = st.columns(2)
                        
                        with pred_col:
                            st.subheader("Test Set Predictions")
                            st.write(y_pred)
                        
                        with forecast_col:
                            st.subheader("Future Forecast Values")
                            st.write(y_forecast)                        
                        # st.subheader("Test Set Predictions")
                        # st.write(y_pred)
                        
                        # st.subheader("Future Forecast Values")
                        # st.write(y_forecast)
                    except Exception as e:
                        st.error(f"An error occurred during forecasting: {str(e)}")
                else:
                    st.warning("Please upload data and select a target variable before running the forecast.")

    with tab2:
        st.header("Model Comparison")

        # Add button to clear all models
        if st.button("Clear All Models"):
            st.session_state.model_history = []
        
        col_select, col_plot = st.columns([1, 3])
        
        with col_select:
            st.subheader("Select Models to Display")
            visible_models = {}
            models_to_remove = []
            
            for i, model in enumerate(st.session_state.model_history):
                col1, col2 = st.columns([3, 1])
                with col1:
                    try:
                        visible_models[model['name']] = st.checkbox(model['name'], value=True)
                    except Exception as e:
                        st.error(f"Error with {model['name']}: {str(e)}")
                        models_to_remove.append(i)
                with col2:
                    if st.button("Remove", key=f"remove_{i}"):
                        models_to_remove.append(i)

        with col_plot:
            if st.session_state.model_history:
                # Get test data from the latest model run
                y_test = st.session_state.model_history[-1]['test_data']
                fig = plot_model_comparison(st.session_state.model_history, visible_models, y_test)
                st.pyplot(fig)
                
                # Create metrics table
                st.subheader("Model Performance Metrics")
                metrics_data = []
                for model in st.session_state.model_history:
                    metrics = calculate_metrics(y_test, model['predictions'])
                    metrics['Model'] = model['name']
                    metrics_data.append(metrics)
                
                # Convert to DataFrame and display
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df = metrics_df.set_index('Model')
                st.dataframe(metrics_df)
            else:
                st.info("Run some models in the 'Single Model Forecast' tab to compare them here.")


if __name__ == "__main__":
    main()
