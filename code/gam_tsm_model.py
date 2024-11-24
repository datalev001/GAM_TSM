import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Define the bounded nonlinear function
def gam_function(x, a, b):
    return b * np.sqrt(np.maximum(0, 1 - a * x**2))  # Ensure no negative inputs to sqrt

# Define the GAM ARIMA loss function
def gam_arima_loss(params, data, p, q, periods, weights):
    n = len(data)
    a, b = params[0], params[1]  # AR encapsulation parameters
    a_ma, b_ma = params[2], params[3]  # MA encapsulation parameters
    a_cycle, b_cycle = params[4], params[5]  # Cycle encapsulation parameters
    ar_coeffs = params[6:6 + p]
    ma_coeffs = params[6 + p:6 + p + q]
    residuals = np.zeros(n)

    for t in range(max(p, q), n):
        # Encapsulated AR terms
        ar_part = np.sum([gam_function(data[t - i], a, b) * ar_coeffs[i - 1] for i in range(1, p + 1)])
        
        # Encapsulated MA terms
        ma_part = np.sum([gam_function(residuals[t - i], a_ma, b_ma) * ma_coeffs[i - 1] for i in range(1, q + 1)])
        
        # Encapsulated cycle terms
        cycle_part = np.sum([
            gam_function(weights[i] * np.sin(2 * np.pi * t / periods[i]), a_cycle, b_cycle)
            for i in range(len(periods))
        ])
        
        predicted = ar_part + ma_part + cycle_part
        residuals[t] = data[t] - predicted

    mse = np.mean(residuals[max(p, q):]**2)
    return mse

# GAM ARIMA fitting function
def fit_gam_arima(data, p, q, periods, weights, initial_params=None):
    if initial_params is None:
        initial_params = [0.1, 1.0, 0.1, 1.0, 0.1, 1.0] + [0.1] * (p + q)  # Add encapsulation params for MA and cycle

    result = minimize(
        gam_arima_loss,
        initial_params,
        args=(data, p, q, periods, weights),
        method='L-BFGS-B',
        bounds=[(0, 1)] * (6 + p + q)
    )
    return result

# Function to test GAM ARIMA and print coefficients
def evaluate_gam_arima(file_path, k=3, p=2, q=1, n_recent=None):
    """
    Evaluate GAM ARIMA using the most recent N data points for training.
    """
    # Load the data
    data = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')['IPG2211A2N'].values

    # Split into train and test data
    train_data, test_data = data[:-k], data[-k:]

    # If n_recent is specified, use only the most recent N data points for training
    if n_recent is not None:
        train_data = train_data[-n_recent:]

    periods = [12]  # Assume seasonality of 12 months
    weights = [0.5]  # Example weight for periodic term
    initial_params = [0.2, 1.0, 0.2, 1.0, 0.2, 1.0] + [0.1] * (p + q)

    # Normalize training data
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)
    train_data_normalized = (train_data - train_mean) / train_std

    # Fit GAM ARIMA model
    gam_model = fit_gam_arima(train_data_normalized, p, q, periods, weights, initial_params)

    # Print estimated coefficients
    print("Estimated Coefficients:")
    print(f"a (AR): {gam_model.x[0]}, b (AR): {gam_model.x[1]}")
    print(f"a_ma (MA): {gam_model.x[2]}, b_ma (MA): {gam_model.x[3]}")
    print(f"a_cycle (Cycle): {gam_model.x[4]}, b_cycle (Cycle): {gam_model.x[5]}")
    print(f"AR Coefficients: {gam_model.x[6:6 + p]}")
    print(f"MA Coefficients: {gam_model.x[6 + p:6 + p + q]}")

    # GAM ARIMA forecasting
    gam_forecast = []
    predicted_values = list(train_data_normalized[-max(p, q):])

    for i in range(k):
        ar_part = np.sum([gam_function(predicted_values[-j], gam_model.x[0], gam_model.x[1]) * gam_model.x[6 + j - 1] for j in range(1, p + 1)])
        ma_part = np.sum([gam_function(predicted_values[-j], gam_model.x[2], gam_model.x[3]) * gam_model.x[6 + p + j - 1] for j in range(1, q + 1)])
        cycle_part = np.sum([gam_function(weights[j] * np.sin(2 * np.pi * (len(predicted_values) + i) / periods[j]), gam_model.x[4], gam_model.x[5]) for j in range(len(periods))])
        
        forecast_value = ar_part + ma_part + cycle_part
        gam_forecast.append(forecast_value)
        predicted_values.append(forecast_value)

    # Transform forecast back to original scale
    gam_forecast = np.array(gam_forecast) * train_std + train_mean

    # Traditional ARMA model
    arma_model = ARIMA(train_data, order=(p, 0, q)).fit()
    arma_forecast = arma_model.forecast(steps=k)

    # Compare error measures
    actual_avg = np.mean(test_data)
    gam_forecast_avg = np.mean(gam_forecast)
    arma_forecast_avg = np.mean(arma_forecast)

    gam_rmse = np.sqrt(mean_squared_error([actual_avg], [gam_forecast_avg]))
    arma_rmse = np.sqrt(mean_squared_error([actual_avg], [arma_forecast_avg]))
    gam_mape = mean_absolute_percentage_error([actual_avg], [gam_forecast_avg])
    arma_mape = mean_absolute_percentage_error([actual_avg], [arma_forecast_avg])

    return {
        "GAM_ARIMA": {"RMSE": gam_rmse, "MAPE": gam_mape},
        "ARMA": {"RMSE": arma_rmse, "MAPE": arma_mape},
        "Actual Average": actual_avg,
        "GAM Forecast Average": gam_forecast_avg,
        "ARMA Forecast Average": arma_forecast_avg
    }

# Test 
file_path = 'Electric_Production_tm.csv'

results = evaluate_gam_arima(file_path, k=5, n_recent=200)  # Use the most recent 200 data points for training

results = evaluate_gam_arima(file_path, k=3, n_recent=270)  # Use the most recent 270 data points for training

results = evaluate_gam_arima(file_path, k=7, n_recent=320)  # Use the most recent 320 data points for training

results = evaluate_gam_arima(file_path, k=2, n_recent=380)  # Use the most recent 380 data points for training

