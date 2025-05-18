import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

# Set page title
st.title("EcoMove Prediction Dashboard")

# Load models
# Ridge Regression models
with open("models/bev_kt_model.pkl", "rb") as file:
    bev_kt_ridge_model = pickle.load(file)

with open("models/motor_mt_model.pkl", "rb") as file:
    motor_mt_ridge_model = pickle.load(file)

# ARIMA models
with open("models/bev_kt_arima_model.pkl", "rb") as file:
    bev_kt_arima_model = pickle.load(file)

with open("models/motor_mt_arima_model.pkl", "rb") as file:
    motor_mt_arima_model = pickle.load(file)

# Load classification model (if still needed)
#with open("models/graphs_model.pkl", "rb") as file:
 #   graphs_model = pickle.load(file)

# Section 1: Battery Electric Vehicles (bev_kt)
st.header("Battery Electric Vehicles (bev_kt)")

# Load historical data for BEVs
bev_data = pd.read_csv("dataset/ts.csv")

# Ridge Regression: Actual vs Predicted (2010-2020) and Predicted (2021-2030)
st.subheader("Ridge Regression: Actual vs Predicted Emissions (2010-2020) and Predicted (2021-2030)")

# Prepare historical data (2010-2020) for prediction
historical_bev = bev_data[bev_data['year'].between(2010, 2020)]
X_historical_bev = historical_bev[['BEV_sales', 'year']]
y_actual_bev = historical_bev['bev_kt']
y_predicted_bev = bev_kt_ridge_model.predict(X_historical_bev)

# Prepare future data (2021-2030) for prediction
future_years_bev = pd.DataFrame({
    'year': range(2021, 2031),
    'BEV_sales': np.linspace(historical_bev['BEV_sales'].mean(), historical_bev['BEV_sales'].mean() * 1.5, 10)  # Assume linear growth
})
X_future_bev = future_years_bev[['BEV_sales', 'year']]
y_future_bev = bev_kt_ridge_model.predict(X_future_bev)

# Combine data for plotting
ridge_data_bev = pd.concat([
    pd.DataFrame({
        'year': historical_bev['year'],
        'bev_kt': y_actual_bev,
        'Type': 'Actual'
    }),
    pd.DataFrame({
        'year': historical_bev['year'],
        'bev_kt': y_predicted_bev,
        'Type': 'Predicted (2010-2020)'
    }),
    pd.DataFrame({
        'year': future_years_bev['year'],
        'bev_kt': y_future_bev,
        'Type': 'Predicted (2021-2030)'
    })
])

# Animated Plotly graph for Ridge Regression
fig1 = px.line(ridge_data_bev, x='year', y='bev_kt', color='Type',
               title="BEV Emissions: Ridge Regression (Actual vs Predicted)",
               animation_frame='Year', animation_group='Type',
               range_y=[ridge_data_bev['bev_kt'].min() * 0.9, ridge_data_bev['bev_kt'].max() * 1.1])
fig1.update_layout(transition={'duration': 1000})  # Smooth transition for animation
st.plotly_chart(fig1)

# ARIMA Forecast (2021-2030)
st.subheader("ARIMA Forecast: Emissions (2021-2030)")

# Use historical data up to 2020 for ARIMA context
historical_data_bev = bev_data[bev_data['Year'] <= 2020].set_index('Year')['bev_kt']
# Forecast from 2021 to 2030
forecast_steps = 10
forecast_bev = bev_kt_arima_model.forecast(steps=forecast_steps)
forecast_index_bev = pd.date_range(start='2021-01-01', periods=forecast_steps, freq='Y')

# Combine historical (up to 2020) and forecast data for plotting
arima_data_bev = pd.concat([
    pd.DataFrame({
        'year': historical_data_bev.index,
        'bev_kt': historical_data_bev.values,
        'Type': 'Historical'
    }),
    pd.DataFrame({
        'year': forecast_index_bev.year,
        'bev_kt': forecast_bev.values,
        'Type': 'Forecast (2021-2030)'
    })
])

# Animated Plotly graph for ARIMA
fig2 = px.line(arima_data_bev, x='year', y='bev_kt', color='Type',
               title="BEV Emissions: ARIMA Forecast (2021-2030)",
               animation_frame='Year', animation_group='Type',
               range_y=[arima_data_bev['bev_kt'].min() * 0.9, arima_data_bev['bev_kt'].max() * 1.1])
fig2.update_layout(transition={'duration': 1000})  # Smooth transition for animation
st.plotly_chart(fig2)

# Section 2: Motor Cars (motor_mt)
st.header("Motor Cars (motor_mt)")

# Load historical data for motor cars
motor_data = pd.read_csv("dataset/ts.csv")

# Ridge Regression: Actual vs Predicted (2010-2020) and Predicted (2021-2030)
st.subheader("Ridge Regression: Actual vs Predicted Emissions (2010-2020) and Predicted (2021-2030)")

# Prepare historical data (2010-2020) for prediction
historical_motor = motor_data[motor_data['year'].between(2010, 2020)]
X_historical_motor = historical_motor[['total_motor_cars_production', 'year']]
y_actual_motor = historical_motor['motor_mt']
y_predicted_motor = motor_mt_ridge_model.predict(X_historical_motor)

# Prepare future data (2021-2030) for prediction
future_years_motor = pd.DataFrame({
    'Year': range(2021, 2031),
    'total_motor_cars_production': np.linspace(historical_motor['total_motor_cars_production'].mean(), historical_motor['total_motor_cars_production'].mean() * 1.2, 10)  # Assume growth
})
X_future_motor = future_years_motor[['total_motor_cars_production', 'Year']]
y_future_motor = motor_mt_ridge_model.predict(X_future_motor)

# Combine data for plotting
ridge_data_motor = pd.concat([
    pd.DataFrame({
        'year': historical_motor['year'],
        'bev_kt': y_actual_motor,
        'Type': 'Actual'
    }),
    pd.DataFrame({
        'year': historical_motor['year'],
        'bev_kt': y_predicted_motor,
        'Type': 'Predicted (2010-2020)'
    }),
    pd.DataFrame({
        'year': future_years_motor['year'],
        'bev_kt': y_future_motor,
        'Type': 'Predicted (2021-2030)'
    })
])

# Animated Plotly graph for Ridge Regression
fig3 = px.line(ridge_data_motor, x='year', y='bev_kt', color='Type',
               title="Motor Cars Emissions: Ridge Regression (Actual vs Predicted)",
               animation_frame='year', animation_group='Type',
               range_y=[ridge_data_motor['bev_kt'].min() * 0.9, ridge_data_motor['bev_kt'].max() * 1.1])
fig3.update_layout(transition={'duration': 1000})  # Smooth transition for animation
st.plotly_chart(fig3)

# ARIMA Forecast (2021-2030)
st.subheader("ARIMA Forecast: Emissions (2021-2030)")

# Use historical data up to 2020 for ARIMA context
historical_data_motor = motor_data[motor_data['year'] <= 2020].set_index('year')['motor_mt']
# Forecast from 2021 to 2030
forecast_motor = motor_mt_arima_model.forecast(steps=forecast_steps)
forecast_index_motor = pd.date_range(start='2021-01-01', periods=forecast_steps, freq='Y')

# Combine historical (up to 2020) and forecast data for plotting
arima_data_motor = pd.concat([
    pd.DataFrame({
        'year': historical_data_motor.index,
        'bev_kt': historical_data_motor.values,
        'Type': 'Historical'
    }),
    pd.DataFrame({
        'year': forecast_index_motor.year,
        'bev_kt': forecast_motor.values,
        'Type': 'Forecast (2021-2030)'
    })
])

# Animated Plotly graph for ARIMA
fig4 = px.line(arima_data_motor, x='year', y='bev_kt', color='Type',
               title="Motor Cars Emissions: ARIMA Forecast (2021-2030)",
               animation_frame='year', animation_group='Type',
               range_y=[arima_data_motor['bev_kt'].min() * 0.9, arima_data_motor['bev_kt'].max() * 1.1])
fig4.update_layout(transition={'duration': 1000})  # Smooth transition for animation
st.plotly_chart(fig4)

# (Remaining sections like Classification, Renewable Energy, and Aged Vehicles can remain unchanged)