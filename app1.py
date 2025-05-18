import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress non-critical warnings
import matplotlib.animation as animation
import plotly.express as px


st.set_page_config(page_title="Vehicle ML Dashboard", layout="wide")
# Set custom CSS to change background color
# Force background color
background_color = "#F0F2F6"  # Light gray
custom_css = f"""
    <style>
        body, .stApp, [data-testid="stAppViewContainer"] {{
            background-color: {background_color} !important;
            color: #000000 !important;
        }}
        .stSidebar, [data-testid="stSidebar"] {{
            background-color: {background_color} !important;
        }}
        h1, h2, h3, p, div, span, .stMarkdown, .stText {{
            color: #000000 !important;
        }}
        .stDataFrame, [data-testid="stTable"] {{
            background-color: #FFFFFF !important;
        }}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
# Set page config
# Set Matplotlib style to ensure light backgrounds
plt.style.use('default')  # Reset to default Matplotlib style
sns.set_style("whitegrid")  # White background with grid
# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Motor Cars", 
    "Battery Cars", 
    "AQI Category", 
    "Renewable Energy Visualizations", 
    "Aged Vehicles Comparison"
])
data = pd.read_csv('dataset/ts.csv')

# Home Page
if page == "Home":
    st.title("Vehicle Machine Learning Dashboard")
    st.write("Explore predictions and visualizations for motor cars, battery electric vehicles, AQI classification, renewable energy, and aged vehicles.")
    st.write("Use the sidebar to navigate to different sections.")
    st.markdown("[Visit Front-End Website](../frontend/index.html)")

# Motor Cars Regression
elif page == "Battery Cars":
    st.title("Battery Cars Production and Emissions")
    
    
    
    # Load models and preprocessors
    #reg_model = joblib.load('models/motor_mt_model.pkl')
    #scaler = joblib.load('models/motor_mt_scaler.pkl')
    #poly = joblib.load('models/motor_mt_poly.pkl')  # Load PolynomialFeatures

    


    # User input for regression prediction
    st.subheader("Prediction of CO2 produced from Battery")
    X = data[['year', 'BEV_sales']]
    y = data['bev_kt']
    
    # Log-transform bev_sales to reduce skewness
    X['bev_sales_log'] = np.log1p(X['BEV_sales'])

    
    # Create year_offset feature
    X['year_offset'] = X['year'] - 2010
    # Features to use: year_offset and bev_sales_log
    X_final = X[['year_offset', 'bev_sales_log']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    # Save preprocessed data
    preprocessed_data = {'X_scaled': X_scaled, 'y': y, 'scaler': scaler, 'X_final': X_final}
    X_scaled = preprocessed_data['X_scaled']
    y = preprocessed_data['y']
    scaler = preprocessed_data['scaler']
    X_final = preprocessed_data['X_final']
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_final_poly = poly.fit_transform(X_final)
    X_scaled_poly = scaler.fit_transform(X_final_poly)
    
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    ridge_model_poly = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    ridge_model_poly.fit(X_scaled_poly, y)
    best_ridge_poly = ridge_model_poly.best_estimator_
    y_pred_train_poly = np.maximum(best_ridge_poly.predict(X_scaled_poly), 0)
    
    actual_pred_df = pd.DataFrame({
    'Year': data['year'],
    'BEV Sales': data['BEV_sales'],
    'Actual bev_kt': y,
    'Predicted bev_kt': y_pred_train_poly
    })
    st.write(actual_pred_df)
    
    fig2  = plt.figure(figsize=(10, 6))
    plt.scatter(data['year'], y, color='green', label='Actual bev_kt', s=100, alpha=0.8)
    plt.plot(data['year'], y_pred_train_poly, marker='o', color='red', label='Predicted bev_kt', linewidth=2)
    plt.title('Actual vs. Predicted CO2 produced from Battery (2010–2020)')
    plt.xlabel('Year')
    plt.ylabel('CO2 produced from Battery')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    st.pyplot(fig2)





    #PREDICTION FROM 2021-2030
    # Create future data for 2021–2030
    years_future = np.arange(2021, 2031)
    # Assume 10% exponential growth for BEV Sales from 2,000,000 (2020)
    bev_sales_2020 = 2000000
    bev_sales_future = bev_sales_2020 * (1 + 0.1) ** (years_future - 2020)

    # Replace with specific BEV Sales if available, e.g.:
    # bev_sales_future = [2200000, 2500000, 2900000, 3400000, 4000000, 4700000, 5500000, 6400000, 7400000, 8500000]

    # Prepare future data
    future_data = pd.DataFrame({
        'year': years_future,
        'BEV_sales': bev_sales_future
    })
    future_data['year_offset'] = future_data['year'] - 2010
    future_data['bev_sales_log'] = np.log1p(future_data['BEV_sales'])
    future_data_poly = poly.transform(future_data[['year_offset', 'bev_sales_log']])
    future_data_scaled_poly = scaler.transform(future_data_poly)

    # Predict bev_kt for 2021–2030 and clip to avoid negatives
    y_pred_future_poly = np.maximum(best_ridge_poly.predict(future_data_scaled_poly), 0)

    # Create table of predictions for 2021–2030
    predictions_df = pd.DataFrame({
        'Year': years_future,
        'BEV Sales': bev_sales_future,
        'Predicted bev_kt': y_pred_future_poly
    })
    st.write("\nTable: Predicted CO2 Emissions from Battery for 2021–2030")
    st.write(predictions_df)

    # Plot 2: Combined Actual (2010–2020) and Predicted (2021–2030)
    fig3  = plt.figure(figsize=(10, 6))
    # Actual (2010–2020)
    plt.scatter(data['year'], y, color='green', label='Actual bev_kt', s=100, alpha=0.8)
    # Predicted (2021–2030)
    plt.plot(years_future, y_pred_future_poly, marker='o', color='blue', label='Predicted bev_kt', linewidth=2)
    plt.axvline(x=2020, color='gray', linestyle='--', label='2020 (Transition)', alpha=0.7)
    plt.title('CO2 produced from Battery: Actual (2010–2020) and Predicted (2021–2030)')
    plt.xlabel('Year')
    plt.ylabel('CO2 produced from Battery')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    st.pyplot(fig3)

    # ARIMA forecast
    st.subheader("ARIMA Forecast (2021-2035)")
    # Set 'year' as the index for ARIMA (optional, but helps with time series)
    data['year'] = pd.to_datetime(data['year'], format='%Y')  # Convert year to datetime
    data.set_index('year', inplace=True)

    # Fit ARIMA model (p=1, d=1, q=1 as a starting point)
    model = ARIMA(data['bev_kt'], order=(1, 1, 1))
    model_fit = model.fit()

    # Summary of the model
    #print(model_fit.summary())

    # Forecast next 15 years (2021-2035)
    forecast_steps = 15
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create a date range for the forecast (assuming last year in data is 2020)
    last_year = data.index[-1].year
    forecast_years = pd.date_range(start=f'{last_year + 1}-01-01', periods=forecast_steps, freq='YS')
    forecast_series = pd.Series(forecast, index=forecast_years)

    # Print forecast with years
    st.write("\nForecasted CO2 Produced from Battery (2021-2035):")
    st.write(forecast_series)

    # Plot historical and forecast data
    fig1  = plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['bev_kt'], label='Historical')
    plt.plot(forecast_series.index, forecast_series, label='Forecast', linestyle='--')
    plt.legend()
    plt.title('CO2 Produced from Battery Forecast')
    plt.xlabel('Year')
    plt.ylabel('CO2 Produced from Battery')
    plt.grid(True)
    plt.show()
    st.pyplot(fig1)

 
 
 
 
 
    
elif page == "Motor Cars":
    st.title("Motor Cars Production and Emissions") 
    
    # 1. Load Data
    X = data[['year', 'total_motor_cars_production']]
    y = data['motor_mt']

    # 2. Add Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)  # Degree=2 for quadratic terms
    X_poly = poly.fit_transform(X)
    poly_feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

    # 3. Scale Features
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly_df)
    X_poly_scaled_df = pd.DataFrame(X_poly_scaled, columns=poly_feature_names)

    # 4. Train Ridge Model
    ridge_model = Ridge(alpha=0.1, random_state=42)
    ridge_model.fit(X_poly_scaled_df, y)
    predictions = ridge_model.predict(X_poly_scaled_df)

    

    # 6. Create Table for Actual vs Predicted
    results_2010_2020 = pd.DataFrame({
        'Year': data['year'],
        'Total Motor Cars Production': data['total_motor_cars_production'],
        'Actual motor_mt': y,
        'Predicted motor_mt': predictions
    })
    st.write(results_2010_2020)

    
    # 7. Plot Actual vs Predicted
    fig3 = plt.figure(figsize=(10, 5))
    plt.scatter(data['year'], y, color='blue', label='Actual motor_mt', s=100)
    plt.plot(data['year'], predictions, color='red', marker='o', linestyle='--', label='Predicted motor_mt')
    plt.xlabel('Year')
    plt.ylabel('CO2 produced from Motor')
    plt.title('Actual vs Predicted CO2 Emissions from Motor (2010–2020)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig3)
    
    
    
    
    # 1. Load and Preprocess Data

    # 2. Polynomial Features and Scaling
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly_df)
    X_poly_scaled_df = pd.DataFrame(X_poly_scaled, columns=poly_feature_names)

    # 3. Train Ridge Model
    ridge_model = Ridge(alpha=0.1, random_state=42)
    ridge_model.fit(X_poly_scaled_df, y)
    predictions = ridge_model.predict(X_poly_scaled_df)

    # 4. Forecast Total Motor Cars Production with Linear Regression
    lr = LinearRegression()
    years = X['year'].values.reshape(-1, 1)
    production = X['total_motor_cars_production'].values
    lr.fit(years, production)

    # Generate future years (2021–2030)
    future_years = np.arange(2021, 2031).reshape(-1, 1)
    future_production = lr.predict(future_years)

    # Create future data DataFrame
    future_data = pd.DataFrame({
        'year': future_years.flatten(),
        'total_motor_cars_production': future_production
    })

    # 5. Apply Polynomial Features and Scale for Future Data
    future_poly = poly.transform(future_data)
    future_poly_df = pd.DataFrame(future_poly, columns=poly_feature_names)
    future_poly_scaled = scaler.transform(future_poly_df)
    future_poly_scaled_df = pd.DataFrame(future_poly_scaled, columns=poly_feature_names)

    # 6. Predict motor_kt for 2021–2030
    future_predictions = ridge_model.predict(future_poly_scaled_df)

    # 7. Table for 2021–2030 Predictions
    future_results = pd.DataFrame({
        'Year': future_data['year'],
        'Total Motor Cars Production': future_data['total_motor_cars_production'],
        'Predicted motor_mt': future_predictions
    })
    st.write("\nPredictions of CO2 Emissions from Motor for 2021–2030:")
    st.write(future_results)

    
    # 8. Combine Data for Plotting
    all_years = np.concatenate([X['year'], future_data['year']])
    all_predictions = np.concatenate([predictions, future_predictions])
    all_data = pd.DataFrame({
        'year': all_years,
        'motor_mt': np.concatenate([y, [np.nan] * len(future_years)]),
        'predicted_motor_kt': all_predictions
    })

    # 9. Enhanced Plot for 2010–2030
    fig4 = plt.figure(figsize=(14, 8))
    # Plot actual data (2010–2020)
    plt.scatter(all_data['year'][:len(y)], all_data['motor_mt'][:len(y)], 
                color='blue', label='Actual motor_mt', s=150, zorder=5)
    # Plot predictions (2010–2030)
    plt.plot(all_data['year'], all_data['predicted_motor_kt'], 
            color='red', marker='o', linestyle='-', linewidth=3, markersize=12, 
            label='Predicted motor_mt', zorder=4)
    # Highlight forecast period
    plt.axvspan(2020.5, 2030.5, color='gray', alpha=0.2, label='Forecast Period', zorder=1)
    # Zoom y-axis to emphasize fluctuations
    y_min = min(future_predictions.min(), y.min()) - 5
    y_max = max(future_predictions.max(), y.max()) + 5
    plt.ylim(y_min, y_max)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('CO2 Emissions from Motor', fontsize=14)
    plt.title('Predictions (2010–2030) of CO2 Emissions from Motor', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', alpha=0.4)
    plt.xticks(np.arange(2010, 2031, 1), rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig4)
    
    
    
    
    #time series
    # Set 'year' as the index for ARIMA (optional, but helps with time series)
    data['year'] = pd.to_datetime(data['year'], format='%Y')  # Convert year to datetime
    data.set_index('year', inplace=True)

    # Fit ARIMA model (p=1, d=1, q=1 as a starting point)
    model = ARIMA(data['motor_mt'], order=(1, 1, 1))
    model_fit = model.fit()

    # Summary of the model
    print(model_fit.summary())

    # Forecast next 15 years (2021-2035)
    forecast_steps = 15
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create a date range for the forecast (assuming last year in data is 2020)
    last_year = data.index[-1].year
    forecast_years = pd.date_range(start=f'{last_year + 1}-01-01', periods=forecast_steps, freq='YS')
    forecast_series = pd.Series(forecast, index=forecast_years)

    # Print forecast with years
    st.write("\nForecasted CO2 Emissions from Motor (2021-2035):")
    st.write(forecast_series)


    # Plot historical and forecast data
    fig5 = plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['motor_mt'], label='Historical')
    plt.plot(forecast_series.index, forecast_series, label='Forecast', linestyle='--')
    plt.legend()
    plt.title('CO2 Emissions from Motor Forecast')
    plt.xlabel('Year')
    plt.ylabel('CO2 emission from Motor')
    plt.grid(True)
    plt.show()
    st.pyplot(fig5)
    
    
elif page == "Renewable Energy Visualizations":
    st.title("Motor Cars Production and Emissions") 
    #renewable energy
    import streamlit as st
    import pandas as pd

    # Streamlit app setup
    
    st.title("Renewable Energy Share Over Time for India")
    st.markdown("This dashboard shows the share of renewable energy in India over time.")

    # Load the dataset
    df = pd.read_csv("dataset/country_comparison.csv")

    # Select specific countries
    selected_countries = ["India"]
    df = df[df['Country'].isin(selected_countries)]

    # Prepare data for st.bar_chart
    # Ensure 'Year' and 'Renewable Energy Share (%)' are in the DataFrame
    chart_data = df[['Year', 'Renewable Energy Share (%)']].set_index('Year')

    # Create the bar chart
    st.bar_chart(
        data=chart_data,
        x=None,  # Use index (Year) as x-axis
        y='Renewable Energy Share (%)',
        x_label='Year',
        y_label='Renewable Energy Share (%)',
        use_container_width=True
    )

    # Display the data table (optional)
    #st.subheader("Data Table")
    #st.dataframe(df[['Year', 'Renewable Energy Share (%)']].style.format({"Renewable Energy Share (%)": "{:.1f}"}))
    
    
    import streamlit as st
    import pandas as pd

    # Streamlit app setup
    st.title("Renewable Energy Share Over Time for India and USA")
    st.markdown("This dashboard shows the share of renewable energy for India and the USA over time.")

    # Load the dataset
    df = pd.read_csv("dataset/country_comparison.csv")

    # Filter for specific countries
    countries_to_plot = ["India", "USA"]
    filtered_df = df[df['Country'].isin(countries_to_plot)]

    # Pivot the DataFrame for st.line_chart
    chart_data = filtered_df.pivot(index='Year', columns='Country', values='Renewable Energy Share (%)')

    # Create the line chart
    st.line_chart(
        data=chart_data,
        x=None,  # Use index (Year) as x-axis
        y=countries_to_plot,  # Columns for India and USA
        x_label='Year',
        y_label='Renewable Energy Share (%)',
        use_container_width=True
    )

    # Display the data table (optional)
    #st.subheader("Data Table")
    #st.dataframe(filtered_df[['Country', 'Year', 'Renewable Energy Share (%)']].style.format({"Renewable Energy Share (%)": "{:.1f}"}))
    
    
    
    
    
    
elif page == "AQI Category":
    st.title("AQI Category")

    # --- AQI Calculation Functions ---
    # --- AQI Calculation Functions ---
    def nox_gkm_to_ugm3(nox_gkm, speed, fuel_type='petrol'):
        distance_km = speed
        emission_g = nox_gkm * distance_km
        emission_ug = emission_g * 1e6
        air_volume_m3 = 1000000 if fuel_type == 'diesel' else 900000  
        nox_ugm3 = emission_ug / air_volume_m3
        return nox_ugm3

    def calculate_nox_aqi(nox_emissions):
        if nox_emissions <= 40:
            return (50 / 40) * nox_emissions
        elif nox_emissions <= 80:
            return ((100 - 51) / (80 - 41)) * (nox_emissions - 41) + 51
        elif nox_emissions <= 180:
            return ((200 - 101) / (180 - 81)) * (nox_emissions - 81) + 101
        elif nox_emissions <= 280:
            return ((300 - 201) / (280 - 181)) * (nox_emissions - 181) + 201
        elif nox_emissions <= 400:
            return ((400 - 301) / (400 - 281)) * (nox_emissions - 281) + 301
        else:
            return ((nox_emissions - 401) / (800 - 401)) * (nox_emissions - 401) + 401

    def get_aqi_category(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        elif aqi <= 500:
            return "Hazardous"
        else:
            return "Hazardous"

    def predict_nox_emissions(user_input, model, scaler, features, numerical_cols):
        input_df = pd.DataFrame([user_input], columns=['Engine Size', 'Age of Vehicle', 'Temperature',
                                                    'Wind Speed', 'Speed', 'Traffic Conditions', 'Road Type'])
        input_df = pd.get_dummies(input_df, columns=['Traffic Conditions', 'Road Type'], drop_first=True)
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[features]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        nox_emission = model.predict(input_df)[0]
        return max(0, nox_emission)


   # --- Streamlit App ---
    st.title("NOx Emission and AQI Prediction for Vehicles")

    # Sidebar for input parameters
    with st.sidebar:
        st.header("Input Vehicle Parameters")
        st.markdown("**Maximum Allowable Input Values:**")
        st.write("- **Engine Size**: 6.0 liters")
        st.write("- **Age of Vehicle**: 30 years")
        st.write("- **Temperature**: 40°C")
        st.write("- **Vehicle Speed**: 120 km/h")
        st.write("- **Wind Speed**: 30 km/h")
        st.markdown("---")
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Electric", "Diesel"])
        engine_size = st.number_input("Engine Size (liters)", min_value=0.0, max_value=6.0, value=2.0, step=0.1)
        age = st.number_input("Age of Vehicle (years)", min_value=0, max_value=30, value=5)
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=40.0, value=25.0, step=1.0)
        speed = st.number_input("Vehicle Speed (km/h)", min_value=10.0, max_value=120.0, value=50.0, step=5.0)
        wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)
        traffic_condition = st.selectbox("Traffic Condition", ["Free flow", "Moderate", "Heavy"])
        road_type = st.selectbox("Road Type", ["City", "Rural", "Highway"])

    # Load models and scalers
    models = {}
    scalers = {}
    try:
        models['Petrol'] = joblib.load('aqi/gradient_boosting_model.pkl')
        scalers['Petrol'] = joblib.load('aqi/scaler.pkl')
        models['Electric'] = joblib.load('aqi/gradient_boosting_electric_model.pkl')
        scalers['Electric'] = joblib.load('aqi/scaler_electric.pkl')
        models['Diesel'] = joblib.load('aqi/svr_diesel_model.pkl')
        scalers['Diesel'] = joblib.load('aqi/scaler_diesel.pkl')
    except FileNotFoundError:
        st.error("One or more model/scaler files not found. Ensure all .pkl files are in the 'aqi' directory.")
        st.stop()

    # Define features (consistent across all fuel types)
    features = [
        'Engine Size', 'Age of Vehicle', 'Temperature', 'Wind Speed', 'Speed','Traffic Conditions_Heavy',
        'Traffic Conditions_Moderate',
        'Road Type_Highway', 'Road Type_Rural'
    ]
    numerical_cols = ['Engine Size', 'Age of Vehicle', 'Temperature', 'Wind Speed', 'Speed']

    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = {ft: {'nox_emission': None, 'nox_aqi': None, 'aqi_category': None} for ft in ['Petrol', 'Electric', 'Diesel']}

    # Compute NOx and AQI for selected fuel type
    if st.button("Get Result"):
        if engine_size < 0.5 and fuel_type != 'Electric':
            st.warning("Engine Size is too small (less than 0.5 liters). Assuming NOx Emission is 0.")
            nox_emission_gkm = 0.0
        else:
            user_input = {
                'Engine Size': engine_size,
                'Age of Vehicle': age,
                'Temperature': temperature,
                'Wind Speed': wind_speed,
                'Speed': speed,
                'Traffic Conditions': traffic_condition,
                'Road Type': road_type
            }
            # Compute for selected fuel type
            nox_emission_gkm = predict_nox_emissions(user_input, models[fuel_type], scalers[fuel_type], features, numerical_cols)
            st.session_state.results[fuel_type]['nox_emission'] = nox_emission_gkm
            st.write(f"**Debug: Predicted NOx Emission ({fuel_type})**: {nox_emission_gkm:.4f} g/km")

        st.subheader(f"Prediction Result ({fuel_type})")
        st.write(f"**NOx Emission**: {nox_emission_gkm:.4f} g/km")

        if (engine_size < 0.0 or engine_size > 6.0 or age < 0 or age > 30 or
            temperature < -10 or temperature > 40 or speed < 10 or speed > 120 or
            wind_speed < 0 or wind_speed > 30):
            st.warning("Input values are outside typical ranges. Predictions may be less reliable.")

    # Compute AQI for all fuel types
    if st.button("Get AQI"):
        comparison_data = []
        for ft in ['Petrol', 'Electric', 'Diesel']:
            if st.session_state.results[ft]['nox_emission'] is not None:
                nox_emission_ugm3 = nox_gkm_to_ugm3(st.session_state.results[ft]['nox_emission'], speed, fuel_type=ft)
                nox_aqi = calculate_nox_aqi(nox_emission_ugm3)
                aqi_category = get_aqi_category(nox_aqi)
                st.session_state.results[ft]['nox_aqi'] = nox_aqi
                st.session_state.results[ft]['aqi_category'] = aqi_category

                st.subheader(f"AQI Result ({ft})")
                st.write(f"**Debug: NOx Concentration**: {nox_emission_ugm3:.2f} µg/m³")
                st.write(f"**NOx Emission (converted)**: {nox_emission_ugm3:.2f} µg/m³")
                st.write(f"**NOx AQI**: {nox_aqi:.1f}")
                st.write(f"**AQI Category**: {aqi_category}")

                # Individual bar chart
                results = pd.DataFrame({
                    'Metric': ['NOx Emission (µg/m³)', 'NOx AQI'],
                    'Value': [nox_emission_ugm3, nox_aqi]
                })
                fig = px.bar(results, x='Metric', y='Value', title=f'NOx Emission and AQI ({ft})',
                            color='Metric', text='Value', height=400)
                fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
                st.plotly_chart(fig)

                comparison_data.append({
                    'Fuel Type': ft,
                    'NOx AQI': nox_aqi,
                    'AQI Category': aqi_category,
                    'NOx Emission (µg/m³)': nox_emission_ugm3
                })

        # Comparison bar chart
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            fig = px.bar(comparison_df, x='Fuel Type', y='NOx AQI',
                        color='AQI Category', title='NOx AQI Comparison Across Fuel Types',
                        text='NOx AQI', height=500)
            fig.update_traces(texttemplate='%{text:.1f}', textposition='auto')
            st.plotly_chart(fig)

            worst_fuel = comparison_df.loc[comparison_df['NOx AQI'].idxmax(), 'Fuel Type']
            st.write(f"**Conclusion**: {worst_fuel} has the highest NOx AQI ({comparison_df['NOx AQI'].max():.1f}), indicating it is the most harmful to the environment.")
        else:
            st.info("Please click 'Get Result' for at least one fuel type to compute AQI.")