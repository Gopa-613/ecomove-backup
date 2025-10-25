import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from plotly.subplots import make_subplots
import time
from streamlit_lottie import st_lottie
import requests



st.set_page_config(page_title="Comparison Dashboard", layout="wide")
# Set custom CSS to change background color
# Force background color
#background_color = "#9CDBA6"  # Light gray
# Sidebar for user interaction




# Custom CSS for Bona Nova font with higher specificity
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bona+Nova&display=swap');     
    /* Target Streamlit's root container */
    .stApp, .stApp * {
        font-family: 'Bona Nova', serif !important;
        font-weight: 400 !important;
        font-style: normal !important;
    }
    
    /* Target specific Streamlit elements */
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stSelectbox, .stCheckbox, .stDataFrame, .stTable {
        font-family: 'Bona Nova', serif !important;
        font-weight: 400 !important;
        font-style: normal !important;
    }
    
    /* Ensure sidebar uses the font */
    [data-testid="stSidebar"], [data-testid="stSidebar"] * {
        font-family: 'Bona Nova', serif !important;
        font-weight: 400 !important;
        font-style: normal !important;
    }
    
    /* Debug font loading */
    @font-face {
        font-family: 'Bona Nova';
        font-weight: 400 !important;
        font-style: normal !important;
        src: url(@import url('https://fonts.googleapis.com/css2?family=Bona+Nova&display=swap');
    }
    </style>
    <div style="font-family: 'Bona Nova', serif; 
     font-weight: 400 !important;
        font-style: normal !important;display: none;">Font load test</div>
    """,
    unsafe_allow_html=True
)

# Sidebar for user interaction
st.sidebar.markdown(
    """
    <style>
    /* Style for Navigation header background */
    .navigation-header {
        font-size: 1.5em;
    border-bottom: 2px solid #cccc;
    font-weight: bold;
    background-color: #081c15;
    margin-bottom: 2px;
    /* border-radius: 5px; */
    width: 118%;
    padding: 10px;
    position: absolute;
    height: 128px;
    top: -90px;
    left: -25px;
    }
    
    .navigation-header p{
    color: white;
    position: absolute;
    top: 60px;
    left: 26px;
    }
    </style>
    <div class="navigation-header"><p>Ecomove</p></div>
    
    """,
    unsafe_allow_html=True
)



# Set page config
# Set Matplotlib style to ensure light backgrounds
plt.style.use('default')  # Reset to default Matplotlib style
sns.set_style("whitegrid")  # White background with grid
# Sidebar navigation
#st.sidebar.title("Navigation")
page = st.sidebar.radio(" ", [
    "Home",
    "AQI and Emissions",
    "NOx AQI and CO2 Emissions",
    "Age of Vehicles",
    "Light Vehicles",
    "Heavy Vehicles"
    #"Engine Size",
    #"Vehicle Type (NOx AQI)",
    #"Vehicle Type (CO2 Emissions)"
    
])

#st.sidebar.header("Chart Controls")
#color_scheme = st.sidebar.selectbox("Color Scheme", ["Default", "Vivid", "Muted"])
#animate = st.sidebar.checkbox("Enable Animation", value=True)


st.sidebar.header("Visualization Settings")
animate = st.sidebar.checkbox("Enable Animation", value=True)
color_scheme = st.sidebar.selectbox("Color Scheme", ["Default", "Vivid", "Muted"])
    # Color schemes
color_maps = {
    "Default": ['#16423C', '#6A9C89', '#9CDBA6'],
    "Vivid": ['#74d3ae', '#678d58', '#bad7f2'],
    "Muted": ['#8fc0a9', '#c8d5b9', '#68b0ab']
    }
colors = color_maps[color_scheme]

# Close the sidebar-content div
# st.sidebar.markdown("</div>", unsafe_allow_html=True)


# Home Page
if page == "Home":
    st.title("Comparison Dashboard")
    st.write("Explore visualizations based on age, weight of Vehicles and many more.")
    st.markdown("Ecomove empowers you to explore the environmental impact of transportation through vivid, data-driven comparisons. Dive into interactive graphs showcasing CO2, NOx, VOC, PM2.5, and SO2 emissions across Petrol, Diesel, and Electric vehicles—spanning Buses, Trucks, Cars, and Motorcycles. Our cutting-edge visualizations reveal which fuel types and vehicles drive the highest environmental cost, helping you make informed, sustainable choices. Discover the future of mobility with Ecomove’s powerful insights! [Visit Front-End Website](../frontend/index.html)")
    st.write("Use the sidebar to navigate to different sections.")


elif page == "Age of Vehicles":
    st.title("Comparison Based on Age of Vehicles")
    
    df = pd.read_csv('dataset/vehicle_emission_dataset.csv')
    # Data preprocessing
    df_filtered = df[df['Age of Vehicle'].between(10, 30)]

    # Define bins and labels for Age of Vehicle categorization
    bins = [10, 20, 30]
    labels = ['10-20', '20-30']
    df_filtered['Age_Range'] = pd.cut(df_filtered['Age of Vehicle'], bins=bins, labels=labels, include_lowest=True)

    # Filter for relevant fuel types
    fuel_types = ['Diesel', 'Electric', 'Petrol']
    df_filtered = df_filtered[df_filtered['Fuel Type'].isin(fuel_types)]

    # Group by Age_Range and Fuel Type, and calculate mean NOx_AQI
    nox_aqi_means = df_filtered.groupby(['Age_Range', 'Fuel Type'])['NOx_AQI'].mean().unstack().reset_index()

    # Sidebar for user interaction
    # Create interactive bar chart
    st.subheader("📊 Average NOx AQI by Fuel Type and Vehicle Age Range")
    # Create bar chart with animation
    fig = px.bar(
        nox_aqi_means,
        x="Age_Range",
        y=fuel_types,
        barmode="group",
        title="  ",
        template="plotly_dark",
        labels={"Age_Range": "Age of Vehicle Range (Years)", "value": "Average NOx AQI", "variable": "Fuel Type"}
    )

    # Customize hover and styling
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Fuel Type: %{meta}<br>NOx_AQI: %{y:.2f}",
        meta=[col for col in nox_aqi_means[fuel_types].columns] * len(nox_aqi_means),
        marker=dict(line=dict(width=1, color="white"))
    )

    # Apply custom colors
    for i, trace in enumerate(fig.data):
        trace.marker.color = colors[i % len(colors)]

    # Add animation for bars growing on load
    fig.update_layout(
        showlegend=True,
        title_x=0.5,
        font=dict(size=14),
        hovermode="x unified",
        bargap=0.2,
        transition={"duration": 1000, "easing": "cubic-in-out"},  # Animation duration and easing
        yaxis=dict(range=[0, max(nox_aqi_means[fuel_types].max().max() * 1.1, 1)]),  # Set y-axis range
        updatemenus=[{
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "Reset",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": True,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    # Create animation frames for bars growing from zero
    frames = []
    steps = 10  # Number of animation steps
    for step in range(steps + 1):
        fraction = step / steps
        frame_traces = []
        for col in fuel_types:
            if col in nox_aqi_means:
                frame_traces.append(
                    go.Bar(
                        x=nox_aqi_means["Age_Range"],
                        y=nox_aqi_means[col] * fraction,  # Scale bar height
                        name=col,
                        marker=dict(color=colors[fuel_types.index(col) % len(colors)]),
                        hovertemplate="<b>%{x}</b><br>Fuel Type: " + col + "<br>NOx_AQI: %{y:.2f}"
                    )
                )
        frames.append(go.Frame(data=frame_traces))
    fig.frames = frames

    # Auto-play animation on load
    fig.update_layout(
        annotations=[
            dict(
                text="",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=0
            )
        ],
        # Trigger animation on load
        updatemenus=[{
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "Reset",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": True,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
            "visible": True
        }],
        # Start with zero height bars
        yaxis=dict(range=[0, max(nox_aqi_means[fuel_types].max().max() * 1.1, 1)])
    )

    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    # Conclusion
    st.subheader("Conclusion")
    st.markdown("Based on the grouped bar plot and Average NOx AQI values:")
    for engine_range in nox_aqi_means["Age_Range"]:
        st.markdown(f"**For Age Range {engine_range}:**")
        ev_nox = nox_aqi_means[nox_aqi_means["Age_Range"] == engine_range]["Electric"].iloc[0] if 'Electric' in nox_aqi_means else float('nan')
        diesel_nox = nox_aqi_means[nox_aqi_means["Age_Range"] == engine_range]["Diesel"].iloc[0] if 'Diesel' in nox_aqi_means else float('nan')
        petrol_nox = nox_aqi_means[nox_aqi_means["Age_Range"] == engine_range]["Petrol"].iloc[0] if 'Petrol' in nox_aqi_means else float('nan')
        
        if not pd.isna(ev_nox) and not pd.isna(diesel_nox):
            st.markdown(f"- EV vs Diesel: EVs have {'lower' if ev_nox < diesel_nox else 'higher'} NOx AQI ({ev_nox:.2f}) compared to Diesel ({diesel_nox:.2f}).")
        if not pd.isna(ev_nox) and not pd.isna(petrol_nox):
            st.markdown(f"- EV vs Petrol: EVs have {'lower' if ev_nox < petrol_nox else 'higher'} NOx AQI ({ev_nox:.2f}) compared to Petrol ({petrol_nox:.2f}).")
 
    

elif page == "Light Vehicles":
    
    # Title and description
    st.title("📊 NOx AQI and CO2 Emissions for Light Vehicles (Motorcycle & Car)")

    st.markdown("Interactive dashboard showing average Nox AQI and CO2 Emissions by Fuel Type for light vehicles")
    # Sidebar for user interaction
    
    st.sidebar.header("Visualization Settings")
    plot_type = st.sidebar.selectbox("Select Plot to Display", ["NOx AQI", "CO2 Emissions", "Both"], index=2, key="plot_type_selectbox")
    # Load dataset
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("dataset/vehicle_emission_dataset.csv")
        except FileNotFoundError:
            st.error("Dataset 'vehicle_emission_dataset.csv' not found. Please ensure the file is in the 'dataset' directory.")
            st.stop()
        except pd.errors.EmptyDataError:
            st.error("Dataset is empty. Please check the file content.")
            st.stop()

    df = load_data()
    

    # Data preprocessing
    light_vehicles = df[df["Vehicle Type"].isin(["Motorcycle", "Car"])]
    if light_vehicles.empty:
        st.error("No data available for Motorcycle or Car. Please check the dataset.")
        st.stop()
        
        
        
    # Filter for specific fuel types: Petrol, Diesel, Electric
    light_vehicles = light_vehicles[light_vehicles["Fuel Type"].isin(["Petrol", "Diesel", "Electric"])]
    if light_vehicles.empty:
        st.error("No data available for Petrol, Diesel, or Electric fuel types. Please check the dataset.")
        st.stop()

    # Calculate averages
    light_co2_aqi_avg = light_vehicles.groupby("Fuel Type")["NOx_AQI"].mean().reset_index()
    light_co2_emission_avg = light_vehicles.groupby("Fuel Type")["CO2 Emissions"].mean().reset_index()

    
    #line_color = '#FF0000'  # Red for line chart

    # Function to create animated bar and line charts
    def create_charts(data, y_col, y_label, title_prefix, chart_key_prefix):
        # Ensure data contains expected fuel types
        expected_fuels = ["Petrol", "Diesel", "Electric"]
        data = data[data["Fuel Type"].isin(expected_fuels)]
        if data.empty:
            st.error(f"No data available for {y_label} with fuel types: {expected_fuels}")
            return go.Figure()

        # Sort data to ensure consistent order (Petrol, Diesel, Electric)
        data = data.set_index("Fuel Type").reindex(expected_fuels).reset_index()
        data = data.dropna(subset=[y_col])  # Drop any NaN values after reindexing

        fig = go.Figure()


        # Bar chart (static)
        for i, row in data.iterrows():
            fuel, value = row["Fuel Type"], row[y_col]
            fig.add_trace(
                go.Bar(
                    x=[fuel],
                    y=[value],
                    name=fuel,
                    marker=dict(color=colors[i % len(colors)], line=dict(width=1, color="white")),
                    hovertemplate=f"<b>{fuel}</b><br>{y_label}: %{{y:.5f}}<extra></extra>",
                    text=[f"{value:.5f}"],
                    textposition="auto"
                ),
            )

    

        # Animation
        if animate:
            frames = []
            # Include all fuel types in each frame, setting y=0 for not-yet-displayed fuels
            for i in range(len(data) + 1):
                frame_traces = []
                # Bar
                for j, row in data.iterrows():
                    fuel, value = row["Fuel Type"], row[y_col]
                    # Use full value if fuel is included in this frame, else 0
                    y_value = value if j < i else 0
                    frame_traces.append(
                        go.Bar(
                            x=[fuel],
                            y=[y_value],
                            name=fuel,
                            marker=dict(color=colors[j % len(colors)], line=dict(width=1, color="white")),
                            hovertemplate=f"<b>{fuel}</b><br>{y_label}: %{{y:.5f}}<extra></extra>",
                            text=[f"{value:.5f}" if j < i else ""],
                            textposition="auto"
                        )
                    )
            
                frames.append(go.Frame(data=frame_traces, name=f"frame{i}"))
            fig.frames = frames

            # Animation controls
            fig.update_layout(
                updatemenus=[{
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                        },
                        {
                            "label": "Reset",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            )

        # Update layout
        max_y = data[y_col].max() * 1.1 if not data[y_col].empty else 1
        fig.update_layout(
            template="plotly_dark",
            showlegend=True,
            height=500,
            margin=dict(t=100),
            hovermode="closest",
            title=dict(text=title_prefix, font_size=20, x=0.5),
            xaxis_title="Fuel Type",
            yaxis_title=f"Average {y_label}",
            yaxis_range=[0, max_y]
        )
            
        return fig

    # Display charts based on user selection
    if plot_type in ["NOx AQI", "Both"]:
        st.subheader("Average NOx AQI for Light Vehicles (Motorcycle + Car)")
        fig_aqi = create_charts(
            light_co2_aqi_avg, "NOx_AQI", "NOx AQI",
            "  ", "co2_aqi"
        )
        st.plotly_chart(fig_aqi, use_container_width=True)

    if plot_type in ["CO2 Emissions", "Both"]:
        st.subheader("Average CO2 Emissions for Light Vehicles (Motorcycle + Car)")
        fig_emissions = create_charts(
            light_co2_emission_avg, "CO2 Emissions", "CO2 Emissions",
            "  ", "co2_emissions"
        )
        st.plotly_chart(fig_emissions, use_container_width=True)

    # Display data tables
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Average NOx AQI by Fuel Type**")
        st.dataframe(
            light_co2_aqi_avg.set_index("Fuel Type").style.format("{:.5f}").background_gradient(cmap="Greens"),
            use_container_width=True
        )
    with col2:
        st.markdown("**Average CO2 Emissions by Fuel Type**")
        st.dataframe(
            light_co2_emission_avg.set_index("Fuel Type").style.format("{:.5f}").background_gradient(cmap="Greens"),
            use_container_width=True
        )

    # Footer
    st.markdown("---")





elif page == "Heavy Vehicles":
    

    # Title and description
    st.title("📊 NOx AQI and CO2 Emissions for Heavy Vehicles (Bus & Truck)")

    st.markdown("Interactive dashboard showing average Nox AQI and CO2 Emissions by Fuel Type for Heavy vehicles")
    # Sidebar for user interaction
    
    st.sidebar.header("Visualization Settings")
    plot_type = st.sidebar.selectbox("Select Plot to Display", ["NOx AQI", "CO2 Emissions", "Both"], index=2, key="plot_type_selectbox")
    # Load dataset
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("dataset/vehicle_emission_dataset.csv")
        except FileNotFoundError:
            st.error("Dataset 'vehicle_emission_dataset.csv' not found. Please ensure the file is in the 'dataset' directory.")
            st.stop()
        except pd.errors.EmptyDataError:
            st.error("Dataset is empty. Please check the file content.")
            st.stop()

    df = load_data()
    

    # Data preprocessing
    heavy_vehicles = df[df["Vehicle Type"].isin(["Bus", "Truck"])]
    if heavy_vehicles.empty:
        st.error("No data available for Bus or Truck. Please check the dataset.")
        st.stop()
        
        
        
    # Filter for specific fuel types: Petrol, Diesel, Electric
    heavy_vehicles = heavy_vehicles[heavy_vehicles["Fuel Type"].isin(["Petrol", "Diesel", "Electric"])]
    if heavy_vehicles.empty:
        st.error("No data available for Petrol, Diesel, or Electric fuel types. Please check the dataset.")
        st.stop()

    # Calculate averages
    heavy_co2_aqi_avg = heavy_vehicles.groupby("Fuel Type")["NOx_AQI"].mean().reset_index()
    heavy_co2_emission_avg = heavy_vehicles.groupby("Fuel Type")["CO2 Emissions"].mean().reset_index()

    
    # Function to create animated bar and line charts
    def create_charts(data, y_col, y_label, title_prefix, chart_key_prefix):
        # Ensure data contains expected fuel types
        expected_fuels = ["Petrol", "Diesel", "Electric"]
        data = data[data["Fuel Type"].isin(expected_fuels)]
        if data.empty:
            st.error(f"No data available for {y_label} with fuel types: {expected_fuels}")
            return go.Figure()

        # Sort data to ensure consistent order (Petrol, Diesel, Electric)
        data = data.set_index("Fuel Type").reindex(expected_fuels).reset_index()
        data = data.dropna(subset=[y_col])  # Drop any NaN values after reindexing

        fig = go.Figure()


        # Bar chart (static)
        for i, row in data.iterrows():
            fuel, value = row["Fuel Type"], row[y_col]
            fig.add_trace(
                go.Bar(
                    x=[fuel],
                    y=[value],
                    name=fuel,
                    marker=dict(color=colors[i % len(colors)], line=dict(width=1, color="white")),
                    hovertemplate=f"<b>{fuel}</b><br>{y_label}: %{{y:.5f}}<extra></extra>",
                    text=[f"{value:.5f}"],
                    textposition="auto"
                ),
            )

    

        # Animation
        if animate:
            frames = []
            # Include all fuel types in each frame, setting y=0 for not-yet-displayed fuels
            for i in range(len(data) + 1):
                frame_traces = []
                # Bar
                for j, row in data.iterrows():
                    fuel, value = row["Fuel Type"], row[y_col]
                    # Use full value if fuel is included in this frame, else 0
                    y_value = value if j < i else 0
                    frame_traces.append(
                        go.Bar(
                            x=[fuel],
                            y=[y_value],
                            name=fuel,
                            marker=dict(color=colors[j % len(colors)], line=dict(width=1, color="white")),
                            hovertemplate=f"<b>{fuel}</b><br>{y_label}: %{{y:.5f}}<extra></extra>",
                            text=[f"{value:.5f}" if j < i else ""],
                            textposition="auto"
                        )
                    )
            
                frames.append(go.Frame(data=frame_traces, name=f"frame{i}"))
            fig.frames = frames

            # Animation controls
            fig.update_layout(
                updatemenus=[{
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                        },
                        {
                            "label": "Reset",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            )

        # Update layout
        max_y = data[y_col].max() * 1.1 if not data[y_col].empty else 1
        fig.update_layout(
            template="plotly_dark",
            showlegend=True,
            height=500,
            margin=dict(t=100),
            hovermode="closest",
            title=dict(text=title_prefix, font_size=20, x=0.5),
            xaxis_title="Fuel Type",
            yaxis_title=f"Average {y_label}",
            yaxis_range=[0, max_y]
        )
            
        return fig

    # Display charts based on user selection
    if plot_type in ["NOx AQI", "Both"]:
        st.subheader("Average NOx AQI for Heavy Vehicles (Bus + Truck)")
        fig_aqi = create_charts(
            heavy_co2_aqi_avg, "NOx_AQI", "NOx AQI",
            "  ", "co2_aqi"
        )
        st.plotly_chart(fig_aqi, use_container_width=True)

    if plot_type in ["CO2 Emissions", "Both"]:
        st.subheader("Average CO2 Emissions for Heavy Vehicles (Bus + Truck)")
        fig_emissions = create_charts(
            heavy_co2_emission_avg, "CO2 Emissions", "CO2 Emissions",
            " ", "co2_emissions"
        )
        st.plotly_chart(fig_emissions, use_container_width=True)

    # Display data tables
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Average NOx AQI by Fuel Type**")
        st.dataframe(
            heavy_co2_aqi_avg.set_index("Fuel Type").style.format("{:.5f}").background_gradient(cmap="Greens"),
            use_container_width=True
        )
    with col2:
        st.markdown("**Average CO2 Emissions by Fuel Type**")
        st.dataframe(
            heavy_co2_emission_avg.set_index("Fuel Type").style.format("{:.5f}").background_gradient(cmap="Greens"),
            use_container_width=True
        )

    # Footer
    st.markdown("---")


elif page =="NOx AQI and CO2 Emissions":




    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("dataset/vehicle_emission_dataset.csv")
        except FileNotFoundError:
            st.error("Dataset 'vehicle_emission_dataset.csv' not found. Please ensure the file is in the 'dataset' directory.")
            st.stop()
        except pd.errors.EmptyDataError:
            st.error("Dataset is empty. Please check the file content.")
            st.stop()

    df = load_data()

    # Data preprocessing
    # Filter for specific fuel types: Petrol, Diesel, Electric (no vehicle type filter)
    vehicles = df[df["Fuel Type"].isin(["Petrol", "Diesel", "Electric"])]
    if vehicles.empty:
        st.error("No data available for Petrol, Diesel, or Electric fuel types. Please check the dataset.")
        st.stop()

    # Calculate averages
    co2_emission_avg = vehicles.groupby("Fuel Type")["CO2 Emissions"].mean().reset_index()
    nox_aqi_avg = vehicles.groupby("Fuel Type")["NOx_AQI"].mean().reset_index()


    # Function to create animated subplot bar charts
    def create_charts(co2_data, nox_data, chart_key_prefix):
        # Ensure data contains expected fuel types and order
        expected_fuels = ["Petrol", "Diesel", "Electric"]
        co2_data = co2_data[co2_data["Fuel Type"].isin(expected_fuels)].set_index("Fuel Type").reindex(expected_fuels).reset_index()
        nox_data = nox_data[nox_data["Fuel Type"].isin(expected_fuels)].set_index("Fuel Type").reindex(expected_fuels).reset_index()
        co2_data = co2_data.dropna(subset=["CO2 Emissions"])
        nox_data = nox_data.dropna(subset=["NOx_AQI"])
        
        if co2_data.empty or nox_data.empty:
            st.error("No data available for the specified fuel types.")
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Average CO2 Emissions", "Average NOx AQI"],
            horizontal_spacing=0.15
        )

        # Static bar chart: CO2 Emissions
        for i, row in co2_data.iterrows():
            fuel, value = row["Fuel Type"], row["CO2 Emissions"]
            fig.add_trace(
                go.Bar(
                    x=[fuel],
                    y=[value],
                    name=fuel,
                    marker=dict(color=colors[i % len(colors)], line=dict(width=1, color="white")),
                    hovertemplate=f"<b>{fuel}</b><br>CO2 Emissions: %{{y:.2f}} g/km<extra></extra>",
                    text=[f"{value:.2f}"],
                    textposition="auto",
                    showlegend=True
                ),
                row=1, col=1
            )

        # Static bar chart: NOx AQI
        for i, row in nox_data.iterrows():
            fuel, value = row["Fuel Type"], row["NOx_AQI"]
            fig.add_trace(
                go.Bar(
                    x=[fuel],
                    y=[value],
                    name=fuel,
                    marker=dict(color=colors[i % len(colors)], line=dict(width=1, color="white")),
                    hovertemplate=f"<b>{fuel}</b><br>NOx AQI: %{{y:.2f}}<extra></extra>",
                    text=[f"{value:.2f}"],
                    textposition="auto",
                    showlegend=False  # Avoid duplicate legend entries
                ),
                row=1, col=2
            )

        # Animation
        if animate:
            frames = []
            for i in range(len(expected_fuels) + 1):
                frame_traces = []
                # CO2 Emissions bars
                for j, row in co2_data.iterrows():
                    fuel, value = row["Fuel Type"], row["CO2 Emissions"]
                    y_value = value if j < i else 0
                    frame_traces.append(
                        go.Bar(
                            x=[fuel],
                            y=[y_value],
                            name=fuel,
                            marker=dict(color=colors[j % len(colors)], line=dict(width=1, color="white")),
                            hovertemplate=f"<b>{fuel}</b><br>CO2 Emissions: %{{y:.2f}} g/km<extra></extra>",
                            text=[f"{value:.2f}" if j < i else ""],
                            textposition="auto",
                            showlegend=True
                        )
                    )
                # NOx AQI bars
                for j, row in nox_data.iterrows():
                    fuel, value = row["Fuel Type"], row["NOx_AQI"]
                    y_value = value if j < i else 0
                    frame_traces.append(
                        go.Bar(
                            x=[fuel],
                            y=[y_value],
                            name=fuel,
                            marker=dict(color=colors[j % len(colors)], line=dict(width=1, color="white")),
                            hovertemplate=f"<b>{fuel}</b><br>NOx AQI: %{{y:.2f}}<extra></extra>",
                            text=[f"{value:.2f}" if j < i else ""],
                            textposition="auto",
                            showlegend=False
                        )
                    )
                frames.append(go.Frame(data=frame_traces, name=f"frame{i}"))
            fig.frames = frames

            # Animation controls
            fig.update_layout(
                updatemenus=[{
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                        },
                        {
                            "label": "Reset",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            )

        # Update layout
        max_y_co2 = co2_data["CO2 Emissions"].max() * 1.1 if not co2_data["CO2 Emissions"].empty else 1
        max_y_nox = nox_data["NOx_AQI"].max() * 1.1 if not nox_data["NOx_AQI"].empty else 1
        fig.update_layout(
            template="plotly_dark",
            showlegend=True,
            height=500,
            margin=dict(t=100),
            hovermode="closest",
            title=dict(text="  ", font_size=20, x=0.5)
        )
        fig.update_xaxes(title_text="Fuel Type", row=1, col=1)
        fig.update_xaxes(title_text="Fuel Type", row=1, col=2)
        fig.update_yaxes(title_text="Average CO2 Emissions (g/km)", range=[0, max_y_co2], row=1, col=1)
        fig.update_yaxes(title_text="Average NOx AQI (µg/m³)", range=[0, max_y_nox], row=1, col=2)

        return fig

    # Display charts
    st.subheader("Average CO2 Emissions and NOx AQI for all Vehicles")
    fig = create_charts(co2_emission_avg, nox_aqi_avg, "co2_nox")
    st.plotly_chart(fig, use_container_width=True)

    # Display data tables
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Average CO2 Emissions by Fuel Type**")
        st.dataframe(
            co2_emission_avg.set_index("Fuel Type").style.format("{:.2f}").background_gradient(cmap="Greens"),
            use_container_width=True
        )
    with col2:
        st.markdown("**Average NOx AQI by Fuel Type**")
        st.dataframe(
            nox_aqi_avg.set_index("Fuel Type").style.format("{:.2f}").background_gradient(cmap="Greens"),
            use_container_width=True
        )

    # Conclusion
    st.subheader("Conclusion")
    if not co2_emission_avg.empty and not nox_aqi_avg.empty:
        max_co2_fuel = co2_emission_avg.loc[co2_emission_avg["CO2 Emissions"].idxmax(), "Fuel Type"]
        min_co2_fuel = co2_emission_avg.loc[co2_emission_avg["CO2 Emissions"].idxmin(), "Fuel Type"]
        max_nox_fuel = nox_aqi_avg.loc[nox_aqi_avg["NOx_AQI"].idxmax(), "Fuel Type"]
        min_nox_fuel = nox_aqi_avg.loc[nox_aqi_avg["NOx_AQI"].idxmin(), "Fuel Type"]
        
        conclusion = f"""
        Based on the analysis of all vehicle types:
        - **CO2 Emissions**: {max_co2_fuel} has the highest average CO2 emissions, while {min_co2_fuel} has the lowest.
        - **NOx AQI**: {max_nox_fuel} has the highest average NOx AQI, contributing most to air quality degradation, while {min_nox_fuel} has the lowest, making it the least impactful on air quality.
        """
        st.markdown(conclusion)
    else:
        st.warning("Insufficient data to generate a conclusion.")
        
elif page == "AQI and Emissions":
    
    
    
    # Title and description
    st.title("📊 Emissions Analysis for All Vehicles")
    
    st.markdown("Interactive dashboard showing average CO2, NOx, VOC, PM2.5, and SO2 Emissions by Fuel Type (Petrol, Diesel, Electric)")

    # Load dataset
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("dataset/sample.csv")
        except FileNotFoundError:
            st.error("Dataset 'sample.csv' not found. Please ensure the file is in the 'dataset' directory.")
            st.stop()
        except pd.errors.EmptyDataError:
            st.error("Dataset is empty. Please check the file content.")
            st.stop()

    df = load_data()
    

    # Data preprocessing
    # Filter for specific fuel types: Petrol, Diesel, Electric (no vehicle type filter)
    vehicles = df[df["Fuel Type"].isin(["Petrol", "Diesel", "Electric"])]
    if vehicles.empty:
        st.error("No data available for Petrol, Diesel, or Electric fuel types. Please check the dataset.")
        st.stop()

    # Calculate averages
    emission_types = ["CO2 Emissions", "NOx Emissions", "VOC Emissions", "PM2.5 Emissions", "SO2 Emissions"]
    avg_data = {emission: vehicles.groupby("Fuel Type")[emission].mean().reset_index() for emission in emission_types}

    

    
    # Function to create animated subplot bar charts
    def create_charts(avg_data, chart_key_prefix):
        # Ensure data contains expected fuel types and order
        expected_fuels = ["Petrol", "Diesel", "Electric"]
        for emission in avg_data:
            avg_data[emission] = avg_data[emission][avg_data[emission]["Fuel Type"].isin(expected_fuels)].set_index("Fuel Type").reindex(expected_fuels).reset_index()
            avg_data[emission] = avg_data[emission].dropna(subset=[emission])
            if avg_data[emission].empty:
                st.error(f"No data available for {emission}.")
                return go.Figure()

        # Create subplots (2x3 grid, last cell empty)
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=["CO2 Emissions", "NOx Emissions", "VOC Emissions", "PM2.5 Emissions", "SO2 Emissions"],
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
            specs=[[{}, {}, {}], [{}, {}, None]]  # Last cell in second row is empty
        )

        # Add static bar charts
        for idx, emission in enumerate(emission_types):
            row = 1 if idx < 3 else 2
            col = (idx % 3) + 1
            data = avg_data[emission]
            for i, row_data in data.iterrows():
                fuel, value = row_data["Fuel Type"], row_data[emission]
                fig.add_trace(
                    go.Bar(
                        x=[fuel],
                        y=[value],
                        name=fuel,
                        marker=dict(color=colors[i % len(colors)], line=dict(width=1, color="white")),
                        hovertemplate=f"<b>{fuel}</b><br>{emission}: %{{y:.2f}} g/km<extra></extra>",
                        text=[f"{value:.2f}"],
                        textposition="auto",
                        showlegend=(idx == 0)  # Show legend only for first subplot
                    ),
                    row=row, col=col
                )

        # Animation
        if animate:
            frames = []
            for i in range(len(expected_fuels) + 1):
                frame_traces = []
                trace_idx = 0
                for idx, emission in enumerate(emission_types):
                    data = avg_data[emission]
                    for j, row_data in data.iterrows():
                        fuel, value = row_data["Fuel Type"], row_data[emission]
                        y_value = value if j < i else 0
                        frame_traces.append(
                            go.Bar(
                                x=[fuel],
                                y=[y_value],
                                name=fuel,
                                marker=dict(color=colors[j % len(colors)], line=dict(width=1, color="white")),
                                hovertemplate=f"<b>{fuel}</b><br>{emission}: %{{y:.2f}} g/km<extra></extra>",
                                text=[f"{value:.2f}" if j < i else ""],
                                textposition="auto",
                                showlegend=(idx == 0 and j < i)
                            )
                        )
                        trace_idx += 1
                frames.append(go.Frame(data=frame_traces, name=f"frame{i}"))
            fig.frames = frames

            # Animation controls
            fig.update_layout(
                updatemenus=[{
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                        },
                        {
                            "label": "Reset",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            )

        # Update layout
        max_y = {emission: avg_data[emission][emission].max() * 1.1 if not avg_data[emission][emission].empty else 1 for emission in emission_types}
        fig.update_layout(
            template="plotly_dark",
            showlegend=True,
            height=800,
            margin=dict(t=100),
            hovermode="closest",
            title=dict(text=" ", font_size=20, x=0.5)
        )
        for idx, emission in enumerate(emission_types):
            row = 1 if idx < 3 else 2
            col = (idx % 3) + 1
            fig.update_xaxes(title_text="Fuel Type", row=row, col=col)
            fig.update_yaxes(title_text=f"Average {emission} (g/km)", range=[0, max_y[emission]], row=row, col=col)

        return fig

    # Display charts
    st.subheader("Average Emissions by Fuel Type")
    fig = create_charts(avg_data, "emissions")
    st.plotly_chart(fig, use_container_width=True)

    # Display data tables
    st.subheader("Data Summary")
    cols = st.columns(3)
    for idx, emission in enumerate(emission_types):
        col = cols[idx % 3]
        with col:
            st.markdown(f"**Average {emission} by Fuel Type**")
            st.dataframe(
                avg_data[emission].set_index("Fuel Type").style.format("{:.2f}").background_gradient(cmap="Greens"),
                use_container_width=True
            )

    # Conclusion
    st.subheader("Conclusion")
    if all(not avg_data[emission].empty for emission in emission_types):
        conclusion = "Based on the analysis of all vehicle types:\n"
        for emission in emission_types:
            max_fuel = avg_data[emission].loc[avg_data[emission][emission].idxmax(), "Fuel Type"]
            min_fuel = avg_data[emission].loc[avg_data[emission][emission].idxmin(), "Fuel Type"]
            conclusion += f"- **{emission}**: {max_fuel} has the highest average emissions, while {min_fuel} has the lowest, indicating {min_fuel} is the most environmentally friendly for this pollutant.\n"
        #conclusion += """
        #**Insight**: Electric vehicles consistently show lower emissions across CO2, NOx, VOC, PM2.5, and SO2 compared to Petrol and Diesel, highlighting their potential to reduce environmental impact across all vehicle types. Diesel and Petrol exhibit trade-offs, with Diesel often higher in NOx and PM2.5, while Petrol may contribute more to VOC and CO2. These findings suggest prioritizing electric vehicles for sustainability, with careful consideration of Diesel vs. Petrol based on specific emission priorities.
        #"""
        st.markdown(conclusion)
    else:
        st.warning("Insufficient data to generate a conclusion.")
    
    

    
    # Data preprocessing
    # Filter for specific fuel types: Petrol, Diesel, Electric (no vehicle type filter)
    vehicles = df[df["Fuel Type"].isin(["Petrol", "Diesel", "Electric"])]
    if vehicles.empty:
        st.error("No data available for Petrol, Diesel, or Electric fuel types. Please check the dataset.")
        st.stop()

    # Calculate averages
    emission_types = ["NOx_AQI", "PM2.5_AQI", "SO2_AQI"]
    avg_data = {emission: vehicles.groupby("Fuel Type")[emission].mean().reset_index() for emission in emission_types}

    # Sidebar for user interaction


    # Function to create animated subplot bar charts
    def create_charts(avg_data, chart_key_prefix):
        # Ensure data contains expected fuel types and order
        expected_fuels = ["Petrol", "Diesel", "Electric"]
        for emission in avg_data:
            avg_data[emission] = avg_data[emission][avg_data[emission]["Fuel Type"].isin(expected_fuels)].set_index("Fuel Type").reindex(expected_fuels).reset_index()
            avg_data[emission] = avg_data[emission].dropna(subset=[emission])
            if avg_data[emission].empty:
                st.error(f"No data available for {emission}.")
                return go.Figure()

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["NOx AQI", "PM2.5 AQI", "SO2 AQI"],
            horizontal_spacing=0.1,
        )

        # Add static bar charts
        for idx, emission in enumerate(emission_types):
            data = avg_data[emission]
            col = idx + 1  # Column 1, 2, 3
            for i, row_data in data.iterrows():
                fuel, value = row_data["Fuel Type"], row_data[emission]
                fig.add_trace(
                    go.Bar(
                        x=[fuel],
                        y=[value],
                        name=fuel if col == 1 else None,  # Show legend only in first subplot,
                        marker=dict(color=colors[i % len(colors)], line=dict(width=1, color="white")),
                        hovertemplate=f"<b>{fuel}</b><br>{emission}: %{{y:.2f}} g/km<extra></extra>",
                        text=[f"{value:.2f}"],
                        textposition="auto",
                        showlegend=(col == 1)  # Show legend only for first subplot
                    ),
                    row=1, col=col
                )

        # Animation
        if animate:
            frames = []
            for i in range(len(expected_fuels) + 1):
                frame_traces = []
                trace_idx = 0
                for idx, emission in enumerate(emission_types):
                    data = avg_data[emission]
                    for j, row_data in data.iterrows():
                        fuel, value = row_data["Fuel Type"], row_data[emission]
                        y_value = value if j < i else 0
                        frame_traces.append(
                            go.Bar(
                                x=[fuel],
                                y=[y_value],
                                name=fuel,
                                marker=dict(color=colors[j % len(colors)], line=dict(width=1, color="white")),
                                hovertemplate=f"<b>{fuel}</b><br>{emission}: %{{y:.2f}} g/km<extra></extra>",
                                text=[f"{value:.2f}" if j < i else ""],
                                textposition="auto",
                                showlegend=(idx == 0 and j < i)
                            )
                        )
                        trace_idx += 1
                frames.append(go.Frame(data=frame_traces, name=f"frame{i}"))
            fig.frames = frames

            # Animation controls
            fig.update_layout(
                updatemenus=[{
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                        },
                        {
                            "label": "Reset",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            )

        # Update layout
        max_y = {emission: avg_data[emission][emission].max() * 1.1 if not avg_data[emission][emission].empty else 1 for emission in emission_types}
        fig.update_layout(
            template="plotly_dark",
            showlegend=True,
            height=500,
            margin=dict(t=100),
            hovermode="closest",
            title=dict(text=" ", font_size=20, x=0.5)
        )
        for idx, emission in enumerate(emission_types):
            row = 1 if idx < 3 else 2
            col = (idx % 3) + 1
            fig.update_xaxes(title_text="Fuel Type", row=row, col=col)
            fig.update_yaxes(title_text=f"Average {emission} (ug/m³)", range=[0, max_y[emission]], row=row, col=col)

        return fig

    # Display charts
    st.subheader("Average AQI by Fuel Type")
    fig = create_charts(avg_data, "emissions")
    st.plotly_chart(fig, use_container_width=True)

    # Display data tables
    st.subheader("Data Summary")
    cols = st.columns(3)
    for idx, emission in enumerate(emission_types):
        col = cols[idx % 3]
        with col:
            st.markdown(f"**Average {emission} by Fuel Type**")
            st.dataframe(
                avg_data[emission].set_index("Fuel Type").style.format("{:.2f}").background_gradient(cmap="Greens"),
                use_container_width=True
            )

    # Conclusion
    st.subheader("Conclusion")
    if all(not avg_data[emission].empty for emission in emission_types):
        conclusion = "Based on the analysis of all vehicle types:\n"
        for emission in emission_types:
            max_fuel = avg_data[emission].loc[avg_data[emission][emission].idxmax(), "Fuel Type"]
            min_fuel = avg_data[emission].loc[avg_data[emission][emission].idxmin(), "Fuel Type"]
            conclusion += f"- **{emission}**: {max_fuel} has the highest average AQI, while {min_fuel} has the lowest, indicating {min_fuel} is the most environmentally friendly for this pollutant.\n"
        #conclusion += """
        #**Insight**: Electric vehicles consistently show lower emissions across CO2, NOx, VOC, PM2.5, and SO2 compared to Petrol and Diesel, highlighting their potential to reduce environmental impact across all vehicle types. Diesel and Petrol exhibit trade-offs, with Diesel often higher in NOx and PM2.5, while Petrol may contribute more to VOC and CO2. These findings suggest prioritizing electric vehicles for sustainability, with careful consideration of Diesel vs. Petrol based on specific emission priorities.
        
        #"""
        st.markdown(conclusion)
    else:
        st.warning("Insufficient data to generate a conclusion.")






        
        

            