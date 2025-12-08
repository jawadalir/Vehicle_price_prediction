# streamlit_app.py
import streamlit as st
# Clear cache at startup
st.cache_data.clear()
st.cache_resource.clear()

import numpy as np
import pandas as pd
import joblib
import warnings
import re
import json
import os
from datetime import datetime
import base64
from io import StringIO
warnings.filterwarnings('ignore')

# --------------------------------------------------------
# 1️⃣ Helper Functions
# --------------------------------------------------------

def parse_year(year_input):
    """
    Parse various year formats to integer year.
    """
    if pd.isna(year_input):
        return np.nan

    x = str(year_input).strip()

    # Handle formats like "Jan-23", "Mar-22"
    if re.match(r"[A-Za-z]{3}-\d{2}", x, re.IGNORECASE):
        try:
            return datetime.strptime(x, "%b-%y").year
        except:
            pass

    # Handle formats like "01/2023", "12/2022"
    if re.match(r"\d{1,2}/\d{4}", x):
        try:
            return datetime.strptime(x, "%m/%Y").year
        except:
            pass

    # Handle just the year
    try:
        year = int(float(x))
        if 1900 <= year <= 2025:  # Reasonable range
            return year
    except:
        pass

    return np.nan

def determine_car_type(brand, model, year):
    """
    Determine if car is luxury, premium, or modern.
    """
    luxury_brands = ['mercedes', 'bmw', 'audi', 'volvo', 'lexus']
    premium_brands = ['volkswagen', 'ford', 'peugeot']

    is_luxury = 1 if brand.lower() in luxury_brands else 0
    is_premium = 1 if brand.lower() in premium_brands else 0
    is_modern = 1 if (2025 - year) <= 5 else 0  # Car from last 5 years

    return is_luxury, is_premium, is_modern

def estimate_co2_from_emission_fuel(emission_class, fuel_type):
    """
    Estimate CO₂ emissions based on emission class and fuel type.
    Returns a normalized value between 1-10 (higher = lower emissions).
    """
    # Base emissions by fuel type (g/km)
    fuel_base_emissions = {
        'electric': 0,
        'hydrogen': 0,
        'plug-in hybrid': 40,
        'hybrid - petrol': 80,
        'hybrid': 90,
        'hybrid - diesel': 110,
        'petrol': 120,
        'gasoline': 125,
        'petrol super': 130,
        'diesel': 140
    }

    # Emission class reduction factors (lower = better emissions)
    emission_reduction = {
        'euro 6e': 0.8,
        'euro 6d-temp-evap': 0.85,
        'euro 6d': 0.9,
        'euro 6d-temp': 0.95,
        'euro 6c': 1.0,
        'euro 6': 1.05,
        'euro 6b': 1.1,
        'euro 5': 1.2,
        'euro 4': 1.5,
        'euro 3': 2.0
    }

    # Get base emissions for fuel type
    if fuel_type.lower() in fuel_base_emissions:
        base_emissions = fuel_base_emissions[fuel_type.lower()]
    else:
        base_emissions = 120  # Default for unknown fuel types

    # Apply emission class factor
    if emission_class.lower() in emission_reduction:
        emissions = base_emissions * emission_reduction[emission_class.lower()]
    else:
        emissions = base_emissions

    # Normalize to 1-10 scale (higher = better/lower emissions)
    normalized = max(1.0, min(10.0, 10 - (emissions / 20)))

    return normalized

def get_filtered_data(brand, model, emission_class, car_age, min_mileage=None, max_mileage=None, original_dataset_path="FilterCars.csv"):
    """
    Filter the original dataset based on user inputs and return filtered DataFrame.
    """
    try:
        # Check if original dataset exists
        if not os.path.exists(original_dataset_path):
            st.warning(f"Original dataset not found at: {original_dataset_path}")
            return None
        
        # Load original dataset
        df = pd.read_csv("FilterCars.csv")
        
        # Apply filters
        # Filter by brand
        if brand:
            df = df[df['general_information__brand'].str.lower() == brand.lower()]
        
        # Filter by model
        if model:
            df = df[df['general_information__model'].str.lower() == model.lower()]
        
        # Filter by emission class
        if emission_class:
            df = df[df['energy_consumption__emission_class'].str.lower() == emission_class.lower()]
        
        # Filter by car age (exact match)
        if car_age is not None:
            df = df[df['car_age'] == car_age]
        
        # Filter by mileage range
        if min_mileage is not None:
            df = df[df['mileage'] >= min_mileage]
        
        if max_mileage is not None:
            df = df[df['mileage'] <= max_mileage]
        
        return df
    
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        return None

def create_download_link(df, filename="filtered_data.csv"):
    """
    Create a download link for a DataFrame as CSV.
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download Filtered CSV File</a>'
    return href

# --------------------------------------------------------
# 2️⃣ Initialize Streamlit App
# --------------------------------------------------------

st.set_page_config(page_title="Car Price Prediction System", page_icon="🚗", layout="wide")

st.title("🚗 CAR PRICE PREDICTION SYSTEM")
st.markdown("---")

# Check if model files exist
model_files_exist = all([
    os.path.exists("models/scaler.joblib"),
    os.path.exists("models/features.joblib"),
    os.path.exists("output.json")
])

if not model_files_exist:
    st.error("❌ Required model files not found!")
    st.info("Please ensure you have the following files:")
    st.write("- models/scaler.joblib (for feature scaling)")
    st.write("- models/features.joblib (feature names in correct order)")
    st.write("- output.json (for model rankings)")
    st.stop()

# Check if original dataset exists for filtering feature
original_dataset_exists = os.path.exists("FilterCars.csv")
if not original_dataset_exists:
    st.warning("⚠️ Original dataset not found. CSV filtering feature will be disabled.")
    st.info("To enable CSV filtering, please place 'original_dataset.csv' in the same directory.")

# --------------------------------------------------------
# 3️⃣ Load Model Artifacts and Data
# --------------------------------------------------------

@st.cache_resource
def load_models():
    """Load all model artifacts once"""
    try:
        # Fix numpy compatibility issue
        try:
            import numpy.core
            import sys
            sys.modules['numpy._core'] = numpy.core
        except:
            pass
        
        # Load trained model artifacts
        features = joblib.load("models/features.joblib")
        scaler = joblib.load("models/scaler.joblib")
        
        # Load the correct model file (check which one exists)
        model_files = ["cat_model.joblib", "models/best_model.joblib", "best_model.joblib"]
        best_model = None
        for model_file in model_files:
            if os.path.exists(model_file):
                best_model = joblib.load(model_file)
                st.info(f"Loaded model from: {model_file}")
                break
        
        if best_model is None:
            st.error("No model file found. Please ensure cat_model.joblib or best_model.joblib exists.")
            st.stop()
        
        # Load model rankings from JSON file
        with open("output.json", "r") as f:
            model_rankings = json.load(f)
        
        return features, scaler, best_model, model_rankings, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False

# Load models with progress
with st.spinner("Loading ML models and data..."):
    features, scaler, best_model, model_rankings, loaded = load_models()

if not loaded:
    st.error("Failed to load models. Please check the console for errors.")
    st.stop()

st.success("✅ ML Models loaded successfully!")

# --------------------------------------------------------
# 4️⃣ Brand and Model Ranking Mappings
# --------------------------------------------------------

# Brand ranking (based on typical market perception) - Scale 1-10
brand_ranking = {
    'mercedes': 10, 'bmw': 9, 'audi': 8, 'volvo': 7,
    'volkswagen': 6, 'ford': 5, 'peugeot': 4, 'kia': 3
}

# Emission class ranking (higher = better) - Scale 1-10
emission_ranking = {
    'euro 6e': 10, 'euro 6d-temp-evap': 9, 'euro 6d': 8, 'euro 6d-temp': 7,
    'euro 6c': 6, 'euro 6': 5, 'euro 6b': 4, 'euro 5': 3, 'euro 6 ea': 5,
    'euro 4': 2, 'euro 3': 1
}

# Fuel type ranking (based on efficiency/environmental impact) - Scale 1-10
fuel_ranking = {
    'electric': 10, 'hybrid - petrol': 9, 'hybrid': 8, 'petrol': 7,
    'diesel': 6, 'petrol super': 5, 'hybrid - diesel': 8,
    'gasoline': 7, 'plug-in hybrid': 9, 'hydrogen': 10
}

# Transmission is automatic by default
transmission_ranking = 8  # Automatic

# --------------------------------------------------------
# 5️⃣ User Input Collection - BRAND SELECTION
# --------------------------------------------------------

st.header("1️⃣ Brand Selection")

available_brands = list(brand_ranking.keys())
brand = st.selectbox(
    "Select Brand",
    available_brands,
    format_func=lambda x: x.capitalize()
)

if brand:
    brand_rank = brand_ranking[brand]
    st.success(f"✓ Selected: {brand.capitalize()} (Rank: {brand_rank})")

# --------------------------------------------------------
# 5.5️⃣ Model Selection (IMPROVED - No manual typing)
# --------------------------------------------------------

st.header("2️⃣ Model Selection")

# Get all models for the selected brand
available_models = list(model_rankings.get(brand, {}).keys())

if not available_models:
    st.error(f"No models found for brand '{brand}' in database.")
    st.info("Available brands in database:")
    for b in model_rankings.keys():
        st.write(f"- {b.capitalize()}")
    st.stop()
else:
    # Sort models alphabetically
    available_models_sorted = sorted(available_models)
    
    # Let user select model from dropdown
    model_name = st.selectbox(
        f"Select {brand.capitalize()} model:",
        options=available_models_sorted,
        format_func=lambda x: x.title(),
        key="model_select"
    )
    
    # Once selected, get its rank
    if model_name in model_rankings[brand]:
        model_rank = model_rankings[brand][model_name]
        st.success(f"✓ Selected model: {model_name.title()} (Rank: {model_rank:.4f})")
    else:
        st.error(f"Model '{model_name}' not found in rankings for brand '{brand}'")
        st.stop()

# --------------------------------------------------------
# 6️⃣ Vehicle Details
# --------------------------------------------------------

st.header("3️⃣ Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    year_input = st.text_input("Vehicle history year (e.g., 2023, Jan-23, 01/2023):", key="year_input")
    
with col2:
    mileage = st.number_input("Mileage (in km):", min_value=100, max_value=900000, value=50000, step=1000)

# Parse year
vehicle_year = None
if year_input:
    vehicle_year = parse_year(year_input)
    if not pd.isna(vehicle_year):
        car_age = 2025 - vehicle_year
        if 0 <= car_age <= 50:
            st.info(f"✓ Year: {int(vehicle_year)} (Age: {car_age} years)")
        else:
            st.error(f"Car age {car_age} years seems unrealistic (should be 0-50 years)")
            st.stop()
    else:
        st.error("Could not parse year. Please enter a valid year.")
        st.stop()
else:
    st.warning("Please enter vehicle year")
    st.stop()

# --------------------------------------------------------
# 7️⃣ Emission and Fuel Details
# --------------------------------------------------------

st.header("4️⃣ Emission & Fuel Details")

# Emission class selection
st.subheader("Emission Class")
available_emissions = list(emission_ranking.keys())

# Use selectbox instead of buttons for simplicity
emission_choice = st.selectbox(
    "Select Emission Class:",
    options=available_emissions,
    format_func=lambda x: x.upper()
)

if emission_choice:
    emission_class = emission_choice
    emission_rank = emission_ranking[emission_class]
    st.success(f"✓ Selected: {emission_class.upper()} (Rank: {emission_rank})")

# Fuel type selection
st.subheader("Fuel Type")
available_fuels = list(fuel_ranking.keys())

# Use selectbox instead of buttons for simplicity
fuel_choice = st.selectbox(
    "Select Fuel Type:",
    options=available_fuels,
    format_func=lambda x: x.title()
)

if fuel_choice:
    fuel_type = fuel_choice
    fuel_rank = fuel_ranking[fuel_type]
    st.success(f"✓ Selected: {fuel_type.title()} (Rank: {fuel_rank})")

# --------------------------------------------------------
# 8️⃣ CO₂ Emissions Input
# --------------------------------------------------------

st.header("5️⃣ CO₂ Emissions Details")

# Estimate CO₂ based on emission class and fuel type
if 'emission_class' in locals() and 'fuel_type' in locals():
    estimated_co2 = estimate_co2_from_emission_fuel(emission_class, fuel_type)
    st.info(f"Based on {emission_class.upper()} emission class and {fuel_type.title()} fuel type:")
    st.info(f"Estimated CO₂ emissions score: {estimated_co2:.2f} (1-10 scale, higher = lower emissions)")

# CO2 input method
co2_method = st.radio(
    "Choose CO₂ emissions input method:",
    [
        "Use estimated value based on emission class and fuel type",
        "Enter specific CO₂ emissions value (1-10 scale)",
        "Enter actual CO₂ emissions in g/km"
    ]
)

co2_emissions = None

if co2_method == "Use estimated value based on emission class and fuel type":
    if 'estimated_co2' in locals():
        co2_emissions = estimated_co2
        st.success(f"✓ Using estimated CO₂ emissions: {co2_emissions:.2f}")
    else:
        st.warning("Please select emission class and fuel type first")
        st.stop()

elif co2_method == "Enter specific CO₂ emissions value (1-10 scale)":
    co2_input = st.slider("Enter CO₂ emissions (1-10 scale):", 1.0, 10.0, 5.0, 0.1)
    co2_emissions = float(co2_input)
    st.success(f"✓ CO₂ emissions set to: {co2_emissions:.2f}")

elif co2_method == "Enter actual CO₂ emissions in g/km":
    co2_gkm = st.number_input("Enter CO₂ emissions in g/km (e.g., 120 for petrol car):", 
                             min_value=0, max_value=300, value=120)
    co2_value = float(co2_gkm)
    # Convert g/km to 1-10 scale
    co2_emissions = max(1.0, min(10.0, 10 - (co2_value / 20)))
    st.success(f"✓ {co2_value} g/km converted to score: {co2_emissions:.2f}")

# --------------------------------------------------------
# 9️⃣ Other Details - INCLUDING WARRANTY INPUT
# --------------------------------------------------------

st.header("6️⃣ Other Details")

# Warranty Input
st.subheader("Warranty Period")
warranty = st.slider(
    "Select warranty period (in months):",
    min_value=0,
    max_value=60,  # 5 years maximum
    value=12,  # Default 1 year
    step=1,
    help="Warranty period remaining for the vehicle"
)

# Display warranty in years and months
years = warranty // 12
months = warranty % 12

if warranty == 0:
    st.info(f"✓ No warranty remaining")
elif warranty < 12:
    st.info(f"✓ Warranty: {warranty} months")
else:
    if months > 0:
        st.info(f"✓ Warranty: {years} years and {months} months ({warranty} months total)")
    else:
        st.info(f"✓ Warranty: {years} years ({warranty} months total)")

# Common warranty options for quick selection
st.subheader("Quick Warranty Options")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("No Warranty"):
        warranty = 0
        st.rerun()

with col2:
    if st.button("6 Months"):
        warranty = 6
        st.rerun()

with col3:
    if st.button("1 Year"):
        warranty = 12
        st.rerun()

with col4:
    if st.button("2 Years"):
        warranty = 24
        st.rerun()

with col5:
    if st.button("3 Years"):
        warranty = 36
        st.rerun()

# Transmission is automatic by default
st.subheader("Transmission")
transmission_options = {
    "Automatic": 8,
    "Manual": 6,
    "Semi-automatic": 7
}

selected_transmission = st.selectbox(
    "Select transmission type:",
    options=list(transmission_options.keys())
)

transmission_ranking = transmission_options[selected_transmission]
st.info(f"✓ Transmission set to: {selected_transmission} (Rank: {transmission_ranking})")

# Determine car type automatically
if 'brand' in locals() and 'model_name' in locals() and vehicle_year:
    is_luxury, is_premium, is_modern = determine_car_type(brand, model_name, vehicle_year)
    st.info(f"✓ Car type determined: Luxury={is_luxury}, Premium={is_premium}, Modern={is_modern}")
else:
    is_luxury, is_premium, is_modern = 0, 0, 0

# --------------------------------------------------------
# 🔟 CSV FILTERING OPTIONS
# --------------------------------------------------------

st.header("📊 CSV Filtering Options")

# Only show filtering options if original dataset exists
if original_dataset_exists:
    st.subheader("Filter Original Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Filter Settings:**")
        # Mileage range filter
        st.markdown("**Mileage Range (Optional):**")
        use_mileage_filter = st.checkbox("Apply mileage filter", value=False)
        
        if use_mileage_filter:
            min_mileage = st.number_input("Minimum mileage (km):", min_value=0, value=40000, step=1000)
            max_mileage = st.number_input("Maximum mileage (km):", min_value=0, value=100000, step=1000)
        else:
            min_mileage = None
            max_mileage = None
        
        # Exact age filter
        exact_age_filter = st.checkbox("Filter by exact age", value=True)
        if exact_age_filter:
            filter_age = car_age
            st.info(f"Will filter by exact age: {filter_age} years")
        else:
            filter_age = None
    
    with col2:
        st.markdown("**Filter Preview:**")
        st.info(f"**Brand:** {brand.capitalize()}")
        st.info(f"**Model:** {model_name.title()}")
        st.info(f"**Emission Class:** {emission_class.upper()}")
        st.info(f"**Car Age:** {car_age} years")
        if use_mileage_filter:
            st.info(f"**Mileage Range:** {min_mileage:,} - {max_mileage:,} km")
        
        # Show what will be filtered
        filter_criteria = []
        filter_criteria.append(f"Brand = '{brand.capitalize()}'")
        filter_criteria.append(f"Model = '{model_name.title()}'")
        filter_criteria.append(f"Emission Class = '{emission_class.upper()}'")
        filter_criteria.append(f"Car Age = {car_age} years")
        if use_mileage_filter:
            filter_criteria.append(f"Mileage ≥ {min_mileage:,} km")
            if max_mileage:
                filter_criteria.append(f"Mileage ≤ {max_mileage:,} km")
        
        st.markdown("**Filter Criteria:**")
        for criterion in filter_criteria:
            st.write(f"• {criterion}")
    
    # Add a toggle to enable/disable filtering feature
    enable_filtering = st.checkbox("Enable CSV filtering feature", value=True)
    
else:
    enable_filtering = False
    st.warning("CSV filtering is disabled because 'original_dataset.csv' was not found.")

# --------------------------------------------------------
# 🔟 Prepare Test Car Data
# --------------------------------------------------------

if st.button("🚀 PREDICT CAR PRICE", type="primary", use_container_width=True):
    
    # Validate all required inputs
    required_fields = [brand, model_name, vehicle_year, co2_emissions is not None, warranty is not None]
    if not all(required_fields):
        st.error("Please fill in all required fields!")
        st.stop()
    
    with st.spinner("Processing input data..."):
        
        # Prepare test car data
        test_car = {
            'brand_rank': float(brand_rank),
            'model_rank': float(model_rank),
            'vehicle_history__year': float(vehicle_year),
            'car_age': float(car_age),
            'mileage': float(mileage),
            'emission_rank': float(emission_rank),
            'energy_consumption__co2_emissions': float(co2_emissions),
            'general_information__warranty': int(warranty),
            'fuel_rank': float(fuel_rank),
            'transmission_rank': float(transmission_ranking),
            'is_luxury': int(is_luxury),
            'is_premium': int(is_premium),
            'is_modern': int(is_modern)
        }

        # Feature Engineering
        df = pd.DataFrame([test_car])

        # 1. Basic mileage/age features
        df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 0.5)
        df['log_mileage'] = np.log1p(df['mileage'])
        df['log_mileage_per_year'] = np.log1p(df['mileage_per_year'])
        df['age_squared'] = df['car_age'] ** 2
        df['mileage_squared'] = df['mileage'] ** 2
        df['age_mileage_ratio'] = df['car_age'] / (df['mileage'] + 1)
        df['age_mileage_interaction'] = df['car_age'] * df['log_mileage']

        # 2. Brand/model features
        df['brand_model_product'] = df['brand_rank'] * df['model_rank']
        df['brand_model_ratio'] = df['brand_rank'] / (df['model_rank'] + 1e-6)
        df['brand_model_diff'] = df['brand_rank'] - df['model_rank']
        df['brand_rank_sq'] = df['brand_rank'] ** 2
        df['model_rank_sq'] = df['model_rank'] ** 2

        # 3. Emission features
        df['emission_brand_interaction'] = df['emission_rank'] * df['brand_rank']
        df['emission_model_interaction'] = df['emission_rank'] * df['model_rank']
        df['emission_age_interaction'] = df['emission_rank'] / (df['car_age'] + 1)

        # 4. Frequency features - use neutral/default values
        df['brand_count_norm'] = 0.5
        df['model_count_norm'] = 0.5
        df['brand_model_count_norm'] = 0.5

        # Fill missing features with 0
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0

        # Prepare features in correct order
        X = df[features]

        # Scale features
        X_scaled = scaler.transform(X)

        # Make prediction
        with st.spinner("Making prediction..."):
            pred_log = best_model.predict(X_scaled)[0]
            predicted_price = np.expm1(pred_log)

        # Display Results
        st.markdown("---")
        st.markdown("## 🎯 PREDICTION RESULTS")
        st.markdown("---")

        # Display car details
        st.subheader("📋 Car Details:")
        
        details_col1, details_col2 = st.columns(2)
        
        with details_col1:
            st.write(f"**Brand:** {brand.capitalize()} (Rank: {brand_rank})")
            st.write(f"**Model:** {model_name.title()} (Rank: {model_rank:.4f})")
            st.write(f"**Year:** {int(vehicle_year)} (Age: {int(car_age)} years)")
            st.write(f"**Mileage:** {mileage:,.0f} km")
            st.write(f"**Emission:** {emission_class.upper()} (Rank: {emission_rank})")
            st.write(f"**Fuel:** {fuel_type.title()} (Rank: {fuel_rank})")
            
        with details_col2:
            st.write(f"**CO₂ Emissions:** {co2_emissions:.2f} (1-10 scale)")
            st.write(f"**Transmission:** {selected_transmission} (Rank: {transmission_ranking})")
            st.write(f"**Warranty:** {warranty} months")
            if warranty > 0:
                if warranty >= 12:
                    st.write(f"  ({warranty//12} year{'s' if warranty//12 > 1 else ''}" + 
                            (f" {warranty%12} month{'s' if warranty%12 > 1 else ''}" if warranty%12 > 0 else "") + ")")
                else:
                    st.write(f"  ({warranty} month{'s' if warranty > 1 else ''})")
            st.write(f"**Type:** Luxury={is_luxury}, Premium={is_premium}, Modern={is_modern}")

        # Display predicted price
        st.markdown("---")
        st.markdown(f"# 🏷️ Predicted Price: **€{predicted_price:,.2f}**")
        st.markdown("---")

        # Show model type
        model_type = type(best_model).__name__
        st.info(f"**Model used:** {model_type}")

        # Confidence range
        price_range_low = predicted_price * 0.9
        price_range_high = predicted_price * 1.1
        
        st.warning(f"**Estimated Price Range:** €{price_range_low:,.2f} - €{price_range_high:,.2f}")

        # --------------------------------------------------------
        # CSV FILTERING AND DOWNLOAD
        # --------------------------------------------------------
        
        if original_dataset_exists and enable_filtering:
            st.markdown("---")
            st.subheader("📊 Filtered Dataset")
            
            with st.spinner("Filtering original dataset..."):
                # Apply filters based on user inputs
                filtered_df = get_filtered_data(
                    brand=brand,
                    model=model_name,
                    emission_class=emission_class,
                    car_age=car_age if exact_age_filter else None,
                    min_mileage=min_mileage if use_mileage_filter else None,
                    max_mileage=max_mileage if use_mileage_filter else None
                )
                
                if filtered_df is not None and not filtered_df.empty:
                    st.success(f"✅ Found {len(filtered_df)} matching records in the dataset")
                    
                    # Display filtered data preview
                    st.markdown("**Preview of filtered data:**")
                    st.dataframe(filtered_df.head(10), use_container_width=True)
                    
                    # Show basic statistics
                    st.markdown("**Filtered Data Statistics:**")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric("Total Records", len(filtered_df))
                    
                    with stats_col2:
                        if 'price' in filtered_df.columns:
                            avg_price = filtered_df['price'].mean()
                            st.metric("Average Price", f"€{avg_price:,.2f}")
                    
                    with stats_col3:
                        if 'mileage' in filtered_df.columns:
                            avg_mileage = filtered_df['mileage'].mean()
                            st.metric("Average Mileage", f"{avg_mileage:,.0f} km")
                    
                    # Create download link
                    st.markdown("---")
                    st.subheader("📥 Download Filtered Data")
                    
                    # Generate filename based on filters
                    filename = f"filtered_{brand}_{model_name}_{emission_class}_age{car_age}.csv"
                    
                    # Create download button
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv",
                        help="Download the filtered dataset as CSV"
                    )
                    
                    # Also show the direct link
                    st.markdown(create_download_link(filtered_df, filename), unsafe_allow_html=True)
                    
                elif filtered_df is not None and filtered_df.empty:
                    st.warning("No matching records found in the dataset with the specified filters.")
                    st.info("Try relaxing some filter criteria (e.g., remove mileage filter or exact age filter).")
                else:
                    st.error("Could not filter the dataset. Please check if the original dataset is in the correct format.")

        # Save prediction to file
        prediction_data = {
            'brand': brand.capitalize(),
            'model': model_name.title(),
            'year': int(vehicle_year),
            'mileage_km': mileage,
            'emission_class': emission_class.upper(),
            'fuel_type': fuel_type.title(),
            'transmission': selected_transmission,
            'warranty_months': warranty,
            'co2_emissions_score': float(co2_emissions),
            'predicted_price_eur': float(predicted_price),
            'model_type': model_type,
            'prediction_timestamp': datetime.now().isoformat()
        }

        try:
            # Save to JSON file
            if os.path.exists('prediction_history.json'):
                with open('prediction_history.json', 'r') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
                existing_data.append(prediction_data)
            else:
                existing_data = [prediction_data]

            with open('prediction_history.json', 'w') as f:
                json.dump(existing_data, f, indent=2)

            st.success("✓ Prediction saved to 'prediction_history.json'")

        except Exception as e:
            st.error(f"Could not save prediction: {e}")

# --------------------------------------------------------
# Footer
# --------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🚗 Car Price Prediction System • Using Single Best Model</p>
        <p><small>Features engineered to match ensemble model training</small></p>
        <p><small>📊 CSV Filtering: {'Enabled' if original_dataset_exists and enable_filtering else 'Disabled'}</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
