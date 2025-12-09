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
warnings.filterwarnings('ignore')

# Load the mean price mapping
mean_price_mapping = {'audi a1': 22864.602325581396, 'audi a3': 28772.46208869814, 'audi a4': 27838.820422535213, 'audi a5': 39351.75471698113, 'audi a6': 54669.98540145985, 'audi a7': 59742.77419354839, 'audi a8': 64213.333333333336, 'audi q3': 34199.807570977915, 'audi q3 sportback': 36668.982300884956, 'audi q4': 50215.739726027394, 'audi q4 sportback': 50524.6129032258, 'audi q5': 46701.494047619046, 'audi q6 e-tron': 77776.73033707865, 'audi q7': 72948.04166666667, 'audi q8': 72483.11206896552, 'audi rio': 45472.0, 'audi rs': 63567.36448598131, 'audi tt': 34437.895833333336, 'audi up': 33890.0, 'bmw 1 series': 22771.25806451613, 'bmw 2 series': 24738.077922077922, 'bmw 240': 45633.333333333336, 'bmw 3 series': 34735.220183486235, 'bmw 4 series': 41238.89, 'bmw 5 series': 39580.28358208955, 'bmw 6 series': 47098.0, 'bmw 7 series': 47282.958333333336, 'bmw 8 series': 57333.2, 'bmw gla': 67390.0, 'bmw i3': 18607.272727272728, 'bmw i4': 53024.029411764706, 'bmw i7': 82740.66666666667, 'bmw ion': 83885.0, 'bmw ix': 57486.05882352941, 'bmw up': 40653.568181818184, 'bmw x1': 28912.57461024499, 'bmw x2': 28231.104938271605, 'bmw x3': 42576.74893617021, 'bmw x4': 40743.87301587302, 'bmw x5': 61321.86885245902, 'bmw x6': 51863.76190476191, 'bmw x7': 78177.5, 'bmw z4': 42363.26315789474, 'ford b-max': 11985.052631578947, 'ford bronco': 61499.0, 'ford connect': 17490.0, 'ford ecosport': 15662.716981132075, 'ford edge': 21242.571428571428, 'ford explorer': 48057.08, 'ford fiesta': 14407.42, 'ford focus': 19539.11724137931, 'ford focus clipper': 21474.615384615383, 'ford kuga': 24791.71020408163, 'ford mondeo': 15980.387096774193, 'ford mustang': 39282.32, 'ford mustang mach-e': 47074.45238095238, 'ford ranger': 44330.0, 'ford ranger raptor': 57990.0, 'ford tourneo connect': 18962.30769230769, 'ford tourneo courier': 18226.81818181818, 'ford transit': 13463.333333333334, 'ford transit connect': 25339.10769230769, 'ford transit courier': 23581.333333333332, 'ford transit custom': 24370.0, 'ford up': 25400.0, 'ford x4': 72800.0, 'kia ceed': 21393.133333333335, 'kia ceed gt': 22365.666666666668, 'kia ceed sportswagon': 20964.738255033557, 'kia ev3': 44563.8, 'kia ev6': 45210.851851851854, 'kia ev9': 70958.33333333333, 'kia niro': 26448.44827586207, 'kia niro hev': 24174.166666666668, 'kia niro phev': 29156.333333333332, 'kia picanto': 16123.427272727273, 'kia proceed': 22248.837837837837, 'kia rio': 15144.078947368422, 'kia sorento': 41712.444444444445, 'kia soul': 27746.8, 'kia sportage': 28770.94849785408, 'kia stinger': 29974.5, 'kia stonic': 16656.178571428572, 'kia venga': 8435.42857142857, 'mercedes amg gt': 83902.33333333333, 'mercedes c-class': 36586.13972055888, 'mercedes cla': 31577.8959778086, 'mercedes cls': 48184.125, 'mercedes e-class': 47803.1375, 'mercedes g-class': 77755.0, 'mercedes gla': 33342.65079365079, 'mercedes glb': 39221.82142857143, 'mercedes glc': 47813.92424242424, 'mercedes gle': 73339.05084745762, 'mercedes gls': 81563.33333333333, 'mercedes rio': 65457.0, 'mercedes rs': 26480.0, 'mercedes s-class': 77838.65, 'mercedes sl': 29255.73076923077, 'mercedes up': 64085.25, 'peugeot 107': 4904.875, 'peugeot 108': 10631.25, 'peugeot 2008': 19181.745989304814, 'peugeot 206': 3600.0, 'peugeot 207': 5506.125, 'peugeot 208': 17512.039087947884, 'peugeot 3008': 25383.28205128205, 'peugeot 308': 30957.184210526317, 'peugeot 408': 27976.879120879123, 'peugeot 5008': 23090.878787878788, 'peugeot 508': 24522.702479338845, 'peugeot 508 sw': 28850.0, 'peugeot boxer': 24978.18918918919, 'peugeot expert': 30155.18, 'peugeot ion': 8974.5, 'peugeot partner': 17147.617021276597, 'peugeot rcz': 12350.0, 'peugeot traveller': 29776.9, 'volkswagen amarok': 47887.86666666667, 'volkswagen arteon': 32312.363636363636, 'volkswagen caddy': 22629.350515463917, 'volkswagen caddy life': 22432.5, 'volkswagen caddy maxi': 30043.652173913044, 'volkswagen california': 67307.14285714286, 'volkswagen crafter': 37683.83908045977, 'volkswagen golf': 23562.09461663948, 'volkswagen golf gti': 33874.0, 'volkswagen golf plus': 7625.0, 'volkswagen golf r': 37250.0, 'volkswagen golf sportsvan': 13792.111111111111, 'volkswagen golf variant': 20204.219512195123, 'volkswagen ion': 37500.0, 'volkswagen multivan': 56659.637931034486, 'volkswagen passat': 23502.583333333332, 'volkswagen passat variant': 31203.892405063292, 'volkswagen polo': 19738.278177458033, 'volkswagen rio': 17948.090909090908, 'volkswagen t-cross': 22343.724683544304, 'volkswagen t-roc': 26801.402173913044, 'volkswagen taigo': 23776.913705583756, 'volkswagen tiguan': 32609.82339449541, 'volkswagen tiguan allspace': 30860.660194174758, 'volkswagen touareg': 54522.75, 'volkswagen transporter': 32543.991150442478, 'volkswagen tt': 10990.0, 'volkswagen up': 18994.363636363636, 'volvo c40': 42753.0, 'volvo ex90': 92561.1875, 'volvo gle': 33270.0, 'volvo s60': 35771.71875, 'volvo s80': 12990.0, 'volvo s90': 28361.533333333333, 'volvo v40': 13957.655172413793, 'volvo v40 cross country': 14137.4, 'volvo v60': 32748.842105263157, 'volvo v60 cross country': 31977.666666666668, 'volvo v90': 44893.333333333336, 'volvo v90 cross country': 37990.0, 'volvo x3': 39324.88888888889, 'volvo x4': 46706.78571428572, 'volvo xc40': 31384.02332361516, 'volvo xc60': 42411.73964497042, 'volvo xc90': 59935.869047619046}

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

def get_filtered_data(brand, model, emission_class, mileage_limit, car_age, original_dataset_path="FilterCars.csv"):
    """
    Filter the original dataset based on user inputs and return filtered DataFrame.
    Now includes mileage (≤) and age (±1 year) filters.
    """
    try:
        # Check if original dataset exists
        if not os.path.exists(original_dataset_path):
            st.warning(f"Original dataset not found at: {original_dataset_path}")
            return None
        
        # Load original dataset
        df = pd.read_csv(original_dataset_path)
        
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
        
        # Filter by mileage (≤ selected mileage)
        if mileage_limit is not None:
            df = df[df['mileage'] <= mileage_limit]
        
        # Filter by car age (±1 year from current year minus vehicle year)
        if car_age is not None:
            # Calculate age range (±1 year)
            min_age = max(0, car_age - 1)  # Minimum age (0 if car_age is 0)
            max_age = car_age + 1  # Maximum age
            
            # Apply age filter
            df = df[(df['car_age'] >= min_age) & (df['car_age'] <= max_age)]
        
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

def prepare_features_for_cbm(car_data, mean_price_mapping):
    """
    Prepare features in the format expected by the .cbm model.
    Returns a dictionary with one-hot encoded features in correct order.
    """
    # Extract brand from car type (first word)
    car_type_lower = car_data['car_type'].lower()
    brand = car_type_lower.split()[0]
    
    # Get mean price for this car type
    mean_price = mean_price_mapping.get(car_type_lower, 30000)  # Default 30000 if not found
    
    # Get model encoded value - we need to create a numeric encoding for model
    # Simple approach: use mean price normalized
    model_encoded = mean_price / 100000  # Normalize to 0-1 range
    
    # Initialize features dictionary - ORDER IS IMPORTANT!
    # Based on error message and your example, we need to have features in this order:
    features = {
        'mileage': float(car_data['mileage']),
        'energy_consumption__co2_emissions': float(car_data['co2_emissions']),
        'general_information__warranty': int(car_data['warranty']),
        'vehicle_age': int(car_data['car_age']),
        
        # model_encoded should be here based on the error
        'model_encoded': float(model_encoded),
        
        # Transmission - one-hot (these come AFTER model_encoded based on error)
        'transmission_automatic': 0,
        'transmission_manual': 0,
        
        # Emission class - one-hot
        'energy_consumption__emission_class_Euro 5': 0,
        'energy_consumption__emission_class_Euro 6': 0,
        'energy_consumption__emission_class_Euro 6b': 0,
        'energy_consumption__emission_class_Euro 6c': 0,
        'energy_consumption__emission_class_Euro 6d': 0,
        'energy_consumption__emission_class_Euro 6d-TEMP': 0,
        'energy_consumption__emission_class_Euro 6d-TEMP-EVAP': 0,
        'energy_consumption__emission_class_Euro 6e': 0,
        
        # Fuel type - one-hot
        'energy_consumption__fuel_diesel': 0,
        'energy_consumption__fuel_electric': 0,
        'energy_consumption__fuel_hybrid': 0,
        'energy_consumption__fuel_hybrid - diesel': 0,
        'energy_consumption__fuel_hybrid - petrol': 0,
        'energy_consumption__fuel_petrol': 0,
        'energy_consumption__fuel_petrol super': 0,
        
        # Brand - one-hot
        'general_information__brand_audi': 0,
        'general_information__brand_bmw': 0,
        'general_information__brand_ford': 0,
        'general_information__brand_kia': 0,
        'general_information__brand_mercedes': 0,
        'general_information__brand_peugeot': 0,
        'general_information__brand_volkswagen': 0,
        'general_information__brand_volvo': 0,
        
        # Additional feature: mean price for car type
        'car_type_mean_price': mean_price
    }
    
    # Set transmission
    transmission = car_data.get('transmission', 'manual').lower()
    if transmission == 'automatic':
        features['transmission_automatic'] = 1
    else:
        features['transmission_manual'] = 1
    
    # Set emission class
    emission_class = car_data['emission_class'].lower()
    emission_mapping = {
        'euro 5': 'energy_consumption__emission_class_Euro 5',
        'euro 6': 'energy_consumption__emission_class_Euro 6',
        'euro 6b': 'energy_consumption__emission_class_Euro 6b',
        'euro 6c': 'energy_consumption__emission_class_Euro 6c',
        'euro 6d': 'energy_consumption__emission_class_Euro 6d',
        'euro 6d-temp': 'energy_consumption__emission_class_Euro 6d-TEMP',
        'euro 6d-temp-evap': 'energy_consumption__emission_class_Euro 6d-TEMP-EVAP',
        'euro 6e': 'energy_consumption__emission_class_Euro 6e'
    }
    
    for key, feature_name in emission_mapping.items():
        if key in emission_class:
            features[feature_name] = 1
            break
    
    # Set fuel type
    fuel_type = car_data['fuel_type'].lower()
    fuel_mapping = {
        'diesel': 'energy_consumption__fuel_diesel',
        'electric': 'energy_consumption__fuel_electric',
        'hybrid': 'energy_consumption__fuel_hybrid',
        'hybrid - diesel': 'energy_consumption__fuel_hybrid - diesel',
        'hybrid - petrol': 'energy_consumption__fuel_hybrid - petrol',
        'petrol': 'energy_consumption__fuel_petrol',
        'petrol super': 'energy_consumption__fuel_petrol super'
    }
    
    for key, feature_name in fuel_mapping.items():
        if key in fuel_type:
            features[feature_name] = 1
            break
    
    # Set brand
    brand_mapping = {
        'audi': 'general_information__brand_audi',
        'bmw': 'general_information__brand_bmw',
        'ford': 'general_information__brand_ford',
        'kia': 'general_information__brand_kia',
        'mercedes': 'general_information__brand_mercedes',
        'peugeot': 'general_information__brand_peugeot',
        'volkswagen': 'general_information__brand_volkswagen',
        'volvo': 'general_information__brand_volvo'
    }
    
    for key, feature_name in brand_mapping.items():
        if brand.lower().startswith(key):
            features[feature_name] = 1
            break
    
    return features

# --------------------------------------------------------
# 2️⃣ Initialize Streamlit App
# --------------------------------------------------------

st.set_page_config(page_title="Car Price Prediction System", page_icon="🚗", layout="wide")

st.title("🚗 CAR PRICE PREDICTION SYSTEM")
st.markdown("---")

# Check if model files exist
model_files_exist = os.path.exists("catboost_car_price_model.cbm")

if not model_files_exist:
    st.error("❌ Model file not found!")
    st.info("Please ensure you have the following file:")
    st.write("- catboost_car_price_model.cbm (CatBoost model)")
    st.stop()

# Check if original dataset exists for filtering feature
original_dataset_exists = os.path.exists("FilterCars.csv")
if not original_dataset_exists:
    st.warning("⚠️ Original dataset not found. CSV filtering feature will be disabled.")
    st.info("To enable CSV filtering, please place 'FilterCars.csv' in the same directory.")

# --------------------------------------------------------
# 3️⃣ Load Model
# --------------------------------------------------------

@st.cache_resource
def load_catboost_model():
    """Load CatBoost model"""
    try:
        # Try to import CatBoost
        from catboost import CatBoostRegressor
        
        # Load the model
        model = CatBoostRegressor()
        model.load_model("catboost_car_price_model.cbm")
        
        # Try to get feature names from the model
        try:
            feature_names = model.feature_names_
            st.info(f"Model expects {len(feature_names)} features")
            st.info(f"First 10 features: {feature_names[:10]}")
        except:
            pass
        
        return model, True
    except Exception as e:
        st.error(f"Error loading CatBoost model: {e}")
        return None, False

# Load model with progress
with st.spinner("Loading CatBoost model..."):
    catboost_model, loaded = load_catboost_model()

if not loaded:
    st.error("Failed to load model. Please check the console for errors.")
    st.stop()

st.success("✅ CatBoost Model loaded successfully!")

# --------------------------------------------------------
# 4️⃣ User Input Collection - CAR TYPE SELECTION
# --------------------------------------------------------

st.header("1️⃣ Car Type Selection")

# Get all available car types from mean price mapping
available_car_types = sorted(mean_price_mapping.keys())

# Group by brand for better organization
car_types_by_brand = {}
for car_type in available_car_types:
    brand = car_type.split()[0].upper()
    if brand not in car_types_by_brand:
        car_types_by_brand[brand] = []
    car_types_by_brand[brand].append(car_type)

# Display organized selection
selected_brand = st.selectbox(
    "Select Brand:",
    options=sorted(car_types_by_brand.keys()),
    format_func=lambda x: x.title(),
    key="brand_select"
)

if selected_brand:
    # Get models for selected brand
    models_for_brand = car_types_by_brand[selected_brand]
    
    # Create a nicer display format
    display_options = {}
    for model in models_for_brand:
        # Convert "audi a4" to "Audi A4"
        display_name = " ".join(word.title() for word in model.split())
        display_options[display_name] = model
    
    selected_display = st.selectbox(
        f"Select {selected_brand.title()} Model:",
        options=list(display_options.keys()),
        key="model_select"
    )
    
    if selected_display:
        car_type = display_options[selected_display]  # Get the lowercase key
        mean_price = mean_price_mapping.get(car_type, 0)
        st.success(f"✓ Selected: {selected_display}")
        st.info(f"Average market price for this model: €{mean_price:,.2f}")

# --------------------------------------------------------
# 5️⃣ Vehicle Details
# --------------------------------------------------------

st.header("2️⃣ Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    year_input = st.text_input("Manufacturing Year (e.g., 2023, Jan-23, 01/2023):", key="year_input")
    
with col2:
    mileage = st.number_input("Mileage (in km):", min_value=100, max_value=900000, value=50000, step=1000)

# Parse year - UPDATED: Subtract from 2026 for age calculation
vehicle_year = None
if year_input:
    vehicle_year = parse_year(year_input)
    if not pd.isna(vehicle_year):
        # Calculate car age: 2026 minus vehicle year
        current_year = 2026
        car_age = current_year - vehicle_year
        if 0 <= car_age <= 50:
            st.info(f"✓ Year: {int(vehicle_year)} (Age: {car_age} years, calculated from 2026)")
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
# 6️⃣ Emission and Fuel Details
# --------------------------------------------------------

st.header("3️⃣ Emission & Fuel Details")

# Emission class selection
st.subheader("Emission Class")
emission_classes = [
    'Euro 5', 'Euro 6', 'Euro 6b', 'Euro 6c', 'Euro 6d',
    'Euro 6d-TEMP', 'Euro 6d-TEMP-EVAP', 'Euro 6e'
]

emission_class = st.selectbox(
    "Select Emission Class:",
    options=emission_classes,
    index=4,  # Default to Euro 6d
    key="emission_select"
)

st.success(f"✓ Selected: {emission_class}")

# Fuel type selection
st.subheader("Fuel Type")
fuel_types = [
    'diesel', 'electric', 'hybrid', 'hybrid - diesel', 
    'hybrid - petrol', 'petrol', 'petrol super'
]

fuel_type = st.selectbox(
    "Select Fuel Type:",
    options=fuel_types,
    index=0,  # Default to diesel
    key="fuel_select"
)

st.success(f"✓ Selected: {fuel_type.title()}")

# --------------------------------------------------------
# 7️⃣ CO₂ Emissions Input
# --------------------------------------------------------

st.header("4️⃣ CO₂ Emissions Details")

# CO2 emissions in g/km
co2_emissions = st.number_input("Enter CO₂ emissions in g/km:", 
                               min_value=0, max_value=300, value=120)
st.success(f"✓ CO₂ emissions set to: {co2_emissions} g/km")

# --------------------------------------------------------
# 8️⃣ Other Details
# --------------------------------------------------------

st.header("5️⃣ Other Details")

# Warranty Input
st.subheader("Warranty Period")
warranty = st.slider(
    "Select warranty period (in months):",
    min_value=0,
    max_value=60,
    value=12,
    step=1,
    help="Warranty period remaining for the vehicle"
)

# Display warranty in years and months
if warranty == 0:
    st.info(f"✓ No warranty remaining")
elif warranty < 12:
    st.info(f"✓ Warranty: {warranty} months")
else:
    years = warranty // 12
    months = warranty % 12
    if months > 0:
        st.info(f"✓ Warranty: {years} years and {months} months ({warranty} months total)")
    else:
        st.info(f"✓ Warranty: {years} years ({warranty} months total)")

# Quick warranty options
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

# Transmission selection
st.subheader("Transmission")
transmission_options = {
    "Automatic": "automatic",
    "Manual": "manual"
}

selected_transmission = st.radio(
    "Select transmission type:",
    options=list(transmission_options.keys()),
    horizontal=True
)

transmission = transmission_options[selected_transmission]
st.info(f"✓ Transmission set to: {selected_transmission}")

# --------------------------------------------------------
# 9️⃣ CSV FILTERING OPTIONS - UPDATED
# --------------------------------------------------------

st.header("📊 CSV Filtering Options")

# Enable filtering by default if dataset exists
enable_filtering = original_dataset_exists

if enable_filtering:
    st.subheader("Filter Original Dataset")
    
    # Extract brand and model from car_type
    brand_filter = car_type.split()[0]
    model_filter = " ".join(car_type.split()[1:])
    
    # Apply filters based on user inputs
    filtered_df = get_filtered_data(
        brand=brand_filter,
        model=model_filter,
        emission_class=emission_class,
        mileage_limit=mileage,  # Use selected mileage as limit
        car_age=car_age  # Use calculated car age
    )
    
    if filtered_df is not None and not filtered_df.empty:
        st.success(f"✅ Found {len(filtered_df)} matching records")
        
        # Display filter criteria
        st.info(f"**Filter Criteria:**")
        st.info(f"• **Brand:** {brand_filter.title()}")
        st.info(f"• **Model:** {model_filter.title()}")
        st.info(f"• **Emission Class:** {emission_class}")
        st.info(f"• **Maximum Mileage:** ≤ {mileage:,} km")
        st.info(f"• **Car Age:** {car_age} years (±1 year)")
        
        # Display filtered data preview
        st.markdown("### Preview of Filtered Data")
        
        # Display complete dataframe with all features (max 10 rows)
        st.dataframe(filtered_df.head(10), use_container_width=True)
        
        # Show basic statistics
        st.markdown("### Filtered Data Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Total Records", len(filtered_df))
        
        with stats_col2:
            if 'price' in filtered_df.columns:
                avg_price = filtered_df['price'].mean()
                st.metric("Average Price", f"€{avg_price:,.2f}")
        
        with stats_col3:
            if 'mileage' in filtered_df.columns:
                avg_mileage = filtered_df['mileage'].mean()
                st.metric("Avg Mileage", f"{avg_mileage:,.0f} km")
        
        with stats_col4:
            if 'car_age' in filtered_df.columns:
                avg_age = filtered_df['car_age'].mean()
                st.metric("Avg Age", f"{avg_age:.1f} years")
        
        # Age distribution info
        if 'car_age' in filtered_df.columns:
            age_min = filtered_df['car_age'].min()
            age_max = filtered_df['car_age'].max()
            st.info(f"Age range in filtered data: {age_min} to {age_max} years")
        
        # Create download link
        st.markdown("---")
        st.subheader("📥 Download Filtered Data")
        
        # Generate filename based on filters
        filename = f"filtered_{brand_filter}_{model_filter.replace(' ', '_')}_{emission_class.replace(' ', '_')}_mileage_to_{mileage}_age_{car_age}_±1.csv"
        
        # Create download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Filtered CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            help="Download the complete filtered dataset as CSV"
        )
        
    elif filtered_df is not None and filtered_df.empty:
        st.warning(f"No matching records found for:")
        st.warning(f"- Brand: {brand_filter.title()}")
        st.warning(f"- Model: {model_filter.title()}")
        st.warning(f"- Emission Class: {emission_class}")
        st.warning(f"- Maximum Mileage: ≤ {mileage:,} km")
        st.warning(f"- Car Age: {car_age} years (±1 year)")
        st.info("Try relaxing the mileage or age filters.")
    else:
        st.error("Could not filter the dataset. Please check if the original dataset is in the correct format.")
else:
    st.warning("CSV filtering is disabled because 'FilterCars.csv' was not found.")

# --------------------------------------------------------
# 🔟 PREDICTION
# --------------------------------------------------------

if st.button("🚀 PREDICT CAR PRICE", type="primary", use_container_width=True):
    
    # Validate all required inputs
    if not all([car_type, vehicle_year, co2_emissions is not None, warranty is not None]):
        st.error("Please fill in all required fields!")
        st.stop()
    
    with st.spinner("Processing input data..."):
        
        # Prepare car data - UPDATED: Use car_age calculated from 2026
        car_data = {
            'car_type': car_type,
            'mileage': mileage,
            'co2_emissions': co2_emissions,
            'warranty': warranty,
            'car_age': car_age,  # This is already calculated as 2026 - vehicle_year
            'emission_class': emission_class,
            'fuel_type': fuel_type,
            'transmission': transmission
        }
        
        # Prepare features for CatBoost model
        features_dict = prepare_features_for_cbm(car_data, mean_price_mapping)
        
        # Convert to DataFrame
        df_features = pd.DataFrame([features_dict])
        
        # Debug: Show features
        st.info(f"Prepared {len(features_dict)} features for prediction")
        
        # Check if model_encoded is in features
        if 'model_encoded' not in features_dict:
            st.error("model_encoded feature is missing! This is required by the model.")
            st.stop()
        
        # Make prediction
        with st.spinner("Making prediction..."):
            try:
                predicted_price = catboost_model.predict(df_features)[0]
                
                # Ensure price is positive
                predicted_price = max(predicted_price, 1000)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.error("This usually means the features are not in the correct order or format.")
                st.info("Try checking the model's expected feature names and order.")
                st.stop()
        
        # Display Results
        st.markdown("---")
        st.markdown("## 🎯 PREDICTION RESULTS")
        st.markdown("---")
        
        # Display car details
        st.subheader("📋 Car Details:")
        
        brand_from_type = car_type.split()[0].title()
        model_from_type = " ".join(word.title() for word in car_type.split()[1:])
        
        details_col1, details_col2 = st.columns(2)
        
        with details_col1:
            st.write(f"**Car Type:** {brand_from_type} {model_from_type}")
            st.write(f"**Mean Market Price:** €{mean_price_mapping.get(car_type, 0):,.2f}")
            st.write(f"**Manufacturing Year:** {int(vehicle_year)}")
            st.write(f"**Vehicle Age (2026 - Year):** {int(car_age)} years")
            st.write(f"**Mileage:** {mileage:,.0f} km")
            
        with details_col2:
            st.write(f"**Emission Class:** {emission_class}")
            st.write(f"**Fuel Type:** {fuel_type.title()}")
            st.write(f"**CO₂ Emissions:** {co2_emissions} g/km")
            st.write(f"**Transmission:** {selected_transmission}")
            st.write(f"**Warranty:** {warranty} months")
            if warranty > 0:
                if warranty >= 12:
                    st.write(f"  ({warranty//12} year{'s' if warranty//12 > 1 else ''}" + 
                            (f" {warranty%12} month{'s' if warranty%12 > 1 else ''}" if warranty%12 > 0 else "") + ")")
                else:
                    st.write(f"  ({warranty} month{'s' if warranty > 1 else ''})")
        
        # Display predicted price
        st.markdown("---")
        st.markdown(f"# 🏷️ Predicted Price: **€{predicted_price:,.2f}**")
        st.markdown("---")
        
        # Show model type
        st.info(f"**Model used:** CatBoost Regressor")
        
        # Confidence range
        price_range_low = predicted_price * 0.9
        price_range_high = predicted_price * 1.1
        
        st.warning(f"**Estimated Price Range:** €{price_range_low:,.2f} - €{price_range_high:,.2f}")
        
        # Show comparison with filtered data average if available
        if enable_filtering and filtered_df is not None and not filtered_df.empty and 'price' in filtered_df.columns:
            avg_price = filtered_df['price'].mean()
            price_diff = ((predicted_price - avg_price) / avg_price) * 100
            
            st.markdown("---")
            st.subheader("📊 Comparison with Filtered Dataset")
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.metric("Filtered Dataset Avg", f"€{avg_price:,.2f}")
            
            with comp_col2:
                st.metric("Predicted Price", f"€{predicted_price:,.2f}")
            
            with comp_col3:
                st.metric("Difference", f"{price_diff:+.1f}%", 
                         delta=f"{price_diff:+.1f}%", 
                         delta_color="normal" if abs(price_diff) < 10 else "inverse")
            
            # Show filtered dataset age range
            if 'car_age' in filtered_df.columns:
                age_min = filtered_df['car_age'].min()
                age_max = filtered_df['car_age'].max()
                st.info(f"Filtered dataset age range: {age_min} to {age_max} years (±1 year from {car_age} years)")
        
        # Save prediction to file
        prediction_data = {
            'car_type': car_type,
            'brand': brand_from_type,
            'model': model_from_type,
            'year': int(vehicle_year),
            'mileage_km': mileage,
            'emission_class': emission_class,
            'fuel_type': fuel_type,
            'transmission': selected_transmission,
            'warranty_months': warranty,
            'co2_emissions_gkm': float(co2_emissions),
            'predicted_price_eur': float(predicted_price),
            'mean_market_price': float(mean_price_mapping.get(car_type, 0)),
            'vehicle_age_2026_minus_year': int(car_age),
            'model_type': 'CatBoost',
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
        <p>🚗 Car Price Prediction System • Using CatBoost Model</p>
        <p><small>Features engineered with one-hot encoding</small></p>
        <p><small>📊 CSV Filtering: {'Enabled' if enable_filtering else 'Disabled'}</small></p>
        <p><small>🚨 Age calculation: 2026 - Manufacturing Year</small></p>
        <p><small>📈 Filters: Mileage ≤ selected, Age ±1 year</small></p>
    </div>
    """,
    unsafe_allow_html=True
)