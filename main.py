# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
import re
import json
import os
from datetime import datetime
from difflib import SequenceMatcher
warnings.filterwarnings('ignore')

# --------------------------------------------------------
# 1️⃣ Helper Functions
# --------------------------------------------------------

def find_best_match(user_input, options_list, threshold=0.7):
    """
    Find the best match for user input from a list of options.
    Returns the best match if similarity >= threshold, else None.
    """
    user_input = user_input.lower().strip()
    best_match = None
    best_score = 0

    for option in options_list:
        option_lower = option.lower().strip()
        score = SequenceMatcher(None, user_input, option_lower).ratio()
        if score > best_score:
            best_score = score
            best_match = option

    return best_match if best_score >= threshold else None

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
    # 0 g/km = 10, 200+ g/km = 1
    normalized = max(1.0, min(10.0, 10 - (emissions / 20)))

    return normalized

# --------------------------------------------------------
# 2️⃣ Initialize Streamlit App
# --------------------------------------------------------

st.set_page_config(page_title="Car Price Prediction System", page_icon="🚗", layout="wide")

st.title("🚗 CAR PRICE PREDICTION SYSTEM")
st.markdown("---")

# Check if model files exist - SIMPLIFIED
model_files_exist = all([
    os.path.exists("models\extra.joblib"),
    os.path.exists("models/scaler.joblib"),
    os.path.exists("models/features.joblib"),
    os.path.exists("output.json")
])

if not model_files_exist:
    st.error("❌ Required model files not found!")
    st.info("Please ensure you have the following files:")
    st.write("- models/best_model.joblib (single trained model)")
    st.write("- models/scaler.joblib (for feature scaling)")
    st.write("- models/features.joblib (feature names in correct order)")
    st.write("- output.json (for model rankings)")
    st.stop()

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
        best_model = joblib.load("cat_model.joblib")
        
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
    f"Select Brand (available: {', '.join(available_brands)})",
    available_brands,
    format_func=lambda x: x.capitalize()
)

if brand:
    brand_rank = brand_ranking[brand]
    st.success(f"✓ Selected: {brand.capitalize()} (Rank: {brand_rank})")

# --------------------------------------------------------
# 5.5️⃣ Display ALL available models for the selected brand
# --------------------------------------------------------

st.header("2️⃣ Model Selection")

# Get all models for the selected brand
available_models = list(model_rankings.get(brand, {}).keys())

if not available_models:
    st.warning(f"No models found for brand '{brand}' in database.")
    model_rank = 1.0  # Default value
    model_name = st.text_input("Enter model name:").strip().lower()
else:
    # Display all available models in an expander
    with st.expander(f"📋 View all {len(available_models)} available {brand.capitalize()} models"):
        # Sort models alphabetically
        available_models_sorted = sorted(available_models)
        
        # Display in columns
        cols_per_row = 4
        models_chunks = [available_models_sorted[i:i + cols_per_row] 
                        for i in range(0, len(available_models_sorted), cols_per_row)]
        
        for chunk in models_chunks:
            cols = st.columns(cols_per_row)
            for i, model in enumerate(chunk):
                with cols[i]:
                    st.text(model.title())

    # Ask for model name
    st.subheader("Enter Model Name")
    st.info(f"Examples: {', '.join(available_models[:5])}")
    
    model_input = st.text_input(f"Enter {brand.capitalize()} model name:", key="model_input")
    
    if model_input:
        model_input = model_input.strip().lower()
        
        # Special handling for common model name variations
        model_variations = {
            # Mercedes variations
            'a class': 'a-class', 'a klasse': 'a-class', 'a-klasse': 'a-class',
            'b class': 'b-class', 'b klasse': 'b-class', 'b-klasse': 'b-class',
            'c class': 'c-class', 'c klasse': 'c-class', 'c-klasse': 'c-class',
            'e class': 'e-class', 'e klasse': 'e-class', 'e-klasse': 'e-class',
            's class': 's-class', 's klasse': 's-class', 's-klasse': 's-class',

            # BMW variations
            '1 series': '1 series', '1 serie': '1 series', '1-reeks': '1 series',
            '2 series': '2 series', '2 serie': '2 series', '2-reeks': '2 series',
            '3 series': '3 series', '3 serie': '3 series', '3-reeks': '3 series',
            '4 series': '4 series', '4 serie': '4 series', '4-reeks': '4 series',
            '5 series': '5 series', '5 serie': '5 series', '5-reeks': '5 series',
            '7 series': '7 series', '7 serie': '7 series', '7-reeks': '7 series',
            'x1': 'x1', 'x2': 'x2', 'x3': 'x3', 'x4': 'x4', 'x5': 'x5', 'x6': 'x6',

            # Audi variations
            'a1': 'a1', 'a3': 'a3', 'a4': 'a4', 'a5': 'a5', 'a6': 'a6', 'a7': 'a7', 'a8': 'a8',
            'q2': 'q2', 'q3': 'q3', 'q4': 'q4', 'q5': 'q5', 'q7': 'q7', 'q8': 'q8',

            # Volkswagen variations
            'golf': 'golf', 'polo': 'polo', 'passat': 'passat', 'tiguan': 'tiguan',
            't-roc': 't-roc', 't-cross': 't-cross', 'taigo': 'taigo',

            # Ford variations
            'fiesta': 'fiesta', 'focus': 'focus', 'kuga': 'kuga', 'mustang': 'mustang',
            'puma': 'puma', 'mondeo': 'mondeo',

            # Volvo variations
            'xc40': 'xc40', 'xc60': 'xc60', 'xc90': 'xc90',
            's60': 's60', 's90': 's90', 'v60': 'v60', 'v90': 'v90',

            # Peugeot variations
            '208': '208', '2008': '2008', '308': '308', '3008': '3008',
            '508': '508', '5008': '5008',

            # Kia variations
            'picanto': 'picanto', 'rio': 'rio', 'ceed': 'ceed', 'proceed': 'proceed',
            'sportage': 'sportage', 'niro': 'niro', 'stonic': 'stonic', 'xceed': 'xceed'
        }

        # Check for common variations first
        normalized_input = model_input.lower()
        if normalized_input in model_variations:
            normalized_input = model_variations[normalized_input]

        best_match = find_best_match(normalized_input, available_models, threshold=0.7)

        if best_match:
            model_name = best_match
            model_rank = model_rankings[brand][model_name]
            st.success(f"✓ Recognized as: {model_name.title()} (Rank: {model_rank:.4f})")
        else:
            st.error(f"Model '{model_input}' not recognized.")
            
            # Show closest matches
            st.subheader("Closest matches found:")
            
            # Find and show top 5 closest matches
            similarity_scores = []
            for option in available_models:
                score = SequenceMatcher(None, normalized_input, option.lower()).ratio()
                similarity_scores.append((option, score))

            # Sort by similarity score (highest first)
            similarity_scores.sort(key=lambda x: x[1], reverse=True)

            # Show top 5 matches
            for i, (option, score) in enumerate(similarity_scores[:5]):
                st.write(f"{i+1}. {option.title()} (similarity: {score:.1%})")
            
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
if year_input:
    vehicle_year = parse_year(year_input)
    if not pd.isna(vehicle_year):
        car_age = 2025 - vehicle_year
        if 0 <= car_age <= 50:  # Reasonable car age
            st.info(f"✓ Year: {vehicle_year} (Age: {car_age} years)")
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

# Create columns for better display
emission_cols = st.columns(3)
emission_choice = None

for i, emission in enumerate(available_emissions):
    with emission_cols[i % 3]:
        if st.button(f"{emission.upper()} (Rank: {emission_ranking[emission]})", key=f"emission_{i}"):
            emission_choice = emission

# Also allow text input
if not emission_choice:
    emission_input = st.selectbox("Or select from dropdown:", available_emissions, format_func=lambda x: x.upper())
    emission_choice = emission_input

if emission_choice:
    emission_class = emission_choice
    emission_rank = emission_ranking[emission_class]
    st.success(f"✓ Selected: {emission_class.upper()} (Rank: {emission_rank})")

# Fuel type selection
st.subheader("Fuel Type")
available_fuels = list(fuel_ranking.keys())

# Create columns for better display
fuel_cols = st.columns(3)
fuel_choice = None

for i, fuel in enumerate(available_fuels):
    with fuel_cols[i % 3]:
        if st.button(f"{fuel.title()} (Rank: {fuel_ranking[fuel]})", key=f"fuel_{i}"):
            fuel_choice = fuel

# Also allow text input
if not fuel_choice:
    fuel_input = st.selectbox("Or select from dropdown:", available_fuels, format_func=lambda x: x.title())
    fuel_choice = fuel_input

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
        "1. Use estimated value based on emission class and fuel type",
        "2. Enter specific CO₂ emissions value (1-10 scale)",
        "3. Enter actual CO₂ emissions in g/km"
    ]
)

if co2_method.startswith("1"):
    # Use estimated value
    if 'estimated_co2' in locals():
        co2_emissions = estimated_co2
        st.success(f"✓ Using estimated CO₂ emissions: {co2_emissions:.2f}")
    else:
        st.warning("Please select emission class and fuel type first")
        st.stop()

elif co2_method.startswith("2"):
    # Enter specific CO₂ emissions value (1-10 scale)
    co2_input = st.slider("Enter CO₂ emissions (1-10 scale):", 1.0, 10.0, 5.0, 0.1)
    co2_emissions = float(co2_input)
    st.success(f"✓ CO₂ emissions set to: {co2_emissions:.2f}")

elif co2_method.startswith("3"):
    # Enter actual CO₂ emissions in g/km
    co2_gkm = st.number_input("Enter CO₂ emissions in g/km (e.g., 120 for petrol car):", 
                             min_value=0, max_value=300, value=120)
    co2_value = float(co2_gkm)
    # Convert g/km to 1-10 scale
    co2_emissions = max(1.0, min(10.0, 10 - (co2_value / 20)))
    st.success(f"✓ {co2_value} g/km converted to score: {co2_emissions:.2f}")

# --------------------------------------------------------
# 9️⃣ Default Values and Automatic Determinations
# --------------------------------------------------------

st.header("6️⃣ Other Details")

# Warranty
warranty = 12  # Default 1 year warranty
st.info(f"✓ Warranty set to default: {warranty} months")

# Transmission is automatic by default
st.info(f"✓ Transmission set to: Automatic (Rank: {transmission_ranking})")

# Determine car type automatically
if 'brand' in locals() and 'model_name' in locals() and 'vehicle_year' in locals():
    is_luxury, is_premium, is_modern = determine_car_type(brand, model_name, vehicle_year)
    st.info(f"✓ Car type determined: Luxury={is_luxury}, Premium={is_premium}, Modern={is_modern}")

# --------------------------------------------------------
# 🔟 Prepare Test Car Data
# --------------------------------------------------------

if st.button("🚀 PREDICT CAR PRICE", type="primary", use_container_width=True):
    
    with st.spinner("Processing input data..."):
        
        # Prepare test car data - EXACTLY the same as before
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

        # --------------------------------------------------------
        # 1️⃣1️⃣ Feature Engineering (EXACTLY the same as before)
        # --------------------------------------------------------

        df = pd.DataFrame([test_car])

        # 1. Basic mileage/age features (EXACTLY the same)
        df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 0.5)
        df['log_mileage'] = np.log1p(df['mileage'])
        df['log_mileage_per_year'] = np.log1p(df['mileage_per_year'])
        df['age_squared'] = df['car_age'] ** 2
        df['mileage_squared'] = df['mileage'] ** 2
        df['age_mileage_ratio'] = df['car_age'] / (df['mileage'] + 1)
        df['age_mileage_interaction'] = df['car_age'] * df['log_mileage']

        # 2. Brand/model features (EXACTLY the same)
        df['brand_model_product'] = df['brand_rank'] * df['model_rank']
        df['brand_model_ratio'] = df['brand_rank'] / (df['model_rank'] + 1e-6)
        df['brand_model_diff'] = df['brand_rank'] - df['model_rank']
        df['brand_rank_sq'] = df['brand_rank'] ** 2
        df['model_rank_sq'] = df['model_rank'] ** 2

        # 3. Emission features (EXACTLY the same)
        df['emission_brand_interaction'] = df['emission_rank'] * df['brand_rank']
        df['emission_model_interaction'] = df['emission_rank'] * df['model_rank']
        df['emission_age_interaction'] = df['emission_rank'] / (df['car_age'] + 1)

        # 4. Frequency features - use neutral/default values (EXACTLY the same)
        df['brand_count_norm'] = 0.5
        df['model_count_norm'] = 0.5
        df['brand_model_count_norm'] = 0.5

        # --------------------------------------------------------
        # 1️⃣2️⃣ Fill Missing Features and Scale (SIMILAR but simpler)
        # --------------------------------------------------------

        # Fill missing features with 0
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0

        # Prepare features in correct order
        X = df[features]

        # Scale features
        X_scaled = scaler.transform(X)

        # --------------------------------------------------------
        # 1️⃣3️⃣ Make Prediction with Single Model
        # --------------------------------------------------------

        with st.spinner("Making prediction with single model..."):
            # Make prediction using the single best model
            pred_log = best_model.predict(X_scaled)[0]  # Log space prediction
            
            # Convert from log to euros
            predicted_price = np.expm1(pred_log)

        # --------------------------------------------------------
        # 1️⃣4️⃣ Display Results
        # --------------------------------------------------------

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
            
        with details_col2:
            st.write(f"**Fuel:** {fuel_type.title()} (Rank: {fuel_rank})")
            st.write(f"**CO₂ Emissions:** {co2_emissions:.2f} (1-10 scale)")
            st.write(f"**Transmission:** Automatic (Rank: {transmission_ranking})")
            st.write(f"**Warranty:** {warranty} months")
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
        # Save prediction to file
        # --------------------------------------------------------
        
        prediction_data = {
            'brand': brand.capitalize(),
            'model': model_name.title(),
            'year': int(vehicle_year),
            'mileage_km': mileage,
            'emission_class': emission_class.upper(),
            'fuel_type': fuel_type.title(),
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
    </div>
    """,
    unsafe_allow_html=True
)