import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Wine Quality Prediction", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('winequality-red.csv')  # Relative path
    return df

df = load_data()

# Select features and target
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides']
target = 'quality'

X = df[features]
y = df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Define and train models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Machine': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)

# Streamlit application
st.title('Wine Quality Prediction')

# Sidebar for navigation
# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose an option", ["Prediction", "Comparative Analysis", "Wine Composition", "Important Features"])


if option == "Prediction":
    st.header('Predict Wine Quality')
    
    # Parameter sliders with units
    fixed_acidity = st.slider('Fixed Acidity (g/dm³)', 4.0, 15.0, 7.4)
    volatile_acidity = st.slider('Volatile Acidity (g/dm³)', 0.1, 1.5, 0.7)
    citric_acid = st.slider('Citric Acid (g/dm³)', 0.0, 1.0, 0.0)
    residual_sugar = st.slider('Residual Sugar (g/dm³)', 0.0, 15.0, 1.9)
    chlorides = st.slider('Chlorides (g/dm³)', 0.01, 0.2, 0.076)

    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides]
    })

    # Model selection
    model_choice = st.selectbox('Choose a Model', list(models.keys()))

    # Predict using the selected model
    selected_model = models[model_choice]
    predicted_quality = selected_model.predict(input_data)

    # Display prediction
    st.subheader("Wine Quality Prediction")
    st.write(f"**Predicted Wine Quality using {model_choice}:**")
    st.write(f"<h1 style='text-align: center; color: #4CAF50;'>{predicted_quality[0]:.2f}</h1>", unsafe_allow_html=True)

    # Evaluation metrics
    y_pred = selected_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Model Evaluation Metrics**")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")


if option == "Comparative Analysis":
    st.header('Comparative Analysis of Models')

    # Parameters
    fixed_acidity = st.slider('Fixed Acidity (g/dm³)', 4.0, 15.0, 7.4)
    volatile_acidity = st.slider('Volatile Acidity (g/dm³)', 0.1, 1.5, 0.7)
    citric_acid = st.slider('Citric Acid (g/dm³)', 0.0, 1.0, 0.0)
    residual_sugar = st.slider('Residual Sugar (g/dm³)', 0.0, 15.0, 1.9)
    chlorides = st.slider('Chlorides (g/dm³)', 0.01, 0.2, 0.076)

    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides]
    })

    # Create DataFrame for storing predictions and metrics
    results = {
        'Model': [],
        'Prediction': [],
        'MAE': [],
        'MSE': [],
        'R² Score': []
    }

    for model_name, model in models.items():
        pred = model.predict(input_data)[0]
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results['Model'].append(model_name)
        results['Prediction'].append(f"{pred:.2f}")
        results['MAE'].append(f"{mae:.2f}")
        results['MSE'].append(f"{mse:.2f}")
        results['R² Score'].append(f"{r2:.2f}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display results in a table
    st.subheader("Model Predictions and Evaluation Metrics")
    st.dataframe(results_df)


if option == "Wine Composition":
    st.header('Wine Composition Analysis')

    # Calculate average values
    avg_composition = df[features].mean()
    
    # Prepare data for pie chart
    pie_data = avg_composition.reset_index()
    pie_data.columns = ['Feature', 'Average Value']
    
    # Convert feature names to title case
    pie_data['Feature'] = pie_data['Feature'].str.title()

    # Customize pie chart
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed
    wedges, texts, autotexts = ax.pie(
        pie_data['Average Value'], 
        labels=pie_data['Feature'], 
        autopct='%1.1f%%', 
        colors=plt.cm.Paired(range(len(pie_data))),  # Use color map for better contrast
        startangle=140, 
        wedgeprops=dict(width=0.3)  # Adjust wedge width for better appearance
    )

    # Style pie chart for dark mode
    plt.setp(autotexts, size=10, weight="bold", color="black")  # Adjust text style
    plt.setp(texts, size=10, weight="bold", color="black")      # Adjust text style
    ax.set_facecolor('black')  # Background color for dark mode

    st.pyplot(fig)


if option == "Important Features":
    st.title('Feature Importance and Wine Quality Distribution')

    # Feature Importance Analysis
    st.header('Feature Importance Analysis')
    
    # Train model to get feature importances
    model = RandomForestRegressor()
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

    # Plot feature importance
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
    ax.set_title('Feature Importance')
    st.pyplot(fig)

    # Wine Quality Distribution
    st.header('Wine Quality Distribution')

    # Plot wine quality distribution
    fig, ax = plt.subplots()
    sns.histplot(df[target], bins=range(int(df[target].min()), int(df[target].max()) + 1), ax=ax)
    ax.set_title('Distribution of Wine Quality')
    ax.set_xlabel('Quality')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
