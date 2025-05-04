import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# Set basic plot style
plt.style.use('seaborn')
sns.set_palette('viridis')

def load_data():
    """Load and combine the dataset files"""
    print("Loading datasets...")
    
    try:
        # Load datasets
        cal_data = pd.read_csv('calories.csv')
        exercise_data = pd.read_csv('exercise.csv')
        
        # Combine datasets
        data = pd.concat([exercise_data, cal_data['Calories']], axis=1)
        
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(f"Missing values: {data.isnull().sum().sum()}")
        
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def visualize_data(data):
    """Create basic visualizations of the data"""
    print("\nCreating visualizations...")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gender distribution
    sns.countplot(data=data, x='Gender', ax=axes[0, 0])
    axes[0, 0].set_title('Gender Distribution')
    
    # Duration distribution
    sns.histplot(data['Duration'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Exercise Duration')
    
    # Heart Rate distribution
    sns.histplot(data['Heart_Rate'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Heart Rate')
    
    # Calories distribution
    sns.histplot(data['Calories'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Calories Burned')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    numerical_data = data.select_dtypes(include=[np.number])
    correlation = numerical_data.corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.tight_layout()
    plt.show()

def prepare_data(data):
    """Prepare data for model training"""
    print("\nPreparing data for modeling...")
    
    # Encode categorical features
    data_processed = data.copy()
    data_processed.replace({'Gender': {'male': 0, 'female': 1}}, inplace=True)
    
    # Add BMI as a feature
    data_processed['BMI'] = data_processed['Weight'] / ((data_processed['Height']/100) ** 2)
    
    # Split features and target
    X = data_processed.drop(['User_ID', 'Calories'], axis=1)
    y = data_processed['Calories']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate models"""
    print("\nTraining models...")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = metrics.r2_score(y_train, train_pred)
        test_r2 = metrics.r2_score(y_test, test_pred)
        test_mae = metrics.mean_absolute_error(y_test, test_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae
        }
        
        print(f"\n{name} Results:")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Testing R²: {test_r2:.4f}")
        print(f"  Testing MAE: {test_mae:.4f}")
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    
    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_model.predict(X_test), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title(f'Actual vs Predicted Calories ({best_model_name})')
    plt.tight_layout()
    plt.show()
    
    return best_model

def make_prediction(model, scaler, feature_names):
    """Make a prediction with user input"""
    print("\nCalories Prediction System")
    
    # Get user input or use sample data
    use_sample = input("Use sample data? (y/n): ").lower() == 'y'
    
    if use_sample:
        # Sample data: Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp, BMI
        gender = 0  # male
        age = 35
        height = 175
        weight = 70
        duration = 30
        heart_rate = 120
        body_temp = 37.5
        bmi = weight / ((height/100) ** 2)
        
        print("\nSample data:")
        print(f"Gender: {'male' if gender == 0 else 'female'}")
        print(f"Age: {age}")
        print(f"Height: {height} cm")
        print(f"Weight: {weight} kg")
        print(f"Exercise Duration: {duration} minutes")
        print(f"Heart Rate: {heart_rate} bpm")
        print(f"Body Temperature: {body_temp} °C")
        print(f"BMI: {bmi:.2f}")
        
    else:
        print("\nEnter your details:")
        gender_input = input("Gender (male/female): ").lower()
        gender = 0 if gender_input == 'male' else 1
        age = float(input("Age: "))
        height = float(input("Height (cm): "))
        weight = float(input("Weight (kg): "))
        duration = float(input("Exercise Duration (minutes): "))
        heart_rate = float(input("Heart Rate (bpm): "))
        body_temp = float(input("Body Temperature (°C): "))
        bmi = weight / ((height/100) ** 2)
    
    # Create input array (match the order of feature_names)
    input_data = []
    for feature in feature_names:
        if feature == 'Gender':
            input_data.append(gender)
        elif feature == 'Age':
            input_data.append(age)
        elif feature == 'Height':
            input_data.append(height)
        elif feature == 'Weight':
            input_data.append(weight)
        elif feature == 'Duration':
            input_data.append(duration)
        elif feature == 'Heart_Rate':
            input_data.append(heart_rate)
        elif feature == 'Body_Temp':
            input_data.append(body_temp)
        elif feature == 'BMI':
            input_data.append(bmi)
    
    input_array = np.array(input_data).reshape(1, -1)
    
    # Scale the input
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    print(f"\nPredicted calories burned: {prediction[0]:.2f}")

def main():
    """Main function to execute the workflow"""
    print("CALORIES PREDICTION SYSTEM\n")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Visualize data
    visualize_data(data)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(data)
    
    # Train models
    best_model = train_models(X_train, X_test, y_train, y_test)
    
    # Make prediction
    make_prediction(best_model, scaler, feature_names)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()