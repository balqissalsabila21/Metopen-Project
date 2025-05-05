import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
except ImportError:
    print("XGBoost is not installed. Please install it using 'pip install xgboost'")
    XGBRegressor = None

sns.set_style('whitegrid')
sns.set_palette('viridis')

def load_data():
    print("Loading datasets...")
    try:
        cal_data = pd.read_csv('calories.csv')
        exercise_data = pd.read_csv('exercise.csv')
        if cal_data.shape[0] != exercise_data.shape[0]:
            raise ValueError("Datasets have mismatched rows.")
        data = pd.concat([exercise_data, cal_data['Calories']], axis=1)
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(f"Missing values: {data.isnull().sum().sum()}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def visualize_data(data):
    print("\nCreating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.countplot(data=data, x='Gender', ax=axes[0, 0])
    axes[0, 0].set_title('Gender Distribution')
    sns.histplot(data['Duration'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Exercise Duration')
    sns.histplot(data['Heart_Rate'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Heart Rate')
    sns.histplot(data['Calories'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Calories Burned')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    correlation = data.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.tight_layout()
    plt.show()

def prepare_data(data):
    print("\nPreparing data for modeling...")
    data_processed = data.copy()
    data_processed['Gender'] = data_processed['Gender'].map({'male': 0, 'female': 1})
    data_processed['BMI'] = data_processed['Weight'] / ((data_processed['Height'] / 100) ** 2)
    X = data_processed.drop(['User_ID', 'Calories'], axis=1)
    y = data_processed['Calories']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler

def train_models(X_train, X_test, y_train, y_test):
    print("\nTraining models...")
    models = {'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)}
    if XGBRegressor:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42)
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        results[name] = {
            'model': model,
            'train_r2': metrics.r2_score(y_train, train_pred),
            'test_r2': metrics.r2_score(y_test, test_pred),
            'test_mae': metrics.mean_absolute_error(y_test, test_pred)
        }
        print(f"\n{name} Results:")
        print(f"  Training R²: {results[name]['train_r2']:.4f}")
        print(f"  Testing R²: {results[name]['test_r2']:.4f}")
        print(f"  Testing MAE: {results[name]['test_mae']:.4f}")
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    print(f"\nBest model: {best_model_name}")
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
    print("\nCalories Prediction System")
    use_sample = input("Use sample data? (y/n): ").strip().lower() == 'y'
    if use_sample:
        gender = 0
        age = 35
        height = 175
        weight = 70
        duration = 30
        heart_rate = 120
        body_temp = 37.5
    else:
        def get_float(prompt):
            while True:
                try:
                    return float(input(prompt))
                except ValueError:
                    print("Invalid input. Please enter a number.")
        gender_input = input("Gender (male/female): ").strip().lower()
        gender = 0 if gender_input == 'male' else 1
        age = get_float("Age: ")
        height = get_float("Height (cm): ")
        weight = get_float("Weight (kg): ")
        duration = get_float("Exercise Duration (minutes): ")
        heart_rate = get_float("Heart Rate (bpm): ")
        body_temp = get_float("Body Temperature (°C): ")
    bmi = weight / ((height / 100) ** 2)
    input_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp,
        'BMI': bmi
    }
    input_array = np.array([input_data[col] for col in feature_names]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    print(f"\nPredicted calories burned: {prediction[0]:.2f}")

def main():
    print("CALORIES PREDICTION SYSTEM\n")
    data = load_data()
    if data is None:
        return
    visualize_data(data)
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(data)
    best_model = train_models(X_train, X_test, y_train, y_test)
    make_prediction(best_model, scaler, feature_names)
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
