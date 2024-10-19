import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    print("Dataset:\n", data.head(5))
    print("\nShape of the Dataset:", data.shape)
    
    data.fillna(data.mean(), inplace=True)
    data['OverallScore'] = (
        data['GPA'] * 0.4 +
        data['Hackathons'] * 2.0 +
        data['Papers'] * 1.5 +
        data['Teacher Assistance'] * 0.5 +
        data['Consistency'] * 0.2 +
        data['Extracurriculars'] * 0.3 +
        data['Internships'] * 1.5 +
        data['Leadership Roles'] * 2.0
    )
    return data

# Train and compare models, return best model
def train_and_compare_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = r2_score(y_test, y_pred)
    
    plt.bar(results.keys(), results.values(), color='gray', width=0.5)
    plt.title('R^2 Score Comparison with Different Models')
    plt.show()

    best_model = models['RandomForest']
    return best_model.fit(X_train_scaled, y_train), X_test_scaled, y_test

# Display top features from the best model
def display_top_features(model, X):
    importance = model.feature_importances_
    features = pd.DataFrame({'Feature': X.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
    
    plt.barh(features['Feature'], features['Importance'], color=plt.cm.cool(np.linspace(0, 1, len(features))))
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()

# Convert scores to categories (Low, Medium, High)
def categorize_scores(y_pred):
    bins = [-np.inf, 0.4, 0.7, np.inf]
    labels = ['Low', 'Medium', 'High']
    return pd.cut(y_pred, bins=bins, labels=labels)

# Get top 3 students
def get_top_students(X_test_orig, y_pred, data):
    top_students = pd.DataFrame({'StudentID': data.loc[X_test_orig.index, 'StudentID'], 'PredictedScore': y_pred}).nlargest(3, 'PredictedScore')
    print("\nTop 3 Best-Performing Students:\n", top_students)

# Main function
def main():
    data = load_and_preprocess_data('student_performance_dataset.csv')
    X = data.drop(columns=['StudentID', 'OverallScore'])
    y = data['OverallScore']
    
    model, X_test_scaled, y_test = train_and_compare_models(X, y)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Show classification report and accuracy score based on categories
    y_test_cat = categorize_scores(y_test)
    y_pred_cat = categorize_scores(y_pred)
    
    print("\nClassification Report:\n", classification_report(y_test_cat, y_pred_cat))
    print("Accuracy Score:", accuracy_score(y_test_cat, y_pred_cat))
    
    # Get and display top students
    get_top_students(X_test_scaled, y_pred, data)

if __name__ == "__main__":
    main()
