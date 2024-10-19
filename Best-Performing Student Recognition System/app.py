import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Load data from CSV
def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path, delimiter=',', skipinitialspace=True)
        data.columns = data.columns.str.strip()  # Strip whitespace from column names
        return data
    except Exception as e:
        st.error(f"Error loading the data: {e}")
        return None

# Preprocess the data (handle missing values, normalize data, etc.)
def preprocess_data(data):
    """Handle missing values and normalize the data."""
    # Replace missing values with 0 or median where applicable
    data.fillna({
        'GPA': data['GPA'].median(),
        'Hackathons': 0,
        'Papers': 0,
        'Teacher Assistance': 0,
        'Core Engineering Score': data['Core Engineering Score'].median(),
        'Consistency': data['Consistency'].median(),
        'Extracurriculars': 0,
        'Internships': 0,
        'Leadership Roles': 0
    }, inplace=True)
    
    # Normalize the numeric columns for consistency
    scaler = MinMaxScaler()
    columns_to_scale = ['GPA', 'Hackathons', 'Papers', 'Teacher Assistance', 
                        'Core Engineering Score', 'Consistency', 
                        'Extracurriculars', 'Internships', 
                        'Leadership Roles']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    
    return data

# Calculate the OverallScore
def calculate_overall_score(data):
    """Calculate the overall score based on various parameters."""
    data['OverallScore'] = (
        data['GPA'] * 0.4 +
        data['Hackathons'] * 2.0 +
        data['Papers'] * 1.5 +
        data['Teacher Assistance'] * 0.5 +
        data['Core Engineering Score'] * 0.3 +
        data['Consistency'] * 0.2 +
        data['Extracurriculars'] * 0.3 +
        data['Internships'] * 1.5 +
        data['Leadership Roles'] * 2.0
    )
    return data

# Train the Linear Regression Model
def train_model(data):
    """Train a Linear Regression model to predict the OverallScore."""
    X = data[['GPA', 'Hackathons', 'Papers', 'Teacher Assistance', 
              'Core Engineering Score', 'Consistency', 
              'Extracurriculars', 'Internships', 
              'Leadership Roles']]
    y = data['OverallScore']

    model = LinearRegression()
    model.fit(X, y)

    return model

# Predict OverallScore and rank students
def rank_students(data, model):
    """Predict the OverallScore using the trained model and rank students."""
    X = data[['GPA', 'Hackathons', 'Papers', 'Teacher Assistance', 
              'Core Engineering Score', 
              'Consistency', 'Extracurriculars', 
              'Internships', 'Leadership Roles']]
    
    # Predict the OverallScore using the trained model
    data['PredictedOverallScore'] = model.predict(X)

    # Rank students based on the predicted score
    top_students = data.nlargest(3, 'PredictedOverallScore')[['StudentID', 
                                                               'PredictedOverallScore']]
    
    # Add Rank column
    top_students['Rank'] = range(1, len(top_students) + 1)
    
    return top_students

def main():
    st.title("Top 3 Best-Performing Students Recognition System")

    # File path to the dataset
    file_path = 'student_data.csv'  # Update this path with your actual file location

    # Load the data
    data = load_data(file_path)

    if data is not None:
        # Check if all required columns are present
        required_columns = ['StudentID', 'GPA', 'Hackathons',
                            'Papers', 'Teacher Assistance',
                            'Core Engineering Score',
                            'Consistency', 'Extracurriculars',
                            'Internships', 'Leadership Roles']
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")
            return
        
        # Preprocess the data
        data = preprocess_data(data)

        # Calculate OverallScore
        data = calculate_overall_score(data)

        # Train a Linear Regression model to predict the overall score
        model = train_model(data)

        # Rank students based on predicted scores
        top_students = rank_students(data, model)

        # Display top students in a formatted way with ranks
        st.subheader("Top 3 Best-Performing Students:")
        
        for index, row in top_students.iterrows():
            st.write(f"Rank {row['Rank']}: Student ID {int(row['StudentID'])} - Predicted Score: {row['PredictedOverallScore']:.2f}")

if __name__ == "__main__":
    main()