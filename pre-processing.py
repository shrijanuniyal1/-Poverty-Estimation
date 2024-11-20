# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
df = pd.read_csv("dataset_file.csv")  # Update with the correct file path if necessary

# Basic data exploration
print("Dataset shape:", df.shape)
print(df.head())

# Handle missing values in numeric columns only
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Rename columns to remove spaces for easier access
df.columns = df.columns.str.replace(' ', '_')

# Set a threshold to define poverty status (e.g., 20% poverty rate)
threshold = 20
df['poverty_status'] = (df['2011_-Poverty'] > threshold).astype(int)  # 1 = poor, 0 = non-poor

# Separate features and target variable
X = df.drop(['poverty_status', '2011_-Poverty', '2001_-Poverty'], axis=1)  # Drop target and poverty columns
y = df['poverty_status']  # Target variable

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler (standardization: mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
