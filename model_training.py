import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the data
df = pd.read_csv("dataset_file.csv")  # Replace with your actual file path

# Basic data exploration
print("Dataset shape:", df.shape)
print(df.head())

# Rename columns for easier access
df.columns = df.columns.str.replace(' ', '_')

# Create the target variable: 1 if poverty rate in 2011 is greater than a threshold (e.g., 20%), else 0
threshold = 20
df['poverty_status'] = (df['2011_-Poverty'] > threshold).astype(int)

# Optionally drop original poverty columns if not needed
df.drop(['2011_-Poverty', '2001_-Poverty'], axis=1, inplace=True)

# Handle missing values in numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Encoding non-numeric columns (e.g., categorical variables)
df = pd.get_dummies(df, drop_first=True)

# Separate features and target variable
X = df.drop('poverty_status', axis=1)  # Features
y = df['poverty_status']  # Target variable

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression(max_iter=1000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
