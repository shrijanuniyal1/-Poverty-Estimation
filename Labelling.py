import pandas as pd

# Load the data
df = pd.read_csv("dataset_file.csv")  # Replace with your actual file path

# Basic data exploration
print("Dataset shape:", df.shape)
print(df.head())

# Rename columns for easier access
df.columns = df.columns.str.replace(' ', '_')

# Set a threshold for defining poverty (e.g., 20% poverty rate)
threshold = 20

# Create a binary target variable 'poverty_status'
df['poverty_status'] = (df['2011_-Poverty'] > threshold).astype(int)  # 1 = poor, 0 = non-poor

# Optionally drop original poverty columns
df.drop(['2011_-Poverty', '2001_-Poverty'], axis=1, inplace=True)

# Check the updated DataFrame
print(df.head())
