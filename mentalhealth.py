import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r'C:add own file path  ')
# Data preprocessing
# Fill missing values with the mean for numerical columns
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Remove outliers that are more than 3 standard deviations from the mean
df = df[(np.abs(stats.zscore(df[['Age']])) < 3).all(axis=1)]

# Create age groups
df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '65+'])

# Ensure 'treatment' column exists before separating
if 'treatment' not in df.columns:
    raise ValueError("'treatment' column is missing from the original DataFrame.")

# Separate target variable 'treatment' before encoding
target = df['treatment']
df = df.drop(columns=['treatment'])

# One-hot encode categorical features (excluding target variable)
df_encoded = pd.get_dummies(df, drop_first=True)

# Add the target variable back to the encoded DataFrame
df_encoded['treatment'] = target.values

# Prepare data for training
X = df_encoded.drop(columns=['treatment'])
y = df_encoded['treatment']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning with Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train the model with best parameters
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plotting

# Bar plot for average work interference by age group
if 'age_group' in df.columns and 'work_interfere' in df.columns:
    sns.barplot(x='age_group', y='work_interfere', data=df)
    plt.title("Average Work Interference by Age Group")
    plt.show()
else:
    print("The 'age_group' or 'work_interfere' column is not available in the DataFrame.")

# Box plot for Age by age group
if 'age_group' in df.columns and 'Age' in df.columns:
    sns.boxplot(x='age_group', y='Age', data=df)
    plt.title("Age Distribution by Age Group")
    plt.show()
else:
    print("The 'age_group' or 'Age' column is not available in the DataFrame.")

# Compute the correlation matrix
numeric_df = df.select_dtypes(include=[float, int])  # Ensure only numeric data is used
correlation_matrix = numeric_df.corr()

# Heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
