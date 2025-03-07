import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
city_day = pd.read_csv('city_day.csv')
city_hour = pd.read_csv('city_hour.csv')
station_day = pd.read_csv('station_day.csv')
station_hour = pd.read_csv('station_hour.csv')
stations = pd.read_csv('stations.csv')

# Display basic info
print("City Day Data Columns:", city_day.columns)
print("City Hour Data Columns:", city_hour.columns)
print("Station Day Data Columns:", station_day.columns)
print("Station Hour Data Columns:", station_hour.columns)
print("Stations Data Columns:", stations.columns)

# Selecting relevant columns for AQI prediction
df = city_day[['City', 'Date', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']]

# Handling missing values
df.dropna(inplace=True)

# Convert date to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Feature Engineering: Extracting year, month, and day
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Encode city names numerically
df['City'] = df['City'].astype('category').cat.codes

# Define features and target variable
X = df.drop(columns=['Date', 'AQI'])
y = df['AQI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}')
print(f'R^2 Score: {r2}')

# Visualization 1: Feature Importance
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance in AQI Prediction')
plt.show()

# Visualization 2: AQI Trends Over Time
plt.figure(figsize=(12, 6))
sns.lineplot(x=df['Date'], y=df['AQI'])
plt.title('AQI Trends Over Time')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.show()

# Visualization 3: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of AQI and Pollutants')
plt.show()

# Sustainability Insights
print("\nSustainability Insights:")
print("- AI can help predict air quality trends and guide urban policies.")
print("- Monitoring pollutant correlations can inform sustainable traffic and industrial regulations.")
print("- AQI forecasting can help reduce pollution-related health risks.")
