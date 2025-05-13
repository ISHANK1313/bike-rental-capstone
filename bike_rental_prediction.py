import pandas as bike_data
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Verify dataset exists
dataset_path = 'hour.csv'
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please download 'hour.csv' from UCI Bike Sharing Dataset.")

# Load UCI Bike Sharing Dataset
bike_data = bike_data.read_csv(dataset_path)

# Data Preprocessing
# Rename columns for clarity
bike_data = bike_data.rename(columns={'cnt': 'rentals', 'hr': 'hour', 'temp': 'temperature'})

# Select features and target
features = ['hour', 'temperature', 'hum', 'windspeed', 'season', 'weathersit', 'weekday']
X = bike_data[features]
y = bike_data['rentals']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize feature relationships (new addition for presentation)
sns.pairplot(bike_data, x_vars=['hour', 'temperature', 'hum'], y_vars='rentals', height=4)
plt.suptitle('Bike Rental Relationships', y=1.02)
plt.savefig('bike_rental_pairplot.png')
plt.show()

# Plot predicted vs actual rentals
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Rentals')
plt.ylabel('Predicted Rentals')
plt.title('Predicted vs Actual Bike Rentals')
plt.savefig('predicted_vs_actual.png')
plt.show()