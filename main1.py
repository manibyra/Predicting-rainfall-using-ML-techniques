# Importing libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing
# Read the data in a pandas dataframe
data = pd.read_csv("austin_weather.csv")

# Drop unnecessary columns
data = data.drop(['Events', 'Date', 'SeaLevelPressureHighInches', 'SeaLevelPressureLowInches'], axis=1)

# Replace 'T' and '-' with 0.0 for numerical consistency
data = data.replace('T', 0.0)
data = data.replace('-', 0.0)

# Convert relevant columns to numeric (force non-numeric values to NaN)
data = data.apply(pd.to_numeric, errors='coerce')

# Fill NaN values with the mean of the column
data.fillna(data.mean(), inplace=True)

# Dynamically calculate average values
data['TempAvgF'] = data[['TempHighF', 'TempLowF']].mean(axis=1)
data['DewPointAvgF'] = data[['DewPointHighF', 'DewPointLowF']].mean(axis=1)
data['HumidityAvgPercent'] = data[['HumidityHighPercent', 'HumidityLowPercent']].mean(axis=1)
data['VisibilityAvgMiles'] = data[['VisibilityHighMiles', 'VisibilityLowMiles']].mean(axis=1)
data['WindAvgMPH'] = data[['WindHighMPH', 'WindGustMPH']].mean(axis=1)

data.to_csv('austin_final.csv', index=False)

# Step 2: Load Cleaned Data and Prepare for Training
data = pd.read_csv("austin_final.csv")

# Define features (X) and label (Y)
X = data.drop(['PrecipitationSumInches'], axis=1)
Y = data['PrecipitationSumInches'].values.reshape(-1, 1)

# Step 3: Train the Linear Regression Model
clf = LinearRegression()
clf.fit(X, Y)

# Step 4: Runtime Input for Prediction with Ranges
print("Enter the following weather attributes to predict precipitation:")
print("(Provide values within the specified ranges for better predictions.)")

# Define the attributes and their ranges
attribute_ranges = {
    "TempHighF": "(-10 to 120)", "TempLowF": "(-20 to 90)",
    "DewPointHighF": "(0 to 80)", "DewPointLowF": "(0 to 60)",
    "HumidityHighPercent": "(0 to 100)", "HumidityLowPercent": "(0 to 100)",
    "SeaLevelPressureAvgInches": "(28.0 to 32.0)", "VisibilityHighMiles": "(0 to 10)",
    "VisibilityLowMiles": "(0 to 10)", "WindHighMPH": "(0 to 50)", "WindGustMPH": "(0 to 60)"
}

# Collect user inputs
user_input = []
for attr, attr_range in attribute_ranges.items():
    value = float(input(f"Enter value for {attr} {attr_range}: "))
    user_input.append(value)

# Compute average values for the input dynamically
temp_avg = np.mean([user_input[0], user_input[1]])  # TempAvgF
dewpoint_avg = np.mean([user_input[2], user_input[3]])  # DewPointAvgF
humidity_avg = np.mean([user_input[4], user_input[5]])  # HumidityAvgPercent
visibility_avg = np.mean([user_input[7], user_input[8]])  # VisibilityAvgMiles
wind_avg = np.mean([user_input[9], user_input[10]])  # WindAvgMPH

# Append computed averages to the user input list
user_input.extend([temp_avg, dewpoint_avg, humidity_avg, visibility_avg, wind_avg])

# Convert user input to NumPy array
inp = np.array(user_input).reshape(1, -1)

# Predict precipitation
prediction = clf.predict(inp)
print(f'The predicted precipitation in inches for the given input is: {prediction[0][0]}')

# Step 5: Visualization
# Plot precipitation trend over days
days = np.arange(Y.size)

print("Plotting precipitation trend graph...")
plt.scatter(days, Y, color='g', label="Precipitation levels")
plt.scatter(days[-1], prediction, color='r', label="Predicted Precipitation")
plt.xlabel("Days")
plt.ylabel("Precipitation in inches")
plt.title("Precipitation Levels Over Days")
plt.legend()
plt.grid()
plt.show()

# Plot precipitation vs selected attributes
x_vis = X[['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
           'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH']]

print("Plotting precipitation vs selected attributes graph...")
plt.figure(figsize=(12, 8))
for i, column in enumerate(x_vis.columns):
    plt.subplot(3, 2, i + 1)
    plt.scatter(days, x_vis[column], color='b', label=column)
    plt.scatter(days[-1], user_input[6 + i], color='r', label="Given Input")
    plt.xlabel("Days")
    plt.ylabel(column)
    plt.title(column)
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()
