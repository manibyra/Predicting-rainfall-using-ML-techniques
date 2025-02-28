import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load and clean the data
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(['Events', 'Date', 'SeaLevelPressureHighInches', 'SeaLevelPressureLowInches'], axis=1)
    data = data.replace({'T': 0.0, '-': 0.0})
    data.to_csv('austin_final.csv', index=False)
    return data

# Train the Linear Regression model
def train_model(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model

# Evaluate the model
def evaluate_model(model, X, Y):
    y_pred = model.predict(X)
    r2 = r2_score(Y, y_pred)
    mse = mean_squared_error(Y, y_pred)
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

# Visualization
def plot_precipitation(days, Y, day_index):
    plt.scatter(days, Y, color='g')
    plt.scatter(days[day_index], Y[day_index], color='r')
    plt.title("Precipitation Level Over Days")
    plt.xlabel("Days")
    plt.ylabel("Precipitation (inches)")
    plt.show()

def plot_feature_vs_precipitation(days, X, day_index):
    x_vis = X[['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent', 'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH']]
    plt.figure(figsize=(10, 8))
    for i, column in enumerate(x_vis.columns):
        plt.subplot(3, 2, i + 1)
        plt.scatter(days, x_vis[column], color='g')
        plt.scatter(days[day_index], x_vis[column][day_index], color='r')
        plt.title(column)
    plt.tight_layout()
    plt.show()

# Main script
data = load_and_clean_data("austin_weather.csv")
data = pd.read_csv("austin_final.csv")

X = data.drop(['PrecipitationSumInches'], axis=1)
Y = data['PrecipitationSumInches'].values.reshape(-1, 1)

days = np.arange(Y.size)
day_index = 798

# Train and evaluate the model
model = train_model(X, Y)
evaluate_model(model, X, Y)

# Predict precipitation for a sample input
sample_input = np.array(X.iloc[0]).reshape(1, -1)  # Use first row as sample input
print(f'The predicted precipitation (in inches) is: {model.predict(sample_input)[0][0]:.4f}')

# Plot graphs
plot_precipitation(days, Y, day_index)
plot_feature_vs_precipitation(days, X, day_index)
