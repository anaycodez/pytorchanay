import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Task 1: Load EV charging reports
ev_charging_reports = pd.read_csv('datasets/EV charging reports.csv')
print(ev_charging_reports.head())

# Task 2: Load traffic distribution data
traffic_reports = pd.read_csv('datasets/Local traffic distribution.csv')
print(traffic_reports.head())

# Task 3: Merge datasets
ev_charging_traffic = pd.merge(ev_charging_reports, traffic_reports, left_on='Start_plugin_hour', right_on='Date_from')
print(ev_charging_traffic.head())

# Task 4: Inspect the merged dataset
print(ev_charging_traffic.info())

# Task 5: Drop unnecessary columns
columns_to_drop = ['session_ID', 'Garage_ID', 'User_ID', 'Shared_ID', 'Plugin_category', 'Duration_category', 'Start_plugin', 'Start_plugin_hour', 'End_plugout', 'End_plugout_hour', 'Date_from', 'Date_to']
ev_charging_traffic.drop(columns=columns_to_drop, inplace=True)

# Task 6: Replace commas with dots in numeric columns and convert to float
ev_charging_traffic['El_kWh'] = ev_charging_traffic['El_kWh'].str.replace(',', '.').astype(float)
ev_charging_traffic['Duration_hours'] = ev_charging_traffic['Duration_hours'].str.replace(',', '.').astype(float)

# Task 7: Convert all data to floats
ev_charging_traffic = ev_charging_traffic.astype(float)

# Task 8: Split data into training and testing sets
X = ev_charging_traffic.drop('El_kWh', axis=1)
y = ev_charging_traffic['El_kWh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Task 10: Train a linear regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred = model_lr.predict(X_test)

# Task 11: Calculate the MSE for the linear regression model
test_mse = mean_squared_error(y_test, y_pred)
print("Linear Regression Test MSE:", test_mse)

# Task 12-19: Setup and train a neural network using PyTorch
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 56),
    nn.ReLU(),
    nn.Linear(56, 26),
    nn.ReLU(),
    nn.Linear(26, 1)
)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Training loop
for epoch in range(3000):
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss ▋
