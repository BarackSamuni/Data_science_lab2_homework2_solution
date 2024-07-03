import config as config
import numpy as np

#Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

# Display settings for better visualization
sns.set(style="whitegrid")


# Define data types for each column
dtypes = {'Date': str,
          'Time': str,
          'Global_active_power': float,
          'Global_reactive_power': float,
          'Voltage': float,
          'Global_intensity': float,
          'Sub_metering_1': float,
          'Sub_metering_2': float,
          'Sub_metering_3': float}

# Read the CSV file and handle missing values
file_path = 'household_power_consumption.txt'  # Ensure this file is downloaded and in the correct path
energy = pd.read_csv(file_path, sep=';', dtype=dtypes, na_values="?")


# Combine Date and Time into a single datetime column
energy['datetime'] = pd.to_datetime(energy['Date'] + ' ' + energy['Time'], format='%d/%m/%Y %H:%M:%S')

# Set the datetime column as the index
energy.set_index('datetime', inplace=True)

# Drop the original Date and Time columns as they are now redundant
energy.drop(columns=['Date', 'Time'], inplace=True)

# Calculate the active energy consumed every minute in the household by electrical equipment not measured in sub-meterings 1, 2, and 3
energy['Other_active_power'] = (energy['Global_active_power'] * 1000 / 60) - (energy['Sub_metering_1'] + energy['Sub_metering_2'] + energy['Sub_metering_3'])
energy = energy.ffill()
# Display the first few rows of the dataframe with the new column
print(energy.head())



#
# # Create subplots
# f, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False,
#                        gridspec_kw={"height_ratios": (.15, .85), "width_ratios": (.60, .40)})
#
# # Main boxplot
# sns.boxplot(x=energy["Global_active_power"], color="blue", ax=axes[0, 0])
# axes[0, 0].set_xlabel('')
# axes[0, 0].set_xlim(0, None)
#
# # Main distribution
# sns.histplot(energy["Global_active_power"], kde=True, color="red", ax=axes[1, 0])
# axes[1, 0].set_xlim(0, None)
# axes[1, 0].set_title('Distribution of Global_active_power')
#
# # Zoom boxplot
# sns.boxplot(x=energy["Global_active_power"], color="purple", ax=axes[0, 1])
# axes[0, 1].set_xlabel('')
# axes[0, 1].set_xlim(0, 3.37)
#
# # Zoom distribution
# sns.histplot(energy["Global_active_power"], kde=True, color="purple", ax=axes[1, 1])
# axes[1, 1].set_title('Distribution without Outliers')
# axes[1, 1].set_xlim(0, 3.37)
#
# # Add an arrow to mark the mean value
# axes[1, 1].annotate(
#     'Mean', xy=(energy["Global_active_power"].mean(), 0.15), xytext=(0.9, 0.6),
#     arrowprops=dict(facecolor='black', shrink=0.01))
#
# print(energy.columns)
# plt.tight_layout()
# plt.show()
#
# # Display summary statistics for Global_active_power
# summary_stats = np.round(energy['Global_active_power'].describe(), 2).apply(lambda x: format(x, 'f'))
# print(summary_stats)
#
# # Step 4: Visualize time series trends
#
# # Plot the time series data for Global Active Power
# plt.figure(figsize=(14, 7))
# plt.plot(energy['Global_active_power'], color='blue', label='Global Active Power')
# plt.title('Time Series of Global Active Power')
# plt.xlabel('DateTime')
# plt.ylabel('Global Active Power (kilowatts)')
# plt.legend()
# plt.show()
#
#
# # Task 2: Perform Exploratory Data Analysis (EDA)
# # Resampling to daily mean for better visualization
# daily_energy = energy['Global_active_power'].resample('D').mean()
# plt.figure(figsize=(14, 7))
# plt.plot(daily_energy, color='blue', label='Daily Mean Global Active Power')
# plt.title('Daily Mean Global Active Power')
# plt.xlabel('Date')
# plt.ylabel('Global Active Power (kilowatts)')
# plt.legend()
# plt.show()
#
#
# # 2.2: Check for seasonality and cyclical patterns
# # Resampling to monthly mean to observe seasonality
#
#
# monthly_energy = energy['Global_active_power'].resample('ME').mean()
#
# plt.figure(figsize=(14, 7))
# plt.plot(monthly_energy, color='green', label='Monthly Mean Global Active Power')
# plt.title('Monthly Mean Global Active Power')
# plt.xlabel('Date')
# plt.ylabel('Global Active Power (kilowatts)')
# plt.legend()
# plt.show()
#
# # 2.6: Analyze Distribution of Power Consumption
#
# # Distribution plot for Global Active Power
#
# plt.figure(figsize=(14, 7))
# sns.histplot(energy['Global_active_power'], kde=True, color='red')
# plt.title('Distribution of Global Active Power')
# plt.xlabel('Global Active Power (kilowatts)')
# plt.ylabel('Frequency')
# plt.show()
#
# # 2.7: Identify and Handle Missing Values or Outliers
#
# # Fill any missing values in the DataFrame by carrying forward the last valid observation
# energy = energy.ffill()
# energy.isnull().sum()
#
# # Detect outliers using boxplot
# plt.figure(figsize=(14, 7))
# sns.boxplot(x=energy['Global_active_power'], color='orange')
# plt.title('Boxplot of Global Active Power')
# plt.xlabel('Global Active Power (kilowatts)')
# plt.show()


# # Display settings for better visualization
# sns.set(style="whitegrid")
#
# #Step 3.2: Prepare the Data
# #: For this task, we will consider the "Global_active_power" as the target variable and create lag variables as features.
#
# # Create lag features
# energy['lag_1'] = energy['Global_active_power'].shift(1)
# energy['lag_2'] = energy['Global_active_power'].shift(2)
# energy['lag_3'] = energy['Global_active_power'].shift(3)
#
# # Fill missing values created by the lagging process using forward fill
# energy = energy.dropna()
# print(energy.isna().sum())
#
#
# # Define the feature set (X) and target variable (y)
# X = energy[['lag_1', 'lag_2', 'lag_3']]
# y = energy['Global_active_power']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
# # Step 3.3: Train the Linear Regression Model
#
# # Initialize the model
# lr_model = LinearRegression()
#
# # Train the model
# lr_model.fit(X_train, y_train)
#
# # Step 3.4: Make Predictions
#
# # Make predictions on the test set
# y_pred = lr_model.predict(X_test)
#
# # Compare the first few predicted values with the actual values
# predicted_vs_actual = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(predicted_vs_actual.head())
#
# # Step 3.5: Evaluate the Model
#
# # Calculate evaluation metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
#
# # Print the evaluation metrics
# print(f'Mean Absolute Error (MAE): {mae:.4f}')
# print(f'Mean Squared Error (MSE): {mse:.4f}')
# print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
# print(f'R-squared (R²) value: {r2:.4f}')
#
# # Plot the actual vs predicted values
# plt.figure(figsize=(14, 7))
# plt.plot(y_test.index, y_test, label='Actual', color='blue')
# plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='dashed')
# plt.title('Actual vs Predicted Global Active Power')
# plt.xlabel('DateTime')
# plt.ylabel('Global Active Power (kilowatts)')
# plt.legend()
# plt.show()
#
#
# # Plot the actual vs predicted values for different time periods
# time_periods = [100, 200, 300, 400]
#
# fig, axes = plt.subplots(2, 2, figsize=(20, 14))
# axes = axes.flatten()
#
# for i, period in enumerate(time_periods):
#     axes[i].plot(y_test.index[-period:], y_test[-period:], label='Actual', color='blue')
#     axes[i].plot(y_test.index[-period:], y_pred[-period:], label='Predicted', color='red', linestyle='dashed')
#     axes[i].set_title(f'Actual vs Predicted Global Active Power (Last {period} Data Points)')
#     axes[i].set_xlabel('DateTime')
#     axes[i].set_ylabel('Global Active Power (kilowatts)')
#     axes[i].legend()
#
# plt.tight_layout()
# plt.show()

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Assuming energy is already loaded and processed (missing values handled, datetime index set)
# Select the target variable
data = energy['Global_active_power'].values.reshape(-1, 1)

# Resample data to 10-minute intervals
data_resampled = data.resample('10T').mean()

# Scale the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 10  # You can adjust this value
X, y = create_sequences(data_scaled, time_step)

# Reshape the input to be [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]


# Design the RNN model
model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(SimpleRNN(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, verbose=1)


#

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# Visualize results
plt.figure(figsize=(14, 7))
plt.plot(y_test[0], label='Actual', color='blue')
plt.plot(test_predict, label='Predicted', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Global Active Power')
plt.xlabel('Time Step')
plt.ylabel('Global Active Power (kilowatts)')
plt.legend()
plt.show()


# Calculate performance metrics
mae = mean_absolute_error(y_test[0], test_predict)
mse = mean_squared_error(y_test[0], test_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test[0], test_predict)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")


