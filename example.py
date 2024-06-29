import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate synthetic time series data with anomalies
np.random.seed(0)
dates = pd.date_range(start='2020-01-01', end='2022-12-31')
data = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(scale=0.1, size=len(dates))

# Introduce anomalies
anomaly_indices = [300, 600, 900]
data[anomaly_indices] += 3

# Create pandas Series
ts = pd.Series(data, index=dates)

# Traditional Time Series Analysis: Moving Average and Standard Deviation
window_size = 30
rolmean = ts.rolling(window=window_size).mean()
rolstd = ts.rolling(window=window_size).std()

# Deep Learning: Autoencoder
def create_dataset(data, look_back=30):
    X = []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
    return np.array(X)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(np.expand_dims(data, axis=1))

look_back = 30
X = create_dataset(scaled_data, look_back)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

model = Sequential([
    Dense(64, activation='relu', input_shape=(look_back,)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(look_back)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, X_train, epochs=100, batch_size=16, verbose=0)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Debug statements to print shapes
print("Shape of train_predict:", train_predict.shape)
print("Shape of test_predict:", test_predict.shape)
print("Shape of data:", data.shape)
print("Length of train_predict_plot:", len(data) - look_back)

# Calculate reconstruction error
train_mse = np.mean(np.power(X_train - train_predict, 2), axis=1)
test_mse = np.mean(np.power(X_test - test_predict, 2), axis=1)

# Detect anomalies
threshold = np.mean(train_mse) + 2 * np.std(train_mse)
anomalies = test_mse > threshold

# Get test indices in the original time series
test_indices = ts.index[look_back + train_size:]

# Plotting the results
plt.figure(figsize=(14, 7))

# Traditional Time Series Analysis
plt.subplot(3, 1, 1)
plt.plot(ts, label='Original')
plt.plot(rolmean, label='Rolling Mean')
plt.plot(rolstd, label='Rolling Std')
plt.scatter(ts.index[anomaly_indices], ts.values[anomaly_indices], color='red', label='Anomalies')
plt.legend()
plt.title('Traditional Time Series Analysis')
plt.show()

# # Deep Learning
# plt.subplot(3, 1, 2)
# plt.plot(ts, label='Original')
#
# # Prepare arrays for plotting train and test predictions
# train_predict_plot = np.empty(len(data))
# train_predict_plot[:] = np.nan
# train_predict_plot[look_back:look_back + len(train_predict)] = scaler.inverse_transform(train_predict[:len(train_predict_plot[look_back:look_back + len(train_predict)])]).flatten()
#
# test_predict_plot = np.empty(len(data))
# test_predict_plot[:] = np.nan
# test_predict_plot[look_back + train_size:] = scaler.inverse_transform(test_predict[:len(test_predict_plot[look_back + train_size:])]).flatten()
#
# plt.plot(ts.index, train_predict_plot, label='Train Predictions')
# plt.plot(ts.index, test_predict_plot, label='Test Predictions')
# plt.scatter(test_indices[anomalies], scaler.inverse_transform(X_test[anomalies]).flatten(), color='red', label='Anomalies')
# plt.legend()
# plt.title('Deep Learning (Autoencoder)')
#
# # Anomaly Detection Results
# plt.subplot(3, 1, 3)
# plt.bar(['Traditional TS Analysis', 'Deep Learning'], [len(anomaly_indices), np.sum(anomalies)])
# plt.ylabel('Number of Anomalies Detected')
# plt.title('Anomaly Detection Results')
#
# plt.tight_layout()
# plt.show()