import xml.etree.ElementTree as ET
import pandas as pd

# Load XML file
tree = ET.parse('G:/Downloads/archive/559-ws-testing.xml')
root = tree.getroot()

timestamps = []
glucose = []

# Extract glucose values
for event in root.find('glucose_level').findall('event'):
    timestamps.append(event.get('ts'))
    glucose.append(float(event.get('value')))

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'glucose': glucose
})

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S')

# Save as CSV
df.to_csv('glucose_data.csv', index=False)

print(df.head())

import numpy as np

window_size = 6   # past 30 minutes

X = []
y_15 = []
y_30 = []
y_60 = []

glucose_values = df['glucose'].values

for i in range(window_size, len(glucose_values) - 12):

    # Past values
    X.append(glucose_values[i-window_size:i])

    # Future targets
    y_15.append(glucose_values[i+3])
    y_30.append(glucose_values[i+6])
    y_60.append(glucose_values[i+12])

# Convert to numpy arrays
X = np.array(X)

y_15 = np.array(y_15)
y_30 = np.array(y_30)
y_60 = np.array(y_60)

print("X shape:", X.shape)
print("15 min target shape:", y_15.shape)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_15, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

from sklearn.ensemble import RandomForestRegressor

# Create model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Metrics
rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)


plt.figure(figsize=(12,6))

plt.plot(y_test[:100], label='Actual')
plt.plot(rf_pred[:100], label='Random Forest')

plt.xlabel('Sample')
plt.ylabel('Glucose')

plt.title('Random Forest Prediction')

plt.legend()

plt.show()

from xgboost import XGBRegressor

# Create model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
xgb_pred = xgb_model.predict(X_test)

# Metrics
xgb_mae = mean_absolute_error(y_test, xgb_pred)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print("XGBoost MAE:", xgb_mae)
print("XGBoost RMSE:", xgb_rmse)

plt.figure(figsize=(12,6))

plt.plot(y_test[:100], label='Actual')
plt.plot(xgb_pred[:100], label='XGBoost')

plt.xlabel('Sample')
plt.ylabel('Glucose')

plt.title('XGBoost Prediction')

plt.legend()

plt.show()

# Reshape for LSTM

X_train_lstm = X_train.reshape(
    (X_train.shape[0], X_train.shape[1], 1)
)

X_test_lstm = X_test.reshape(
    (X_test.shape[0], X_test.shape[1], 1)
)

print(X_train_lstm.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create model
lstm_model = Sequential()

lstm_model.add(
    LSTM(50, activation='relu', input_shape=(6,1))
)

lstm_model.add(Dense(1))

# Compile
lstm_model.compile(
    optimizer='adam',
    loss='mse'
)

# Train
history = lstm_model.fit(
    X_train_lstm,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# Predict
lstm_pred = lstm_model.predict(X_test_lstm)

# Metrics
lstm_mae = mean_absolute_error(y_test, lstm_pred)

lstm_rmse = np.sqrt(
    mean_squared_error(y_test, lstm_pred)
)

print("LSTM MAE:", lstm_mae)
print("LSTM RMSE:", lstm_rmse)


plt.figure(figsize=(12,6))

plt.plot(y_test[:100], label='Actual')
plt.plot(lstm_pred[:100], label='LSTM')

plt.xlabel('Sample')
plt.ylabel('Glucose')

plt.title('LSTM Prediction')

plt.legend()

plt.show()



#30 min

# 30 minute prediction split

X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(
    X,
    y_30,
    test_size=0.2,
    random_state=42
)


# Linear Regression 30 min

lr_30 = LinearRegression()

lr_30.fit(X_train_30, y_train_30)

pred_30_lr = lr_30.predict(X_test_30)

mae_30_lr = mean_absolute_error(y_test_30, pred_30_lr)

rmse_30_lr = np.sqrt(
    mean_squared_error(y_test_30, pred_30_lr)
)

print("30 Min Linear MAE:", mae_30_lr)
print("30 Min Linear RMSE:", rmse_30_lr)


# Random Forest 30 min

rf_30 = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_30.fit(X_train_30, y_train_30)

pred_30_rf = rf_30.predict(X_test_30)

mae_30_rf = mean_absolute_error(y_test_30, pred_30_rf)

rmse_30_rf = np.sqrt(
    mean_squared_error(y_test_30, pred_30_rf)
)

print("30 Min RF MAE:", mae_30_rf)
print("30 Min RF RMSE:", rmse_30_rf)


# XGBoost 30 min

xgb_30 = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

xgb_30.fit(X_train_30, y_train_30)

pred_30_xgb = xgb_30.predict(X_test_30)

mae_30_xgb = mean_absolute_error(y_test_30, pred_30_xgb)

rmse_30_xgb = np.sqrt(
    mean_squared_error(y_test_30, pred_30_xgb)
)

print("30 Min XGB MAE:", mae_30_xgb)
print("30 Min XGB RMSE:", rmse_30_xgb)


# Reshape

X_train_30_lstm = X_train_30.reshape(
    (X_train_30.shape[0], X_train_30.shape[1], 1)
)

X_test_30_lstm = X_test_30.reshape(
    (X_test_30.shape[0], X_test_30.shape[1], 1)
)


# LSTM 30 min

lstm_30 = Sequential()

lstm_30.add(
    LSTM(50, activation='relu', input_shape=(6,1))
)

lstm_30.add(Dense(1))

lstm_30.compile(
    optimizer='adam',
    loss='mse'
)

lstm_30.fit(
    X_train_30_lstm,
    y_train_30,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

pred_30_lstm = lstm_30.predict(X_test_30_lstm)

mae_30_lstm = mean_absolute_error(
    y_test_30,
    pred_30_lstm
)

rmse_30_lstm = np.sqrt(
    mean_squared_error(
        y_test_30,
        pred_30_lstm
    )
)

print("30 Min LSTM MAE:", mae_30_lstm)
print("30 Min LSTM RMSE:", rmse_30_lstm)


#60 min prediction split
# 60 minute prediction split

X_train_60, X_test_60, y_train_60, y_test_60 = train_test_split(
    X,
    y_60,
    test_size=0.2,
    random_state=42
)

# Linear Regression 60 min

lr_60 = LinearRegression()

lr_60.fit(X_train_60, y_train_60)

pred_60_lr = lr_60.predict(X_test_60)

mae_60_lr = mean_absolute_error(
    y_test_60,
    pred_60_lr
)

rmse_60_lr = np.sqrt(
    mean_squared_error(
        y_test_60,
        pred_60_lr
    )
)

print("60 Min Linear MAE:", mae_60_lr)
print("60 Min Linear RMSE:", rmse_60_lr)


# Random Forest 60 min

rf_60 = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_60.fit(X_train_60, y_train_60)

pred_60_rf = rf_60.predict(X_test_60)

mae_60_rf = mean_absolute_error(
    y_test_60,
    pred_60_rf
)

rmse_60_rf = np.sqrt(
    mean_squared_error(
        y_test_60,
        pred_60_rf
    )
)

print("60 Min RF MAE:", mae_60_rf)
print("60 Min RF RMSE:", rmse_60_rf)

# XGBoost 60 min

xgb_60 = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

xgb_60.fit(X_train_60, y_train_60)

pred_60_xgb = xgb_60.predict(X_test_60)

mae_60_xgb = mean_absolute_error(
    y_test_60,
    pred_60_xgb
)

rmse_60_xgb = np.sqrt(
    mean_squared_error(
        y_test_60,
        pred_60_xgb
    )
)

print("60 Min XGB MAE:", mae_60_xgb)
print("60 Min XGB RMSE:", rmse_60_xgb)

# Reshape for LSTM

X_train_60_lstm = X_train_60.reshape(
    (X_train_60.shape[0], X_train_60.shape[1], 1)
)

X_test_60_lstm = X_test_60.reshape(
    (X_test_60.shape[0], X_test_60.shape[1], 1)
)

# LSTM 60 min

lstm_60 = Sequential()

lstm_60.add(
    LSTM(50, activation='relu', input_shape=(6,1))
)

lstm_60.add(Dense(1))

lstm_60.compile(
    optimizer='adam',
    loss='mse'
)

lstm_60.fit(
    X_train_60_lstm,
    y_train_60,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

pred_60_lstm = lstm_60.predict(
    X_test_60_lstm
)

mae_60_lstm = mean_absolute_error(
    y_test_60,
    pred_60_lstm
)

rmse_60_lstm = np.sqrt(
    mean_squared_error(
        y_test_60,
        pred_60_lstm
    )
)

print("60 Min LSTM MAE:", mae_60_lstm)
print("60 Min LSTM RMSE:", rmse_60_lstm)

plt.figure(figsize=(12,6))

plt.plot(
    y_test_30[:100],
    label='Actual'
)

plt.plot(
    pred_30_lr[:100],
    label='Linear Regression'
)

plt.xlabel('Sample')

plt.ylabel('Glucose (mg/dL)')

plt.title('30-Minute Glucose Prediction')

plt.legend()

plt.show()


plt.figure(figsize=(12,6))

plt.plot(
    y_test_60[:100],
    label='Actual'
)

plt.plot(
    pred_60_rf[:100],
    label='Random Forest'
)

plt.xlabel('Sample')

plt.ylabel('Glucose (mg/dL)')

plt.title('60-Minute Glucose Prediction')

plt.legend()

plt.show()

models = ['Linear', 'RF', 'XGB', 'LSTM']

rmse_15 = [15.84, 16.35, 16.05, 16.80]
rmse_30 = [24.58, 25.38, 24.89, 25.29]
rmse_60 = [38.64, 38.68, 39.09, 40.11]

x = np.arange(len(models))

width = 0.25

plt.figure(figsize=(10,6))

plt.bar(x - width, rmse_15, width, label='15 min')

plt.bar(x, rmse_30, width, label='30 min')

plt.bar(x + width, rmse_60, width, label='60 min')

plt.xticks(x, models)

plt.ylabel('RMSE')

plt.title('RMSE Comparison Across Prediction Horizons')

plt.legend()

plt.show()