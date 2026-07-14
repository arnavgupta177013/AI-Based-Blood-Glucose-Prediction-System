# ============================================================
# GLUCOSE PREDICTION USING MACHINE LEARNING & DEEP LEARNING
# OhioT1DM Dataset
# ============================================================

# ---------------- IMPORT LIBRARIES ----------------

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ============================================================
# LOAD DATASET
# ============================================================

tree = ET.parse("G:/Downloads/archive/591-ws-testing.xml")
root = tree.getroot()

timestamps = []
glucose = []

for event in root.find('glucose_level').findall('event'):
    timestamps.append(event.get('ts'))
    glucose.append(float(event.get('value')))

df = pd.DataFrame({
    'timestamp': timestamps,
    'glucose': glucose
})

df['timestamp'] = pd.to_datetime(
    df['timestamp'],
    format='%d-%m-%Y %H:%M:%S'
)

print(df.head())

# ============================================================
# CREATE SLIDING WINDOW DATA
# ============================================================

window_size = 6

X = []
y_15 = []
y_30 = []
y_60 = []

glucose_values = df['glucose'].values

for i in range(window_size, len(glucose_values) - 12):

    X.append(glucose_values[i-window_size:i])

    y_15.append(glucose_values[i+3])

    y_30.append(glucose_values[i+6])

    y_60.append(glucose_values[i+12])

X = np.array(X)

y_15 = np.array(y_15)

y_30 = np.array(y_30)

y_60 = np.array(y_60)

print("X shape:", X.shape)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return predictions, mae, rmse


def build_lstm():

    model = Sequential()

    model.add(
        LSTM(
            50,
            activation='relu',
            input_shape=(6,1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


def evaluate_lstm(X_train, X_test, y_train, y_test):

    X_train_lstm = X_train.reshape(
        (X_train.shape[0], X_train.shape[1], 1)
    )

    X_test_lstm = X_test.reshape(
        (X_test.shape[0], X_test.shape[1], 1)
    )

    model = build_lstm()

    model.fit(
        X_train_lstm,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    predictions = model.predict(X_test_lstm, verbose=0)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return predictions, mae, rmse


# ============================================================
# STORE RESULTS
# ============================================================

results = {}

# ============================================================
# 15 MINUTE PREDICTION
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_15,
    test_size=0.2,
    random_state=42
)

# Linear Regression
lr_model = LinearRegression()

pred_15_lr, mae_15_lr, rmse_15_lr = evaluate_model(
    lr_model,
    X_train,
    X_test,
    y_train,
    y_test
)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

pred_15_rf, mae_15_rf, rmse_15_rf = evaluate_model(
    rf_model,
    X_train,
    X_test,
    y_train,
    y_test
)

# XGBoost
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

pred_15_xgb, mae_15_xgb, rmse_15_xgb = evaluate_model(
    xgb_model,
    X_train,
    X_test,
    y_train,
    y_test
)

# LSTM
pred_15_lstm, mae_15_lstm, rmse_15_lstm = evaluate_lstm(
    X_train,
    X_test,
    y_train,
    y_test
)

results['15 min'] = {
    'Linear': rmse_15_lr,
    'RF': rmse_15_rf,
    'XGB': rmse_15_xgb,
    'LSTM': rmse_15_lstm
}

# ============================================================
# 30 MINUTE PREDICTION
# ============================================================

X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(
    X,
    y_30,
    test_size=0.2,
    random_state=42
)

# Linear
pred_30_lr, mae_30_lr, rmse_30_lr = evaluate_model(
    LinearRegression(),
    X_train_30,
    X_test_30,
    y_train_30,
    y_test_30
)

# RF
pred_30_rf, mae_30_rf, rmse_30_rf = evaluate_model(
    RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    X_train_30,
    X_test_30,
    y_train_30,
    y_test_30
)

# XGB
pred_30_xgb, mae_30_xgb, rmse_30_xgb = evaluate_model(
    XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ),
    X_train_30,
    X_test_30,
    y_train_30,
    y_test_30
)

# LSTM
pred_30_lstm, mae_30_lstm, rmse_30_lstm = evaluate_lstm(
    X_train_30,
    X_test_30,
    y_train_30,
    y_test_30
)

results['30 min'] = {
    'Linear': rmse_30_lr,
    'RF': rmse_30_rf,
    'XGB': rmse_30_xgb,
    'LSTM': rmse_30_lstm
}

# ============================================================
# 60 MINUTE PREDICTION
# ============================================================

X_train_60, X_test_60, y_train_60, y_test_60 = train_test_split(
    X,
    y_60,
    test_size=0.2,
    random_state=42
)

# Linear
pred_60_lr, mae_60_lr, rmse_60_lr = evaluate_model(
    LinearRegression(),
    X_train_60,
    X_test_60,
    y_train_60,
    y_test_60
)

# RF
pred_60_rf, mae_60_rf, rmse_60_rf = evaluate_model(
    RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    X_train_60,
    X_test_60,
    y_train_60,
    y_test_60
)

# XGB
pred_60_xgb, mae_60_xgb, rmse_60_xgb = evaluate_model(
    XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ),
    X_train_60,
    X_test_60,
    y_train_60,
    y_test_60
)

# LSTM
pred_60_lstm, mae_60_lstm, rmse_60_lstm = evaluate_lstm(
    X_train_60,
    X_test_60,
    y_train_60,
    y_test_60
)

results['60 min'] = {
    'Linear': rmse_60_lr,
    'RF': rmse_60_rf,
    'XGB': rmse_60_xgb,
    'LSTM': rmse_60_lstm
}

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n================ FINAL RESULTS ================\n")

for horizon in results:

    print(horizon)

    for model in results[horizon]:

        print(
            f"{model} RMSE: "
            f"{results[horizon][model]:.2f}"
        )

    print()

# ============================================================
# SUBPLOT FIGURE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18,5))

# 15 min
axes[0].plot(
    y_test[:100],
    label='Actual'
)

axes[0].plot(
    pred_15_lr[:100],
    label='Linear'
)

axes[0].set_title('15 Minute Prediction')

# 30 min
axes[1].plot(
    y_test_30[:100],
    label='Actual'
)

axes[1].plot(
    pred_30_lr[:100],
    label='Linear'
)

axes[1].set_title('30 Minute Prediction')

# 60 min
axes[2].plot(
    y_test_60[:100],
    label='Actual'
)

axes[2].plot(
    pred_60_rf[:100],
    label='RF'
)

axes[2].set_title('60 Minute Prediction')

for ax in axes:

    ax.set_xlabel('Sample')

    ax.set_ylabel('Glucose (mg/dL)')

    ax.legend()

plt.tight_layout()

plt.show()

# ============================================================
# RMSE BAR CHART
# ============================================================

models = ['Linear', 'RF', 'XGB', 'LSTM']

rmse_15 = [
    results['15 min']['Linear'],
    results['15 min']['RF'],
    results['15 min']['XGB'],
    results['15 min']['LSTM']
]

rmse_30 = [
    results['30 min']['Linear'],
    results['30 min']['RF'],
    results['30 min']['XGB'],
    results['30 min']['LSTM']
]

rmse_60 = [
    results['60 min']['Linear'],
    results['60 min']['RF'],
    results['60 min']['XGB'],
    results['60 min']['LSTM']
]

x = np.arange(len(models))

width = 0.25

plt.figure(figsize=(10,6))

plt.bar(
    x - width,
    rmse_15,
    width,
    label='15 min'
)

plt.bar(
    x,
    rmse_30,
    width,
    label='30 min'
)

plt.bar(
    x + width,
    rmse_60,
    width,
    label='60 min'
)

plt.xticks(x, models)

plt.ylabel('RMSE')

plt.xlabel('Models')

plt.title('RMSE Comparison Across Prediction Horizons')

plt.legend()

plt.show()

# ============================================================
# END
# ============================================================