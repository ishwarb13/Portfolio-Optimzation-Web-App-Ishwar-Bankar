from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.svm import SVR

def get_stock_predictions(stock_selected):
   

    data = pd.read_csv(f"data/{stock_selected}_stock_data.csv")
    data = data.dropna()

    # Features and target variable
    features = data.columns[6:]
    X = data[features]
    y = data['Close']

    # Train-test split
    split_point = int(len(data) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # Scaling
    X_scaler = RobustScaler()
    y_scaler = RobustScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    model=SVR(C= 10, epsilon=0.01, gamma='scale', kernel='linear')
    # # Create GridSearchCV object
    # grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
    #                            cv=tscv, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

    # Fit GridSearchCV
    model.fit(X_train_scaled, y_train_scaled)

    # Best hyperparameters
    # best_params = grid_search.best_params_
    # print(f"Best hyperparameters for {stock}: {best_params}")

    # # Train the model using best hyperparameters
    # best_svr_model = SVR(**best_params)
    # best_svr_model.fit(X_train_scaled, y_train_scaled)

    # Make predictions
    # y_pred_scaled = best_svr_model.predict(X_test_scaled)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    return list(y_test), list(y_pred)