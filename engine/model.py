from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd


class Model:

    def __init__(self):
        self.model_days = None
        self.model_volume = None

    def create_model(self):
        self.model_days = GradientBoostingRegressor()
        self.model_volume = GradientBoostingRegressor()

    def fit_model(self, X_train: pd.DataFrame, y_train_days: pd.DataFrame, y_train_volume: pd.DataFrame):
        self.model_days.fit(X_train, y_train_days)
        self.model_volume.fit(X_train, y_train_volume)

    def predict_days(self, X_test: pd.DataFrame):
        df = pd.DataFrame(X_test)
        return self.model_days.predict(df)
    
    def predict_volume(self, X_test: pd.DataFrame):
        df = pd.DataFrame(X_test)
        return self.model_volume.predict(df)
    
    def evaluate(self, y_fact_days, y_pred_days,
                        y_fact_volume, y_pred_volume):
            
            mae_days = mean_absolute_error(y_fact_days, y_pred_days)
            r2_days = r2_score(y_fact_days, y_pred_days)

            mae_volume = mean_absolute_error(y_fact_volume, y_pred_volume)
            r2_volume = r2_score(y_fact_volume, y_pred_volume)
            return mae_days, r2_days, mae_volume, r2_volume 