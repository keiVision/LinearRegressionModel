from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd


class Model:

    def __init__(self):
        self.model_days = None

    def create_model(self):
        self.model_days = LinearRegression()

    def fit_model(self, X_train: pd.DataFrame, y_train_days: pd.DataFrame):
        self.model_days.fit(X_train, y_train_days)

    def predict_days(self, X_test):
        df = pd.DataFrame(X_test)
        return self.model_days.predict(df)

    def evaluate(self, y_fact_days, y_pred_days):
            mae_days = mean_absolute_error(y_fact_days, y_pred_days)
            r2_days = r2_score(y_fact_days, y_pred_days)

            return mae_days, r2_days