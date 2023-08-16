from data_processor import *
from model import *
from config import *
from train import * 
from main import *

model_instance = Model()
model_instance.create_model()
data_processor_instance = DataProcessor()

df = data_processor_instance.create_df()
# print(df)
X = df.drop(columns = ['Process_volume', 'Final_price', 'Стоимость ч/ч с налогами'])
# print(df)

X_train, X_test, y_train, y_test = data_processor_instance.split_data(X, target_columns = ['Hum_days'])
# print(X_test)

model_instance.fit_model(X_train, y_train)

predicted_value = model_instance.predict_days(X_test)
# print(predicted_value)

mae, r2 = model_instance.evaluate(y_test, predicted_value)
print(f'MAE: {mae}\nR2: {r2}')