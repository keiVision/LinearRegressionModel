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
X = df.drop(columns = ['Final_price','Hum_days', 'Process_volume', 'Стоимость ч/ч с налогами'])
y_days = df[['Hum_days']]
y_volume = df[['Process_volume']]
    
X_train, X_test, y_train_days, y_test_days, y_train_volume, y_test_volume = train_test_split(X, 
                                                                                            y_days,
                                                                                            y_volume,
                                                                                            test_size = 0.2,
                                                                                            random_state = 0)

model_instance.fit_model(X_train, y_train_days, y_train_volume)

predicted_value_days = model_instance.predict_days(X_test)
predicted_value_volume = model_instance.predict_volume(X_test)
#print(predicted_value)

mae_days, r2_days , mae_volume, r2_volume = model_instance.evaluate(y_test_days, predicted_value_days, 
                                            y_test_volume, predicted_value_volume)
print(f'\nDAYS: MAE: {mae_days}\nR2: {r2_days}\n\nVOLUME: MAE: {mae_volume}\nR2: {r2_volume}')
#DAYS: [MAE: 336.7087139661466, R2: 0.9626020023972922], VOLUME: [MAE: 5047.060998058295, R2: 0.9253059617810911]   
