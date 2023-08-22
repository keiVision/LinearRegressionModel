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
y_days = df[['Hum_days']].values.ravel()
y_volume = df[['Process_volume']].values.ravel()

X_train, X_test, y_train_days, y_test_days, y_train_volume, y_test_volume = train_test_split(X, 
                                                                                            y_days,
                                                                                            y_volume,
                                                                                            test_size = 0.2,
                                                                                            random_state = 93) #93
model_instance.fit_model(X_train, y_train_days, y_train_volume)




# TESTING WITH X_TEST
predicted_value_days = model_instance.predict_days(X_test)
predicted_value_volume = model_instance.predict_volume(X_test)
mae_days, r2_days , mae_volume, r2_volume = model_instance.evaluate(y_test_days, predicted_value_days, 
                                            y_test_volume, predicted_value_volume)
print(f'\nDAYS: MAE: {mae_days}\nR2: {r2_days}\n\nVOLUME: MAE: {mae_volume}\nR2: {r2_volume}')
#DAYS: [MAE: 179.69021343851688, R2: 0.9797166960709668], VOLUME: [MAE: 1012.5355645458415, R2: 0.9917991822527271]   




# TESTING INPUT: Формат передачи входных данных: json [{'col_name': 'col_value', ...}]
engine = Engine()

X = [{'Object_area': 18000, 'Process_name':'монтаж кабеля ЭОМ на потолке', 'Directive_perfomance': 5, 'Hour_cost': 546.49}]

days, volume = engine.predict(X)
print(f"\nПрогноз модели... \nЗатраты для времени: {int(days)} человеко-дней.\nЗатраты для объема процесса: {int(volume)} е.м.")

process_name = X[0]['Process_name']
object_area = X[0]['Object_area']
hum_hour_cost = X[0]['Hour_cost']
hum_count = int(days) // X[0]['Directive_perfomance']
final_price = (days * 10) * hum_hour_cost

print(f''' 
Название процесса: {process_name}
Площадь объекта: {object_area} метров квадратных.
Стоимость человеко-часа с учетом налога: {hum_hour_cost} рублей.
Количество человек для выполнения работы: {int(hum_count)} человек.
Количество дней на выполнение работы: {int(days)} дней.
Финальная стоимость работы: {int(final_price)} рублей.
Объем потраченных материалов: {int(volume)} единиц материала.
''')


df = pd.DataFrame([[18000, 'EOM', 5, 546.49]], columns = ['Object_area', 'Process_name', 'Directive_perfomance', 'Hour_cost'])

print(int(days * df['Hour_cost'].iloc[0]))