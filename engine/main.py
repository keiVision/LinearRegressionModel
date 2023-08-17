from model import *
from data_processor import *
from train import *
from config import *
from sklearn.preprocessing import OneHotEncoder

class Engine:
    encoder = DataProcessor.create_basic_encoder()
    model_instance = Model()

    def __init__(self):
        self.trained_model = None

    def train_model(self):
        data = DataProcessor()
        model = Model()

        df = data.create_df()

        X = df.drop(columns = ['Final_price','Hum_days', 'Process_volume', 'Стоимость ч/ч с налогами'])
        y_days = df[['Hum_days']]
        y_volume = df[['Process_volume']]
        
        X_train, X_test, y_train_days, y_test_days, y_train_volume, y_test_volume = train_test_split(X, 
                                                                                                     y_days,
                                                                                                     y_volume,
                                                                                                     test_size = 0.2,
                                                                                                     random_state = 42)
        self._TEST_X = X_test
        self._TEST_Y_D = y_test_days
        self._TEST_Y_V = y_test_volume

        
        model.create_model()

        model.fit_model(X_train, 
                        y_train_days,
                        y_train_volume)
        
        self.trained_model = model
    
    def check_train_status(self):
        try:
            _ = self.trained_model.predict_days(self._TEST_X)
            _ = self.trained_model.predict_volume(self._TEST_X)
            print("Процесс обучения модели прошел успешно. Модель готова к использованию.")

        except:
            print("Ошибка обучения модели. Необходимы корректировки.")

    def predict(self, X: pd.DataFrame):
        df = pd.DataFrame(X)
        
        encoded_df = pd.DataFrame(Engine.encoder.transform(df[['Process_name']]).toarray())
        encoded_df.columns = encoded_df.columns.astype(str)

        X = pd.concat([encoded_df, df['Object_area']], axis = 1)

        if self.trained_model is not None:
            days_predict = self.trained_model.predict_days(X)
            volume_predict = self.trained_model.predict_volume(X)
            
            return days_predict, volume_predict
        
        else:
            print("Модель еще не обучена.")
            return None

# TESTING: Формат передачи входных данных: json [{'col_name': 'col_value', ...}]
#  
engine = Engine()
engine.train_model()
engine.check_train_status()

X_input = [{'Object_area': 2000, 'Process_name': 'монтаж кабеля ЭОМ на потолке'}]
days, volume = engine.predict(X_input)
print(f"Прогноз модели... \nЗатраты для времени: {int(days)} человеко-дней.\nЗатраты для объема процесса: {int(volume)} е.м.")

