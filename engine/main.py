from model import *
from data_processor import *
from train import *
from config import *
from sklearn.preprocessing import OneHotEncoder

class Engine:
    encoder = DataProcessor.create_basic_encoder()

    def __init__(self):
        self.trained_model = None

    def train_model(self):
        data = DataProcessor()
        df = data.create_df()
        train_df = df.drop(columns = ['Process_volume', 'Final_price', 'Стоимость ч/ч с налогами'])

        X_train, X_test, y_train, y_test = data.split_data(train_df, ['Hum_days'], 0.2, 42)
        self._TEST = X_test 

        model = Model()
        model.create_model()

        model.fit_model(X_train, 
                        y_train['Hum_days'])
        
        self.trained_model = model
    
    def check_train_status(self):
        try:
            _ = self.trained_model.predict_days(self._TEST)
            print("Процесс обучения модели прошел успешно. Модель готова к использованию.")

        except:
            print("Ошибка обучения модели. Необходимы корректировки.")

    def predict_hum_days(self, X):
        df = pd.DataFrame(X)
        
        encoded_df = pd.DataFrame(Engine.encoder.transform(df[['Process_name']]).toarray())
        encoded_df.columns = encoded_df.columns.astype(str)

        X = pd.concat([encoded_df, df['Object_area']], axis = 1)

        if self.trained_model is not None:
            return self.trained_model.predict_days(X)
        else:
            print("Модель еще не обучена.")
            return None

# TESTING: Формат передачи входных данных: json [{'col_name': 'col_value', ...}] 
# engine = Engine()
# engine.train_model()
# engine.check_train_status()

# X = [{'Object_area': 2000, 'Process_name': 'монтаж кабеля ЭОМ на потолке'}]

# print(engine.predict_hum_days(X))

