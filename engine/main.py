from model import *
from data_processor import *
from train import *
from config import *

class Engine:
    encoder = DataProcessor.create_basic_encoder()

    def __init__(self):
        self.train_model_instance = TrainModel()
        self.trained_model = self.train_model_instance.model

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