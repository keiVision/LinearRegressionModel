from data_processor import *
from config import *
from model import *
from sklearn.model_selection import train_test_split

class TrainModel:

    def __init__(self):
        self.path = DATA_FILE_PATH.file_path

        if not DATA_FILE_PATH.test_path(self.path):
            self.path = DATA_FILE_PATH.file_path

        self.data = DataProcessor()

        self.df = self.data.create_df()

        self.X = self.df.drop(columns = ['Hum_days', 'Process_volume', 'Final_price', 'Стоимость ч/ч с налогами'])
        
        self.y_days = self.df[['Hum_days']]
        self.y_volume = self.df[['Process_volume']]

        self.X_train, self.X_test, self.y_train_days, self.y_test_days, self.y_train_volume, self.y_test_volume = train_test_split(self.X, 
                                                                                                                                   self.y_days,
                                                                                                                                   self.y_volume,
                                                                                                                                   test_size=0.2, 
                                                                                                                                   random_state=42)
        self.model = Model()
        self.model.create_model()
        self.model.fit_model(self.X_train,
                             self.y_train_days,
                             self.y_train_volume)

# TESTING:

# train_instance = TrainModel()
# print(train_instance.y_test_days)
        


