from data_processor import *
from config import *
from model import *

class TrainModel:

    def __init__(self):
        self.path = DATA_FILE_PATH.file_path

        if not DATA_FILE_PATH.test_path(self.path):
            self.path = DATA_FILE_PATH.file_path

        self.data = DataProcessor()

        self.df = self.data.create_df()
        self.train_df = self.df.drop(columns = ['Process_volume', 'Final_price', 'Стоимость ч/ч с налогами'])

        self.X_train, self.X_test, self.y_train, self.y_test = (DataProcessor.split_data(
                                                                self.train_df, ['Hum_days'],
                                                                test_size = 0.2, random_state = 42))
        self.model = Model()
        self.model.create_model()
        self.model.fit_model(self.X_train,
                             self.y_train)

# TESTING:

# train_instance = TrainModel()
# print(train_instance.df)
# print(train_instance.train_df)
# print(train_instance.model.predict_days(X_test))
        


