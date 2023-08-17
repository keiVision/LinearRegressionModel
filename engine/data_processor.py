#PROCESS DATA

import pandas as pd
import numpy as np
from config import DATA_FILE_PATH
from sklearn.preprocessing import OneHotEncoder

class DataProcessor:
    def __init__(self):
        self.path = DATA_FILE_PATH.file_path
    
    def create_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)

        if 'Unnamed: 0' and 'Unnamed: 0.1' in df.columns:
            df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
        elif 'Unnamed: 0' in df.columns:
            df = df.drop(columns = 'Unnamed: 0')

        if 'Название процесса' in df.columns:
            df = df.rename(columns={'Название процесса': 'Process_name'})

        try:
            for col in df.drop(columns = 'Process_name').columns:
                for index, row in df.iterrows():
                    value = row[col]
                    if isinstance(value, str):
                        df.at[index, col] = value.replace('\xa0', '').replace(' ', '').astype(float)
            df = self.encode_process_names(df, 'Process_name')
            print('Значения в данных были изменены на числовые. Датафрейм создан успешно.')
            return df
        except:
            df = self.encode_process_names(df, 'Process_name')
            print('Датафрейм создан без изменения значений. Возможны конфликты.')
            return df
    
    @staticmethod
    def encode_process_names(df, process_col):
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder()  # sparse=False чтобы получить массив numpy, а не разреженную матрицу

        X_encoded = encoder.fit_transform(df[[process_col]])
        X_encoded_df = pd.DataFrame(X_encoded.toarray())

        df_encoded = pd.concat([X_encoded_df, df.drop(columns=[process_col])], axis=1)
        df_encoded.columns = df_encoded.columns.astype(str)

        return df_encoded
    
    @staticmethod
    def create_basic_encoder():
        default_df = pd.read_csv(DATA_FILE_PATH.file_path)

        if 'Название процесса' in default_df.columns:
            default_df = default_df.rename(columns = {'Название процесса': 'Process_name'})

        try:
            default_df = default_df.drop(columns = ['Unnamed: 0', 'Unnamed: 0.1'])
    
        except:
            default_df = default_df.drop(columns = ['Unnamed: 0'])
        
        encoder = OneHotEncoder()
        X = default_df[['Object_area', 'Process_name']]
        encoder.fit_transform(X[['Process_name']]) 

        return encoder


#TESTING:

# data = DataProcessor()
# df = data.create_df()
# print(df)