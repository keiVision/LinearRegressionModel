class DATA_FILE_PATH:
    #file_path = '/home/kei/Desktop/Projects/Portfolio/LinearRegression/PMiC_Ai/data/data.csv' #FOR LOCAL
    file_path = 'data/data.csv' #FOR SERVER

    @staticmethod
    def test_path(path):
        import pandas as pd

        try:
            pd.read_csv(path)
            print('\nДанные успешно загружены...')

        except FileNotFoundError:
            print('\nПуть до файла данных неверный.')
            while True:
                input_path = input('\nВведите корректный путь до файла данных: ')
                if input_path.lower() == 'exit':
                    print('\nОстановка программы...')
                    return False
                
                try:
                    pd.read_csv(input_path)
                    DATA_FILE_PATH.file_path = input_path
                    print('\nУспешно. Запуск программы...')
                    return False
                
                except FileNotFoundError:
                    print('\nНеверный путь. Попробуйте еще раз. Для выхода введите "exit"')
                    continue

#TESTING:
# testing = DATA_FILE_PATH.file_path
# print(testing)
# DATA_FILE_PATH.test_path(testing)