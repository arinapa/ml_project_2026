import pandas as pd
import numpy as np




def load_data(path: str = '../data/raw/grades.csv') -> pd.DataFrame:

    
    """
    Загружает датасет с сепаратором ';'

    Parameters

    path : str
        Путь до файла .csv по умолчанию = '../data/raw/grades.csv'


    Returns
    pd.DataFrame
   
    """
    return pd.read_csv(path, sep = ';')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    """
    Удаляет дубликаты строк
    Меняет строковые '\\N' на np.nan, числа в виде строк в int
    Меняет nan в колонке  'absence_status' на 'attendance' при наличии оценки за экзамен

    Parameters

    df: pd.DataFrame
        Датафрейм для обработки


    Returns
    pd.DataFrame
        Обработанный датафрейм
   
    """
    
    df = df.drop_duplicates()
    
    df = df.replace({'\\N' : np.nan, '0' : 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 
                     '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '2022' : 2022})
    df.loc[
    df['grade_10'].isin(range(11)) & df['absence_status'].isna(),
        'absence_status'
    ] = 'attendance'
    
    return df




def process_data(df: pd.DataFrame) -> pd.DataFrame:

    """
    Удаляет записи со студентами, имеющими более 1 статуса 
    Удаляет колонку 'subject_unit'
    Удаляет строки с nan

    Parameters

    df: pd.DataFrame
        Датафрейм для обработки


    Returns
    pd.DataFrame
        Обработанный датафрейм
   
    """
    inconsistent = df.groupby('student_id_hash')['student_status'].nunique()
    bad_ids = inconsistent[inconsistent > 1].index
    df_clean = df[~df['student_id_hash'].isin(bad_ids)]
    

    df_clean = df_clean.drop(columns=['subject_unit'])
    df_clean= df_clean.dropna()

    # students_info = df_clean[['student_id_hash', 'campus', 'group', 'faculty', 'place_type', 'program','education_level', 'student_status' ]].copy()
    

    return df_clean

def drop_columns(df: pd.DataFrame, delete_columns: list) -> pd.DataFrame:
    """
    Удаляет колонки по списку 

    Parameters

    df: pd.DataFrame
        Датафрейм для обработки
    delete_columns: list
        Список колонок для удаления


    Returns
    pd.DataFrame
        Обработанный датафрейм
   
    """
    df_clean = df.drop(columns=delete_columns) 
    return df_clean


def save_data(df: pd.DataFrame, path: str) -> None:
    """
    Сохраняет датафрейм по указанному пути

    Parameters

    df: pd.DataFrame
        Датафрейм для сохранения 
    path: str
        Путь для сохранения
   
    """
    df.to_csv(path, index=False)



    
    
    


def str_to_int_student_status (df : pd.DataFrame):
    """
    Заменяет "student_status" на два типа 0 -- выпустился или учится, 1 -- отчислили или отчислился

    Parameters

    df: pd.DataFrame
        Датафрейм для обработки
    


    Returns
    pd.DataFrame
        Обработанный датафрейм
   
    """

    df["student_status"] = df["student_status"].map({
        "study": 0,
        "graduated": 0,
        "expelled": 1,
        "leave" : 1
    })
    return df


def main():
    df = load_data('../data/raw/grades.csv')
    # validate_data(df)
    # df = clean_data(df)
    # df = process_data(df)
    # df = drop_columns(df)

    
    # save_data(df, "output.csv")
    return df


if __name__ == "__main__":
    main()