
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def make_data_from_program(df : pd.DataFrame, program: str, addictional_columns: list, subjects_list=None) -> None:\


    """
    Формирует таблицу оценок студентов для выбранной программы.

    Parameters

    df : pd.DataFrame
        Исходный датафркйм с оценками студентов.

    program : str
        Название образовательной программы.

    additional_columns : list
        Дополнительные столбцы

    subjects_list : list, optional
        Список предметов которые нужно добавить в датасет
        (subject_name, course, module)
        Если None, фильтрация не применяется.

    Returns
    pd.DataFrame
   
    """
    df_baseline = df.loc[df['program'] == program]

    if subjects_list is not None:
        df_baseline = df_baseline.loc[
            df_baseline[['subject_name', 'course', 'module']]
            .apply(tuple, axis=1)
            .isin(subjects_list)
        ]

    
    
    
    df_baseline_encoded = df_baseline.pivot_table(
        index='student_id_hash',
        columns=['course', 'module', 'subject_name'],
        values='grade_10',
        aggfunc = 'mean',
        observed=True
    )

    df_baseline_encoded.columns = [
        "_".join(map(str, col)).strip()
        for col in df_baseline_encoded.columns
    ]
    df_baseline_encoded = df_baseline_encoded.reset_index()

    df_baseline_encoded = df_baseline_encoded.merge(
        df_baseline[['student_id_hash'] + addictional_columns],
        on='student_id_hash',
        how='left'
    )
    df_baseline_encoded = df_baseline_encoded.drop_duplicates()
    return df_baseline_encoded





def nan_cleaner(
    df: pd.DataFrame,
    max_nan_fraction: float = 0.3,
    min_rows: int = 50,
    min_cols: int = 5,
    printer: bool = False
) -> pd.DataFrame:
    """
    Итеративно удаляет строки и столбцы с наибольшим количеством NaN,
    сохраняя максимально возможный размер датафрейма
    Parameters
    
    df : pd.DataFrame
        Исходный датафрейм

    max_nan_fraction : float, default=0.3
        Максимально допустимая доля NaN
        во всём датафрейме

    min_rows : int, default=50
        Минимально допустимое количество строк

    min_cols : int, default=5
        Минимально допустимое количество столбцов

    verbose : bool, default=True
        Выводить информацию об удалениях.

    Returns
    pd.DataFrame
        Очищенный датафрейм
    """

    clean_df = df.copy()

    while True:

        total_nan_fraction = clean_df.isna().mean().mean()

        if total_nan_fraction <= max_nan_fraction:
            reason = "Достигнут допустимый уровень NaN"
            break

        if clean_df.shape[0] <= min_rows:
            reason = "Достигнут минимум строк"
            break

        if clean_df.shape[1] <= min_cols:
            reason = "Достигнут минимум столбцов"
            break

        row_nan = clean_df.isna().mean(axis=1)
        col_nan = clean_df.isna().mean(axis=0)

        worst_row = row_nan.idxmax()
        worst_col = col_nan.idxmax()

        if row_nan.max() > col_nan.max():


            clean_df = clean_df.drop(index=worst_row)

        else:

            clean_df = clean_df.drop(columns=worst_col)
    if (printer):
        print(reason)
    

    return clean_df




def fill_na_knn(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    drop_col: list = None,
    printer: bool = False
) -> pd.DataFrame:
    """
    Заполняет пропуски с помощью KNNImputer (
    Метод основан на поиске ближайших соседей:
    пропущенные значения заменяются значениями
    похожих объектов)

    Parameters
    df : pd.DataFrame
        Исходный датафрейм

    n_neighbors : int, default=5
        Количество ближайших соседей

    drop_col: list, default = []
        Нечисловые колонки, которые не надо заполнять

    printer : bool, default=False
        Выводить статистику заполнения

    Returns
    pd.DataFrame
        Датафрейм 

    """
    if drop_col is None:
        drop_col = []

    df_copy = df.copy()
    non_numeric = df_copy[drop_col].copy() if drop_col else pd.DataFrame()
    df_copy = df_copy.drop(columns=drop_col)
    
    df_copy = df_copy.apply(pd.to_numeric, errors='coerce')

    

    numeric_df = df_copy.select_dtypes(include='number')

    nan_before = numeric_df.isna().sum().sum()

    imputer = KNNImputer(
        n_neighbors=n_neighbors
    )

    numeric_imputed = pd.DataFrame(
        imputer.fit_transform(numeric_df),
        columns=numeric_df.columns,
        index=numeric_df.index
    )

    result_df = pd.concat(
        [numeric_imputed, non_numeric],
        axis=1
    )

    nan_after = result_df.isna().sum().sum()

    if printer:

        print(f"Размер датафрейма: {result_df.shape}")
        print()
        print(f"NaN до: {nan_before}")
        print(f"NaN после: {nan_after}")

    return result_df


def make_features(df : pd.DataFrame) -> pd.DataFrame:
    """
    Создает датафрейм для предсказания статуса студента

    Parameters
    df : pd.DataFrame
        Исходный датафрейм

    

    Returns
    pd.DataFrame
        Датафрейм с колонками :
            'student__id_hash'
            'grades_list'
            'avg_grade'
            'student_status'

    """
    
    result = []
    
    for student_id_hash, group in df.groupby('student_id_hash'):
        grades = group['grade_10'].tolist()
        avg_grade = sum(grades) / len(grades)

        
        student_status = group['student_status'].iloc[0]
        
        result.append({
            'student__id_hash': student_id_hash,
            'grades_list': grades,
            'avg_grade': avg_grade,
            'student_status': student_status
        })

    
    
    return pd.DataFrame(result)


# def main():

#     df = pd.read_csv("data/raw/grades.csv")
#     df_program = make_data_from_program(df, 'Международный бакалавриат по бизнесу и экономике', [])
#     df_clean = nan_cleaner(df, printer=True)
#     df_clean = fill_na_knn(df_clean, 5, True)

#     df_clean.to_csv("data/clean/prob1.csv", index=False)

#     # print("Done:", df_clean.shape)


# if __name__ == "__main__":
#     main()