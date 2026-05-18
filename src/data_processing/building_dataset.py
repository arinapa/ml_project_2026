
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder


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


def make_features_for_status(df : pd.DataFrame) -> pd.DataFrame:
    """
    Создает датафрейм для предсказания статуса студента

    Parameters
    df : pd.DataFrame
        Исходный датафрейм

    

    Returns
    pd.DataFrame
        Датафрейм с колонками :
            'student__id_hash'
            'avg_grade'
            'count_grades' -- количество оценок
            'proportion_retake' -- доля пересдач 
            'proportion_retake_com' -- доля пересдач с комисией
            'program' -- категория (OneHotEncoder)
            'course' -- категория (OneHotEncoder)
            'place_type' -- категория (OneHotEncoder)
            'student_status'

    """
    
    result = []
    
    for student_id_hash, group in df.groupby('student_id_hash'):
        grades = group['grade_10'].tolist()
        avg_grade = sum(grades) / len(grades)

        
        student_status = group['student_status'].iloc[0]
        
        
        result.append({
            'student_id_hash': student_id_hash,
            # 'grades_list': grades,
            'avg_grade': avg_grade,
            # 'median_grade': statistics.median(grades),
            # 'min_grade': min(grades),
            # 'max_grade': max(grades),
            'count_grades': len(grades),
            # 'count_retake': len(group[group['exam_type'] == 'Пересдача']),
            'proportion_retake': len(group[group['exam_type'] == 'Пересдача'])/len(grades),
            # 'count_retake_com': len(group[group['exam_type'] == 'Пересдача с комиссией']),
            'proportion_retake_com': len(group[group['exam_type'] == 'Пересдача с комиссией'])/len(grades),
            'program' : group['program'].iloc[0],
            'course': group['course'].iloc[0],
            'place_type': group['place_type'].iloc[0],
            'student_status': student_status

        })
    df_encoded = pd.DataFrame(result)
    df_encoded['course'] = df_encoded['course'].astype('category')
    
    categorical_cols = ['course', 'program', 'place_type']


    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df_encoded[categorical_cols])

    encoded_df = pd.DataFrame(
                encoded_array,
                columns=encoder.get_feature_names_out(categorical_cols)
    )

    df_final = pd.concat([df_encoded.drop(columns=categorical_cols), encoded_df], axis=1)
    df_final

    
    
    return df_final

def make_features_for_grade(df : pd.DataFrame, course : int, categorical_cols : list = None) -> pd.DataFrame:
    """
    Создает датафрейм для предсказания оценки студента в 4 модуле

    Parameters
    df : pd.DataFrame
        Исходный датафрейм
    course : int
        Номер курса, 4 модуль которого предсказываем
     categorical_cols : list
        Список столбцов, которые надо зарбить через OneHotEncoder
    

    Returns
    pd.DataFrame
        Датафрейм с добавленными колонками:
           'mean_1_grade', 'mean_2_grade', 'mean_3_grade' - среднее по модулям 
           'std_1_grade', 'std_2_grade', 'std_3_grade' - среднее по модулям
           'trend_1-2', 'trend_2-3' - изменение оценок по модулям
           'low_grades_ratio' - доля плохих оценок за 2 модуль
            И разбитые через OneHotEncoder столбцы

    """
    module3_cols = df.filter(like=f'{course}_3_Дисциплина:').columns
    module2_cols = df.filter(like=f'{course}_2_Дисциплина:').columns
    module1_cols = df.filter(like=f'{course}_1_Дисциплина:').columns

    df['mean_3_grade'] = df[module3_cols].mean(axis=1)
    df['mean_2_grade'] = df[module2_cols].mean(axis=1)
    df['mean_1_grade'] = df[module1_cols].mean(axis=1)

    df['std_3_grade'] = df[module3_cols].std(axis=1)
    df['std_2_grade'] = df[module2_cols].std(axis=1)
    df['std_1_grade'] = df[module1_cols].std(axis=1)

    df['trend_2-3'] = df[module3_cols].mean(axis=1) - df[module2_cols].mean(axis=1)
    df['trend_1-2'] = df[module2_cols].mean(axis=1) - df[module1_cols].mean(axis=1)
    df['low_grades_ratio'] = (df[module2_cols] < 4).mean(axis=1)
    
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = encoder.fit_transform(df[categorical_cols])

        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    return df


# def main():

#     df = pd.read_csv("data/raw/grades.csv")
#     df_program = make_data_from_program(df, 'Международный бакалавриат по бизнесу и экономике', [])
#     df_clean = nan_cleaner(df_clean_mb, 0.3, 600,10, True)
#     df_clean = fill_na_knn(df_cleaned_nan_mb, 5, drop_col=['student_id_hash'], printer = True)

#     df_clean.to_csv("data/clean/prob1.csv", index=False)

#     # print("Done:", df_clean.shape)


# if __name__ == "__main__":
#     main()