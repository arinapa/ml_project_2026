import pandas as pd
import numpy as np




def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep = ';')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.drop_duplicates()
    
    df = df.replace({'\\N' : np.nan, '0' : 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 
                     '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '2022' : 2022}, inplace=True)
    df.loc[
    df['grade_10'].isin(range(11)) & df['absence_status'].isna(),
        'absence_status'
    ] = 'attendance'
    
    return df




def process_data(df: pd.DataFrame) -> pd.DataFrame:
    inconsistent = df.groupby('student_id_hash')['student_status'].nunique()
    bad_ids = inconsistent[inconsistent > 1].index
    df_clean = df[~df['student_id_hash'].isin(bad_ids)]
    

    df_clean = df_clean.drop(columns=['subject_unit'])
    df_clean= df_clean.dropna()

    # students_info = df_clean[['student_id_hash', 'campus', 'group', 'faculty', 'place_type', 'program','education_level', 'student_status' ]].copy()
    

    return df_clean

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.drop(columns=['campus', 'group', 'faculty', 'place_type', 'program','education_level', 'student_status']) 
    return df_clean

def validate_data(df: pd.DataFrame) -> None:
    assert df['student_id_hash'].notna().all()


def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def make_features(df : pd.DataFrame) -> pd.DataFrame:
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


    
    
    

def make_data_from_faculty(df : pd.DataFrame, faculty: str) -> None:
    df_baseline = df[df['faculty'] == faculty]

    

    df_baseline_encoded = df_baseline.pivot_table(
        index='student_id_hash',
        columns=['course', 'module', 'subject_name', 'exam_type'],
        values='grade_10',
        aggfunc = 'mean'
    )

    df_baseline_encoded.columns = [
        "_".join(map(str, col)).strip()
        for col in df_baseline_encoded.columns
    ]
    df_baseline_encoded = df_baseline_encoded.reset_index()

    df_baseline_encoded = df_baseline_encoded.merge(
        df_baseline[['student_id_hash', 'student_status']],
        on='student_id_hash',
        how='left'
    )
    df_baseline_encoded = df_baseline_encoded.drop_duplicates()
    return df_baseline_encoded


def make_data_from_program(df : pd.DataFrame, program: str) -> None:
    df_baseline = df[df['program'] == program]

    

    df_baseline_encoded = df_baseline.pivot_table(
        index='student_id_hash',
        columns=['course', 'module', 'subject_name', 'exam_type'],
        values='grade_10',
        aggfunc = 'mean'
    )

    df_baseline_encoded.columns = [
        "_".join(map(str, col)).strip()
        for col in df_baseline_encoded.columns
    ]
    df_baseline_encoded = df_baseline_encoded.reset_index()

    df_baseline_encoded = df_baseline_encoded.merge(
        df_baseline[['student_id_hash', 'student_status']],
        on='student_id_hash',
        how='left'
    )
    df_baseline_encoded = df_baseline_encoded.drop_duplicates()
    return df_baseline_encoded

def str_to_int_stadent_status (df : pd.DataFrame):
    df["student_status"] = df["student_status"].map({
        "study": 0,
        "graduated": 1,
        "expelled": 2,
        "leave" : 3
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