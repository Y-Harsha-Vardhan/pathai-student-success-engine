import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(data_dir='archive'):
    # load raw CSVs and apply early-window filtering
    print("Loading datasets...")

    df_info = pd.read_csv(os.path.join(data_dir, 'studentInfo.csv'))
    df_vle = pd.read_csv(os.path.join(data_dir, 'studentVle.csv'))
    df_student_assess = pd.read_csv(os.path.join(data_dir, 'studentAssessment.csv'))
    df_assessments = pd.read_csv(os.path.join(data_dir, 'assessments.csv'))

    # studentAssessment doesn't have module/presentation, so we map it via assessments.csv
    assessment_map = df_assessments[['id_assessment', 'code_module', 'code_presentation']].drop_duplicates()
    df_assess = df_student_assess.merge(assessment_map, on='id_assessment', how='inner')

    # keep only days 0–41 (early warning window)
    # anything after this would leak future information
    print("Applying strict day-41 cutoff...")
    df_vle = df_vle[(df_vle['date'] >= 0) & (df_vle['date'] <= 41)].copy()
    df_assess = df_assess[(df_assess['date_submitted'] >= 0) & (df_assess['date_submitted'] <= 41)].copy()

    # target: 1 = fail/withdrawn, 0 = pass/distinction
    print("Mapping target variable...")
    df_info['target'] = df_info['final_result'].isin(['Withdrawn', 'Fail']).astype(int)

    # same student can appear across multiple presentations
    # to keep things clean, just take their first one
    df_info = (
        df_info
        .sort_values(['id_student', 'code_presentation'])
        .drop_duplicates(subset=['id_student'], keep='first')
    )

    # filter activity data to match this selection
    presentation_map = df_info[['id_student', 'code_module', 'code_presentation']]

    df_vle = df_vle.merge(
        presentation_map,
        on=['id_student', 'code_module', 'code_presentation'],
        how='inner'
    )

    df_assess = df_assess.merge(
        presentation_map,
        on=['id_student', 'code_module', 'code_presentation'],
        how='inner'
    )

    return df_info, df_vle, df_assess


if __name__ == "__main__":
    df_info, df_vle, df_assess = load_and_prepare_data('archive')

    print(f"\nStudents after dedup: {len(df_info)}")
    print(f"VLE rows in window:   {len(df_vle)}")
    print(f"Assess rows in window:{len(df_assess)}")

    print("\nTarget distribution:")
    print(df_info['target'].value_counts())

    print("\nPresentations in data:")
    print(df_info['code_presentation'].value_counts())