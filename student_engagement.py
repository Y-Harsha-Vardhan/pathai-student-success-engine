import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
from sklearn.preprocessing import MinMaxScaler


def load_data(data_dir='archive'):
    # loads required CSVs and does basic preprocessing
    # date column in studentVle is relative to module start, negative = before start, skip those
    print("Loading data...")

    df_vle = pd.read_csv(os.path.join(data_dir, 'studentVle.csv'))
    df_vle = df_vle[df_vle['date'] >= 0]  # drop pre-module rows

    df_student_assess = pd.read_csv(os.path.join(data_dir, 'studentAssessment.csv'))
    df_info = pd.read_csv(os.path.join(data_dir, 'studentInfo.csv'))
    df_assess = pd.read_csv(os.path.join(data_dir, 'assessments.csv'))

    # convert day-level dates into week buckets 
    print("Converting dates to discrete weeks...")
    df_vle['week'] = df_vle['date'] // 7
    df_student_assess['week'] = df_student_assess['date_submitted'] // 7

    # assessment deadlines sometimes have '?' - convert to NaN (to prevent crashing)
    df_assess['date'] = pd.to_numeric(df_assess['date'], errors='coerce')

    # label: 1 = bad outcome (Fail or Withdrawn), 0 = good (Pass/Distinction)
    print("Mapping final_result to binary label...")
    df_info['label'] = df_info['final_result'].isin(['Withdrawn', 'Fail']).astype(int)

    # important: same student ID can appear in multiple modules
    # merging only on id_student mixes outcomes across courses (wrong)
    # so we build a composite key: student + module + presentation
    df_info['id_student'] = (
        df_info['id_student'].astype(str) + '_' +
        df_info['code_module'] + '_' +
        df_info['code_presentation']
    )
    df_labels = df_info[['id_student', 'label']].drop_duplicates()

    # apply the same composite key to vle
    df_vle['id_student'] = (
        df_vle['id_student'].astype(str) + '_' +
        df_vle['code_module'] + '_' +
        df_vle['code_presentation']
    )

    # studentAssessment doesn't directly have module/presentation
    # so map via assessments table first
    assessment_map = df_assess[['id_assessment', 'code_module', 'code_presentation']].drop_duplicates()
    df_student_assess = df_student_assess.merge(assessment_map, on='id_assessment', how='inner')
    df_student_assess['id_student'] = (
        df_student_assess['id_student'].astype(str) + '_' +
        df_student_assess['code_module'] + '_' +
        df_student_assess['code_presentation']
    )
    df_student_assess.drop(columns=['code_module', 'code_presentation'], inplace=True)

    return df_vle, df_student_assess, df_assess, df_labels


if __name__ == "__main__":
    df_vle, df_student_assess, df_assess, df_labels = load_data('archive')
    print(f"VLE rows: {len(df_vle)}, Students: {df_labels['id_student'].nunique()}")
    print(df_labels['label'].value_counts())