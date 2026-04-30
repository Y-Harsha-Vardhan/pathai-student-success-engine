import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
# sklearn MinMaxScaler removed: using custom expanding-window implementation for preventing leakage


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

def feature_engineering(df_vle, df_student_assess, df_assess, df_labels):
    # compute behavioral features per (student, week)
    print("Engineering features...")

    # procrastination = how late/early a submission was relative to deadline
    assess_merged = pd.merge(df_student_assess, df_assess, on='id_assessment', how='inner')

    # quick sanity check: composite key should exist after merge
    assert assess_merged['id_student'].str.contains('_').all(), "Composite ID issue after merge"

    assess_merged['procrastination'] = assess_merged['date_submitted'] - assess_merged['date']

    students = df_labels['id_student'].unique()

    # create full (student x week) grid so we don't lose inactive weeks
    min_week = int(min(df_vle['week'].min(), assess_merged['week'].min()))
    max_week = int(max(df_vle['week'].max(), assess_merged['week'].max()))
    weeks = range(min_week, max_week + 1)

    grid = pd.MultiIndex.from_product(
        [students, weeks], names=['id_student', 'week']
    ).to_frame(index=False)

    # - Feature 1: total clicks per week -
    vol = df_vle.groupby(['id_student', 'week'])['sum_click'].sum().reset_index(name='interaction_volume')
    grid = grid.merge(vol, on=['id_student', 'week'], how='left').fillna({'interaction_volume': 0})

    # -- Feature 2: number of active days in the week --
    reg = df_vle.groupby(['id_student', 'week'])['date'].nunique().reset_index(name='session_regularity')
    grid = grid.merge(reg, on=['id_student', 'week'], how='left').fillna({'session_regularity': 0})

    # --- Feature 3: how many different resources were accessed ---
    div = df_vle.groupby(['id_student', 'week'])['id_site'].nunique().reset_index(name='activity_diversity')
    grid = grid.merge(div, on=['id_student', 'week'], how='left').fillna({'activity_diversity': 0})

    # ---- Feature 4: days since last login ----
    # using merge_asof to avoid looping over students (much faster)
    grid['week_start'] = grid['week'] * 7
    df_vle_sorted = df_vle.dropna(subset=['date']).sort_values('date')
    grid_sorted = grid.sort_values('week_start').reset_index()

    last_login = pd.merge_asof(
        grid_sorted[['index', 'id_student', 'week_start']],
        df_vle_sorted[['id_student', 'date']],
        by='id_student',
        left_on='week_start',
        right_on='date',
        direction='backward'
    )

    last_login = last_login.set_index('index')
    grid['max_date'] = last_login['date']

    # if no previous login exists, treat as no gap
    grid['max_date'] = grid['max_date'].fillna(grid['week_start'])
    grid['days_since_last_login'] = grid['week_start'] - grid['max_date']
    grid.drop(columns=['week_start', 'max_date'], inplace=True)

    # ----- Feature 5: procrastination index -----
    # carry forward last known value for weeks without submissions
    proc = assess_merged.groupby(['id_student', 'week'])['procrastination'].mean().reset_index(
        name='procrastination_index'
    )
    grid = grid.merge(proc, on=['id_student', 'week'], how='left')
    grid['procrastination_index'] = (
        grid.groupby('id_student')['procrastination_index'].ffill().fillna(0)
    )

    # restrict to standard semester window
    grid = grid[(grid['week'] >= 0) & (grid['week'] <= 39)].copy()

    return grid


def apply_leakage_free_scaling(grid, feature_cols):
    # scale features to [0,1] using only past + current data (no future leakage)
    print("Applying leakage-free scaling...")

    scaled_grid = grid.copy()
    scaled_cols = [f'{c}_scaled' for c in feature_cols]

    for col in scaled_cols:
        scaled_grid[col] = 0.0

    weeks = sorted(scaled_grid['week'].unique())

    # keep track of running min/max
    running_min = {}
    running_max = {}

    for t in weeks:
        week_mask = scaled_grid['week'] == t
        week_data = scaled_grid.loc[week_mask, feature_cols]

        # update stats with current week
        for col in feature_cols:
            c_min = week_data[col].min()
            c_max = week_data[col].max()

            if not pd.isna(c_min):
                running_min[col] = min(running_min.get(col, np.inf), c_min)
            if not pd.isna(c_max):
                running_max[col] = max(running_max.get(col, -np.inf), c_max)

        # scale using stats seen so far
        for i, col in enumerate(feature_cols):
            col_min = running_min.get(col, 0)
            col_max = running_max.get(col, 1)
            denom = col_max - col_min

            if denom == 0 or np.isinf(denom) or pd.isna(denom):
                scaled_grid.loc[week_mask, scaled_cols[i]] = 0.0
            else:
                scaled_grid.loc[week_mask, scaled_cols[i]] = (week_data[col] - col_min) / denom

    # flip features where higher = worse engagement
    scaled_grid['days_since_last_login_scaled'] = 1.0 - scaled_grid['days_since_last_login_scaled']
    scaled_grid['procrastination_index_scaled'] = 1.0 - scaled_grid['procrastination_index_scaled']

    return scaled_grid

if __name__ == "__main__":
    df_vle, df_student_assess, df_assess, df_labels = load_data('archive')

    grid = feature_engineering(df_vle, df_student_assess, df_assess, df_labels)
    print("Feature grid shape:", grid.shape)
    print(grid.head())

    feature_cols = [
        'interaction_volume', 'session_regularity', 'activity_diversity',
        'days_since_last_login', 'procrastination_index'
    ]
    scaled = apply_leakage_free_scaling(grid, feature_cols)
    print("\nScaled sample:")
    print(scaled[[c + '_scaled' for c in feature_cols]].describe())