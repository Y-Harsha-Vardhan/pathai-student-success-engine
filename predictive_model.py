import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, confusion_matrix
import xgboost as xgb
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


def engineer_features(df_info, df_vle, df_assess):
    # build feature matrix: demographics + VLE activity + simple trajectory signals
    print("Engineering features...")

    # --- static features ---
    static_cols = [
        'id_student', 'gender', 'region', 'highest_education', 'imd_band',
        'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability',
        'code_presentation', 'target'
    ]
    df_static = df_info[static_cols].copy()

    # encode categoricals (keeping encoders just in case we need them later)
    cat_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_static[col] = le.fit_transform(df_static[col].astype(str))
        encoders[col] = le

    # --- basic VLE aggregates (days 0–41) ---
    total_interactions = (
        df_vle.groupby('id_student')['sum_click'].sum().reset_index(name='total_interactions')
    )

    unique_sites = (
        df_vle.groupby('id_student')['id_site'].nunique().reset_index(name='unique_sites')
    )

    # longest gap between logins → rough measure of disengagement
    df_vle_sorted = df_vle.sort_values(['id_student', 'date'])
    df_vle_sorted['prev_date'] = df_vle_sorted.groupby('id_student')['date'].shift(1)
    df_vle_sorted['gap'] = df_vle_sorted['date'] - df_vle_sorted['prev_date']

    max_dormancy = (
        df_vle_sorted.groupby('id_student')['gap'].max().fillna(0).reset_index(name='max_dormancy')
    )

    # how many assessments submitted in early window
    early_assess = (
        df_assess.groupby('id_student')['id_assessment'].count().reset_index(name='early_assess_count')
    )

    # --- trajectory features ---
    # trend over weeks matters more than just total activity
    df_vle['week'] = df_vle['date'] // 7
    weekly_vle = df_vle.groupby(['id_student', 'week'])['sum_click'].sum().unstack(fill_value=0)

    # ensure weeks 0–5 exist
    for w in range(6):
        if w not in weekly_vle.columns:
            weekly_vle[w] = 0

    # simple slope approximation (precomputed weights for weeks 0–5)
    weekly_vle['engagement_slope'] = (
        -2.5 * weekly_vle[0] - 1.5 * weekly_vle[1] - 0.5 * weekly_vle[2] +
         0.5 * weekly_vle[3] + 1.5 * weekly_vle[4] + 2.5 * weekly_vle[5]
    ) / 17.5

    # minimum activity in last two weeks
    weekly_vle['recent_minimums'] = weekly_vle[[4, 5]].min(axis=1)
    weekly_vle = weekly_vle.reset_index()

    # compare early vs later activity (first 3 weeks vs next 3 weeks)
    vle_first_half = (
        df_vle[df_vle['date'] <= 20]
        .groupby('id_student')['sum_click'].sum()
        .reset_index(name='fh_clicks')
    )

    vle_second_half = (
        df_vle[df_vle['date'] >= 21]
        .groupby('id_student')['sum_click'].sum()
        .reset_index(name='sh_clicks')
    )

    df_delta = pd.DataFrame({'id_student': df_static['id_student'].unique()})
    df_delta = df_delta.merge(vle_first_half, on='id_student', how='left').fillna(0)
    df_delta = df_delta.merge(vle_second_half, on='id_student', how='left').fillna(0)

    # normalize by days before taking difference
    df_delta['fh_sh_delta'] = (df_delta['fh_clicks'] / 21.0) - (df_delta['sh_clicks'] / 21.0)

    # --- merge everything ---
    df_features = df_static.merge(total_interactions, on='id_student', how='left').fillna({'total_interactions': 0})
    df_features = df_features.merge(unique_sites,     on='id_student', how='left').fillna({'unique_sites': 0})
    df_features = df_features.merge(max_dormancy,     on='id_student', how='left').fillna({'max_dormancy': 0})
    df_features = df_features.merge(early_assess,     on='id_student', how='left').fillna({'early_assess_count': 0})

    df_features = df_features.merge(
        weekly_vle[['id_student', 'engagement_slope', 'recent_minimums']],
        on='id_student', how='left'
    ).fillna(0)

    df_features = df_features.merge(
        df_delta[['id_student', 'fh_sh_delta']],
        on='id_student', how='left'
    ).fillna(0)

    return df_features


def train_and_calibrate(df_features):
    # split by presentation (train on past cohorts, test on the next one)
    # avoids leakage from future students
    print("Splitting data by presentation cohort...")

    train_presentations = ['2013J', '2013B', '2014B']
    test_presentations  = ['2014J']

    train_df = df_features[df_features['code_presentation'].isin(train_presentations)].copy()
    test_df  = df_features[df_features['code_presentation'].isin(test_presentations)].copy()

    drop_cols = ['id_student', 'code_presentation', 'target']
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['target']
    X_test  = test_df.drop(columns=drop_cols)
    y_test  = test_df['target']

    # handle class imbalance (usually more passes than fails)
    print("Training XGBoost classifier...")
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    base_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    base_model.fit(X_train, y_train)

    # calibrate probabilities so scores are more interpretable
    print("Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)

    y_test_probs = calibrated_model.predict_proba(X_test)[:, 1]

    # quick reliability plot
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_probs, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Calibrated XGBoost')
    plt.plot([0, 1], [0, 1], 'k:', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig('calibration_curve.png')
    plt.close()

    print("Saved calibration curve.")

    # choose threshold — prioritise recall (F2 score)
    print("Finding best threshold (F2)...")
    thresholds = np.arange(0.1, 0.9, 0.05)

    best_f2 = 0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_test_probs >= thresh).astype(int)

        if y_pred.sum() == 0:
            continue  # skip if nothing gets flagged

        f2 = fbeta_score(y_test, y_pred, beta=2)

        if f2 > best_f2:
            best_f2 = f2
            best_thresh = thresh

    y_pred_best = (y_test_probs >= best_thresh).astype(int)

    print("\n=== Test Metrics (2014J) ===")
    print(f"Best Threshold : {best_thresh:.3f}")
    print(f"Precision      : {precision_score(y_test, y_pred_best):.3f}")
    print(f"Recall         : {recall_score(y_test, y_pred_best):.3f}")
    print(f"F1 Score       : {f1_score(y_test, y_pred_best):.3f}")
    print(f"ROC-AUC        : {roc_auc_score(y_test, y_test_probs):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best))
    print("============================\n")

    return base_model, X_test, y_test, test_df, y_test_probs, best_thresh


if __name__ == "__main__":
    df_info, df_vle, df_assess = load_and_prepare_data('archive')
    df_features = engineer_features(df_info, df_vle, df_assess)
    base_model, X_test, y_test, test_df, y_probs, f2_thresh = train_and_calibrate(df_features)
    print(f"Best threshold carried forward: {f2_thresh:.3f}")