import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
import os

warnings.filterwarnings('ignore')


def load_data(data_dir='archive'):
    print("Loading datasets...")

    df_info = pd.read_csv(os.path.join(data_dir, 'studentInfo.csv'))
    df_vle  = pd.read_csv(os.path.join(data_dir, 'studentVle.csv'))

    # dataset doesn’t have course domains, so assign some dummy ones
    # using a fixed seed so it stays consistent across runs
    np.random.seed(42)
    unique_courses = df_info['code_module'].unique()
    domains = ['STEM', 'Social Sciences', 'Business', 'Arts']
    domain_map = {course: np.random.choice(domains) for course in unique_courses}

    print("Assigned mock domains to courses.")

    return df_info, df_vle, domain_map, unique_courses


def setup_holdout(df_info):
    """
    Simple proxy holdout:
    take students with multiple enrollments,
    use their first course to predict the next one.
    """
    print("Preparing proxy holdout set...")

    presentation_order = {'2013B': 0, '2013J': 1, '2014B': 2, '2014J': 3}
    df_info['pres_order'] = df_info['code_presentation'].map(presentation_order)

    df_info = df_info.sort_values(['id_student', 'pres_order'])

    counts = df_info.groupby('id_student').size()
    multi_students = counts[counts > 1].index

    df_multi = df_info[df_info['id_student'].isin(multi_students)].copy()

    first_courses  = df_multi.groupby('id_student').first().reset_index()
    second_courses = df_multi.groupby('id_student').nth(1).reset_index()

    holdout_set = first_courses.merge(
        second_courses[['id_student', 'code_module']],
        on='id_student',
        suffixes=('', '_target')
    )

    print(f"Holdout size: {len(holdout_set)} students")

    return holdout_set, df_info


def train_content_based(df_info, unique_courses):
    """
    Train one logistic regression per course.
    Each model estimates probability of success for that course.
    """
    print("Training content-based models...")

    df_features = df_info[[
        'id_student', 'code_module', 'final_result',
        'gender', 'region', 'highest_education',
        'imd_band', 'age_band', 'disability'
    ]]

    df_encoded = pd.get_dummies(
        df_features,
        columns=['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    )

    feature_cols = [
        c for c in df_encoded.columns
        if c not in ['id_student', 'code_module', 'final_result']
    ]

    models = {}

    for course in unique_courses:
        course_data = df_encoded[df_encoded['code_module'] == course].copy()

        if len(course_data) == 0:
            continue

        y = course_data['final_result'].isin(['Pass', 'Distinction']).astype(int)
        X = course_data[feature_cols]

        if y.nunique() > 1:
            model = LogisticRegression(max_iter=200, class_weight='balanced')
            model.fit(X, y)
            models[course] = model
        else:
            # fallback if only one class exists
            models[course] = y.mean()

    return models, feature_cols


def predict_content_based(student_features, models, unique_courses):
    scores = {}

    for course in unique_courses:
        model = models.get(course)

        if isinstance(model, LogisticRegression):
            try:
                prob = model.predict_proba(student_features)[0][1]
            except Exception:
                prob = 0.0
        elif model is not None:
            prob = float(model)
        else:
            prob = 0.0

        scores[course] = prob

    return scores


if __name__ == "__main__":
    # load data
    df_info, df_vle, domain_map, unique_courses = load_data('archive')

    # prepare holdout
    holdout_set, df_info = setup_holdout(df_info)

    # train models
    models, feature_cols = train_content_based(df_info, unique_courses)

    # encode features (same as training)
    df_features = df_info[[
        'id_student', 'code_module', 'final_result',
        'gender', 'region', 'highest_education',
        'imd_band', 'age_band', 'disability'
    ]]

    df_encoded = pd.get_dummies(
        df_features,
        columns=['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    )

    print("\n--- Checking one sample student ---")

    # pick one student
    sample_row = holdout_set.iloc[0]
    student_id = sample_row['id_student']
    true_next  = sample_row['code_module_target']

    student_row = df_encoded[df_encoded['id_student'] == student_id]

    if student_row.empty:
        print("Could not find student features.")
    else:
        student_features = student_row[feature_cols]

        scores = predict_content_based(student_features, models, unique_courses)

        # show top 3 recommendations
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

        print(f"\nStudent ID: {student_id}")
        print(f"Actual next course: {true_next}")
        print("Top 3 predicted courses:")

        for course, score in top_3:
            print(f"  {course} -> {score:.3f}")