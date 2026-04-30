import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')


def load_data(data_dir='archive'):
    print("Loading datasets...")
    df_info = pd.read_csv(os.path.join(data_dir, 'studentInfo.csv'))
    df_vle = pd.read_csv(os.path.join(data_dir, 'studentVle.csv'))

    # combine module + presentation to get a unique course identifier
    # e.g. "AAA_2013B" — there are 22 such combinations in this dataset
    df_info['course_id'] = df_info['code_module'] + '_' + df_info['code_presentation']

    # the dataset doesn't have explicit subject domains, so we mock them
    # using a fixed seed so the mapping is reproducible across runs
    np.random.seed(42)
    unique_courses = df_info['course_id'].unique()
    domains = ['STEM', 'Social Sciences', 'Business', 'Arts']
    domain_map = {course: np.random.choice(domains) for course in unique_courses}

    print("Mocking domains for the 22 unique courses (combinations)...")

    return df_info, df_vle, domain_map, unique_courses


def setup_holdout(df_info):
    """
    Proxy holdout: students who took more than one presentation.
    We use their first course as input and try to predict which course they took next.
    """
    print("Identifying students for Proxy Holdout Evaluation...")

    presentation_order = {'2013B': 0, '2013J': 1, '2014B': 2, '2014J': 3}
    df_info['pres_order'] = df_info['code_presentation'].map(presentation_order)
    df_info = df_info.sort_values(['id_student', 'pres_order'])

    counts = df_info.groupby('id_student').size()
    multi_students = counts[counts > 1].index
    df_multi = df_info[df_info['id_student'].isin(multi_students)].copy()

    first_courses = df_multi.groupby('id_student').first().reset_index()
    second_courses = df_multi.groupby('id_student').nth(1).reset_index()

    holdout_set = first_courses.merge(
        second_courses[['id_student', 'course_id']],
        on='id_student',
        suffixes=('', '_target')
    )

    print(f"Total sample size (N) of proxy holdout set: {len(holdout_set)}")
    return holdout_set, df_info


def train_content_based(df_info, unique_courses):
    """
    Option A: one logistic regression per course.
    Each model predicts pass/fail probability from demographic features alone.
    That probability becomes the 'affinity score' for recommending that course.
    """
    print("Training 22 Content-Based Logistic Regression models...")

    df_features = df_info[[
        'id_student', 'course_id', 'final_result',
        'gender', 'region', 'highest_education',
        'imd_band', 'age_band', 'disability'
    ]]

    df_encoded = pd.get_dummies(
        df_features,
        columns=['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    )
    feature_cols = [c for c in df_encoded.columns if c not in ['id_student', 'course_id', 'final_result']]

    models = {}
    for course in unique_courses:
        course_data = df_encoded[df_encoded['course_id'] == course].copy()
        if len(course_data) == 0:
            continue

        y = course_data['final_result'].isin(['Pass', 'Distinction']).astype(int)
        X = course_data[feature_cols]

        if y.nunique() > 1:
            lr = LogisticRegression(max_iter=200, class_weight='balanced')
            lr.fit(X, y)
            models[course] = lr
        else:
            # if only one class exists, just return the mean as a constant fallback
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


def extract_behavioral_features(df_vle, holdout_student_ids):
    # simple summary stats from the clickstream — nothing fancy yet
    # could add time-decay weighting later but this is fine for now
    features = df_vle.groupby('id_student').agg(
        total_clicks=('sum_click', 'sum'),
        unique_sites=('id_site', 'nunique'),
        active_days=('date', 'nunique')
    ).reset_index()
    return features


def train_collaborative(df_behavioral):
    """
    Option B: KNN on behavioral features (clickstream).
    Cosine similarity works better here than euclidean since
    we care about engagement *patterns*, not raw volume.
    """
    print("Training Collaborative Filtering KNN (k=15)...")

    X_behav = df_behavioral.drop('id_student', axis=1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_behav)

    knn = NearestNeighbors(n_neighbors=15, metric='cosine')
    knn.fit(X_scaled)

    return knn, df_behavioral['id_student'].values, X_scaled


def predict_collaborative(student_idx, X_scaled_student, knn, train_student_ids, df_info):
    scores = {}

    if X_scaled_student is None or np.isnan(X_scaled_student).all():
        return scores

    distances, indices = knn.kneighbors(X_scaled_student)
    neighbor_ids = train_student_ids[indices[0]]

    neighbor_data = df_info[df_info['id_student'].isin(neighbor_ids)]

    # score = how often neighbors took the course × how well they did in it
    course_stats = neighbor_data.groupby('course_id').agg(
        freq=('id_student', 'count'),
        pass_rate=('final_result', lambda x: x.isin(['Pass', 'Distinction']).mean())
    )

    for course, row in course_stats.iterrows():
        scores[course] = row['freq'] * row['pass_rate']

    return scores


def apply_diversity_filter(sorted_courses, domain_map):
    # cap any single domain at 2 slots in the top-3
    # avoids recommending three STEM courses to someone who's only studied STEM
    top_3 = []
    domain_counts = {}

    for course in sorted_courses:
        domain = domain_map[course]
        if domain_counts.get(domain, 0) < 2:
            top_3.append(course)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if len(top_3) == 3:
            break

    return top_3


def generate_recommendations(holdout_set, df_info, models, feature_cols,
                              knn, train_student_ids, X_scaled,
                              unique_courses, domain_map):
    print("Generating recommendations for holdout set...")
    results = []

    # precompute global popularity × pass-rate for cold-start fallback
    # (not actually used below since we fall back to CB, but useful to have)
    global_pop = df_info.groupby('course_id').size()
    global_pass = df_info.groupby('course_id')['final_result'].apply(
        lambda x: x.isin(['Pass', 'Distinction']).mean()
    )
    global_scores = (global_pop * global_pass).to_dict()

    for _, student in holdout_set.iterrows():
        student_id = student['id_student']

        # --- content-based scores ---
        student_feat = pd.DataFrame([student])
        student_encoded = pd.get_dummies(
            student_feat,
            columns=['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
        )

        # align to training feature space — fill missing dummies with 0
        X_cb = pd.DataFrame(0, index=[0], columns=feature_cols)
        for col in student_encoded.columns:
            if col in feature_cols:
                X_cb[col] = student_encoded[col].values

        cb_scores = predict_content_based(X_cb, models, unique_courses)

        # --- collaborative scores ---
        idx_match = np.where(train_student_ids == student_id)[0]
        if len(idx_match) > 0:
            X_cf_student = X_scaled[idx_match[0]].reshape(1, -1)
            cf_scores = predict_collaborative(
                student_id, X_cf_student, knn, train_student_ids, df_info
            )
        else:
            # cold start: no clickstream data for this student
            cf_scores = {}

        # --- hybrid blend ---
        # normalize both score sets to [0,1] before blending
        cb_max = max(cb_scores.values()) if cb_scores and max(cb_scores.values()) > 0 else 1
        cf_max = max(cf_scores.values()) if cf_scores and max(cf_scores.values()) > 0 else 1

        hybrid_scores = {}
        for course in unique_courses:
            cb_val = cb_scores.get(course, 0) / cb_max
            cf_val = cf_scores.get(course, 0) / cf_max

            if len(cf_scores) == 0:
                # cold start: demographic signal only
                final_score = cb_val
            else:
                final_score = 0.5 * cb_val + 0.5 * cf_val

            hybrid_scores[course] = final_score

        # rank and apply diversity constraint
        sorted_hybrid = sorted(hybrid_scores, key=lambda x: hybrid_scores[x], reverse=True)
        sorted_cb = sorted(cb_scores, key=lambda x: cb_scores[x], reverse=True)
        sorted_cf = sorted(cf_scores, key=lambda x: cf_scores[x], reverse=True)

        top3_hybrid = apply_diversity_filter(sorted_hybrid, domain_map)
        top3_cb = apply_diversity_filter(sorted_cb, domain_map)
        top3_cf = apply_diversity_filter(sorted_cf, domain_map) if cf_scores else top3_cb

        results.append({
            'id_student': student_id,
            'target_course': student['course_id_target'],
            'top3_hybrid': top3_hybrid,
            'top3_cb': top3_cb,
            'top3_cf': top3_cf
        })

    return pd.DataFrame(results)


def evaluate(results, n_courses=22):
    def precision_at_3(row, col_name):
        return 1 if row['target_course'] in row[col_name] else 0

    p3_hybrid = results.apply(lambda r: precision_at_3(r, 'top3_hybrid'), axis=1).mean()
    p3_cb = results.apply(lambda r: precision_at_3(r, 'top3_cb'), axis=1).mean()
    p3_cf = results.apply(lambda r: precision_at_3(r, 'top3_cf'), axis=1).mean()

    # random baseline: picking 3 courses at random from 22
    random_baseline = 3 / n_courses

    print("\n" + "=" * 40)
    print("         PHASE 5: EVALUATION REPORT")
    print("=" * 40)
    print(f"Random Chance Baseline (3/{n_courses}): {random_baseline:.2%}")
    print("-" * 40)
    print(f"Option A: Content-Based Precision@3 : {p3_cb:.2%}")
    print(f"Option B: Collaborative Precision@3 : {p3_cf:.2%}")
    print(f"Hybrid Ensemble Precision@3         : {p3_hybrid:.2%}")
    print("-" * 40)
    print(f"Engine Uplift vs Random Baseline    : +{(p3_hybrid - random_baseline) * 100:.1f} percentage points")
    print("=" * 40)


def main():
    df_info, df_vle, domain_map, unique_courses = load_data('archive')
    holdout_set, df_info_sorted = setup_holdout(df_info)

    models, feature_cols = train_content_based(df_info_sorted, unique_courses)

    df_behav = extract_behavioral_features(df_vle, holdout_set['id_student'].values)
    knn, train_student_ids, X_scaled = train_collaborative(df_behav)

    results = generate_recommendations(
        holdout_set, df_info_sorted, models, feature_cols,
        knn, train_student_ids, X_scaled, unique_courses, domain_map
    )

    evaluate(results, n_courses=len(unique_courses))


if __name__ == "__main__":
    main()