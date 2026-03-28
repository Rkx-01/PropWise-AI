import pandas as pd
import os
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# 1. Data Loading & Feature Engineering
# ---------------------------------------------------------------------------

def load_and_engineer_data(file_path):
    """
    Loads data and performs feature engineering.
    Adds: year_built (synthetic), property_age, total_rooms.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    df = pd.read_csv(file_path)

    # Reproducible synthetic year_built
    np.random.seed(42)
    df['year_built'] = np.random.randint(1950, 2024, size=len(df))

    # Feature Engineering
    current_year = datetime.now().year
    df['property_age'] = current_year - df['year_built']
    df['total_rooms']  = df['bedrooms'] + df['bathrooms']

    # Note: price_per_sqft excluded to avoid data leakage
    return df


# ---------------------------------------------------------------------------
# 2. Preprocessing helper
# ---------------------------------------------------------------------------

def build_preprocessor(X):
    """Returns a ColumnTransformer that OHE-encodes categoricals."""
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical Features : {categorical_cols}")
    print(f"Numerical  Features  : {X.select_dtypes(include='number').columns.tolist()}")
    return ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
        remainder='passthrough'
    ), categorical_cols


# ---------------------------------------------------------------------------
# 3. Evaluate a trained pipeline
# ---------------------------------------------------------------------------

def evaluate_pipeline(pipeline, X_test, y_test):
    """Returns a metrics dict for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_test, y_pred)
    return {
        "MAE" : round(float(mae),  2),
        "MSE" : round(float(mse),  2),
        "RMSE": round(rmse,        2),
        "R2"  : round(float(r2),   4),
    }


# ---------------------------------------------------------------------------
# 4. Main training routine
# ---------------------------------------------------------------------------

def train_pipeline(df):
    """
    1. Trains RF, GBR, and XGB with default params.
    2. Picks the best by R² on the hold-out set.
    3. Fine-tunes the winner with GridSearchCV (5-fold CV).
    4. Saves best pipeline → model.pkl
    5. Saves all-model metrics → metrics.json  (list, ordered best→worst R²)
    """
    print("=" * 55)
    print("  PropWise-AI  |  Multi-Model Training Run")
    print("=" * 55)

    # ── Prepare features / target ──────────────────────────────
    X = df.drop(columns=['price'])
    if 'price_per_sqft' in X.columns:
        X = X.drop(columns=['price_per_sqft'])
    y = df['price']

    preprocessor, _ = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Candidate models ──────────────────────────────────────
    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            random_state=42,
            verbosity=0,          # suppress XGBoost chatter
            eval_metric='rmse',
        ),
    }

    all_metrics   = []   # will hold one dict per model
    best_r2       = -np.inf
    best_name     = None
    best_pipeline = None

    # ── Round 1: baseline evaluation ──────────────────────────
    print("\n[Round 1] Baseline evaluation on hold-out set")
    print("-" * 55)

    for name, model in candidates.items():
        print(f"  Fitting {name} …", end=" ", flush=True)
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        m = evaluate_pipeline(pipe, X_test, y_test)
        m["model"] = name
        all_metrics.append({"model": name, **{k: v for k, v in m.items() if k != "model"}})
        print(f"R²={m['R2']:.4f}  RMSE={m['RMSE']:,.0f}")

        if m['R2'] > best_r2:
            best_r2       = m['R2']
            best_name     = name
            best_pipeline = pipe

    print(f"\n  → Best baseline model: {best_name}  (R²={best_r2:.4f})")

    # ── Round 2: GridSearchCV on best model ───────────────────
    print(f"\n[Round 2] GridSearchCV (5-fold CV) on {best_name}")
    print("-" * 55)

    # Param grids keyed by model name
    param_grids = {
        "RandomForest": {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth'   : [None, 10, 20],
            'model__min_samples_split': [2, 5],
        },
        "GradientBoosting": {
            'model__n_estimators' : [100, 200, 300],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__max_depth'    : [3, 5, 7],
        },
        "XGBoost": {
            'model__n_estimators' : [100, 200, 300],
            'model__learning_rate': [0.05, 0.1, 0.2],
            'model__max_depth'    : [3, 5, 7],
            'model__subsample'    : [0.8, 1.0],
        },
    }

    # Re-build a fresh pipeline for the grid search
    fresh_model = {
        "RandomForest"    : RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "XGBoost"         : XGBRegressor(random_state=42, verbosity=0, eval_metric='rmse'),
    }[best_name]

    tuning_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', fresh_model)
    ])

    grid = GridSearchCV(
        tuning_pipe,
        param_grids[best_name],
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    tuned_metrics = evaluate_pipeline(grid.best_estimator_, X_test, y_test)
    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Tuned hold-out → R²={tuned_metrics['R2']:.4f}  RMSE={tuned_metrics['RMSE']:,.0f}")

    # Update the best model entry in all_metrics with tuned numbers
    for entry in all_metrics:
        if entry['model'] == best_name:
            entry.update({
                "MAE" : tuned_metrics["MAE"],
                "MSE" : tuned_metrics["MSE"],
                "RMSE": tuned_metrics["RMSE"],
                "R2"  : tuned_metrics["R2"],
                "tuned": True,
                "best_params": grid.best_params_,
            })
            break

    best_pipeline = grid.best_estimator_

    # ── Sort metrics best → worst R² ──────────────────────────
    all_metrics.sort(key=lambda x: x['R2'], reverse=True)

    # ── Save artifacts ─────────────────────────────────────────
    print("\nSaving model.pkl …")
    joblib.dump(best_pipeline, 'model.pkl')

    print("Saving metrics.json …")
    with open('metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)

    print("\n" + "=" * 55)
    print("  Training complete!")
    print(f"  Best model : {best_name}  (tuned R²={tuned_metrics['R2']:.4f})")
    print("=" * 55 + "\n")

    return best_pipeline, X.columns.tolist()


# ---------------------------------------------------------------------------
# 5. Feature Importance (unchanged API)
# ---------------------------------------------------------------------------

def plot_importance(pipeline):
    """
    Extracts, prints top 10, and saves feature importance to CSV and plot.
    Works for RF, GBR, and XGB (all expose .feature_importances_).
    """
    print("--- Extracting and Saving Feature Importance ---")
    model        = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    # Get all feature names after transformation
    all_features = preprocessor.get_feature_names_out()
    importances  = model.feature_importances_

    # Create DataFrame and sort
    df_importance = pd.DataFrame({
        'Feature'   : all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Print top 10
    print("\nTop 10 Most Important Features:")
    print(df_importance.head(10).to_string(index=False))

    # Save to CSV
    df_importance.to_csv('feature_importance.csv', index=False)
    print("\nFeature importance saved to: feature_importance.csv")

    # Plot top 10
    num_show    = min(10, len(importances))
    top_features = df_importance.head(num_show)

    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Feature Importances (Best Tuned Model)")
    plt.barh(
        range(num_show),
        top_features['Importance'][::-1],
        color='skyblue',
        align='center'
    )
    plt.yticks(range(num_show), top_features['Feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Feature importance plot updated.\n")


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = os.path.join("data", "Housing.csv")
    df = load_and_engineer_data(data_path)

    if df is not None:
        pipeline, feature_names = train_pipeline(df)
        plot_importance(pipeline)
