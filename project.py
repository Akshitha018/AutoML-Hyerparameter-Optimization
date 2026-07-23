import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
 
from xgboost import XGBRegressor
import optuna

df = pd.read_csv("bengaluru_house_prices.csv")
df = df.drop(['society'], axis=1)

df['location'] = df['location'].fillna('Sarjapur Road')
df['bath'] = df['bath'].fillna(df['bath'].median())
df['balcony'] = df['balcony'].fillna(df['balcony'].median())
df = df.dropna(subset=['size', 'total_sqft'])

df['bhk'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]))
df = df.drop('size', axis=1)

def convert_sqft(x):
    tokens = str(x).split('-')
    if len(tokens) == 2:
        try:
            return (float(tokens[0]) + float(tokens[1])) / 2
        except ValueError:
            return None
    try:
        return float(x)
    except ValueError:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna(subset=['total_sqft'])

df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']

location_counts = df['location'].value_counts()

# Increase threshold from 10 to something much higher, e.g. 50
rare_locations = location_counts[location_counts <= 50]
df['location'] = df['location'].apply(lambda x: 'other' if x in rare_locations else x)

print("Unique locations remaining:", df['location'].nunique())

df = df[~(df['total_sqft'] / df['bhk'] < 300)]

def remove_pps_outliers(data):
    out = pd.DataFrame()
    for key, subdf in data.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        out = pd.concat([out, reduced], ignore_index=True)
    return out

df = remove_pps_outliers(df)
df = df[df.bath < df.bhk + 2]

df = df.drop('price_per_sqft', axis=1)

df = pd.get_dummies(df, columns=['location', 'area_type', 'availability'], drop_first=True)

print("Final shape after preprocessing:", df.shape)

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:", X_train.shape, " X_test:", X_test.shape)

# STEP 2: AUTOMATED FEATURE SELECTION
print("\n===== STEP 2: Automated Feature Selection =====")
 
# Use a quick Random Forest to rank feature importance
fs_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
fs_model.fit(X_train, y_train)
 
# SelectFromModel automatically keeps only features above median importance
selector = SelectFromModel(fs_model, threshold='median', prefit=True)
 
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
 
selected_columns = X_train.columns[selector.get_support()]
 
print(f"Original feature count : {X_train.shape[1]}")
print(f"Selected feature count : {X_train_selected.shape[1]}")
print(f"Top selected features  : {list(selected_columns[:10])} ...")
 
# Convert back to DataFrame (keeps column names for readability downstream)
X_train_selected = pd.DataFrame(X_train_selected, columns=selected_columns, index=X_train.index)
X_test_selected = pd.DataFrame(X_test_selected, columns=selected_columns, index=X_test.index)

 # STEP 3: AUTOMATED MODEL SELECTION
print("\n===== STEP 3: Automated Model Selection =====")
 
candidate_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0)
}
 
model_scores = {}
for name, model in candidate_models.items():
    start = time.time()
    score = cross_val_score(model, X_train_selected, y_train, cv=3, scoring='r2').mean()
    elapsed = time.time() - start
    model_scores[name] = {"R2": score, "Time": elapsed}
    print(f"{name:<20} R2: {score:.4f}   Time: {elapsed:.2f}s")
 
model_comparison_df = pd.DataFrame(model_scores).T.sort_values("R2", ascending=False)
print("\nModel comparison (ranked by R2):")
print(model_comparison_df)
 
best_model_name = model_comparison_df.index[0]
print(f"\nBest model selected automatically: {best_model_name}")

 # STEP 4: HYPERPARAMETER OPTIMIZATION ON THE SELECTED BEST MODEL
print(f"\n===== STEP 4: Hyperparameter Optimization on {best_model_name} =====")
 
results = {}
 
#BASELINE MODEL
baseline_model = candidate_models[best_model_name]
start_time = time.time()
baseline_model.fit(X_train_selected, y_train)
baseline_time = time.time() - start_time
 
y_pred = baseline_model.predict(X_test_selected)
results["Baseline"] = {
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE": mean_absolute_error(y_test, y_pred),
    "Time": baseline_time,
    "Best_Params": "Default"
}
 
# --- Define search spaces per model type ---
if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
    grid_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None] if best_model_name != "XGBoost" else [4, 6, 8]
    }
    random_param_dist = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [5, 10, 15, 20] if best_model_name != "XGBoost" else [3, 5, 7, 9]
    }
elif best_model_name == "Decision Tree":
    grid_param_grid = {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10]}
    random_param_dist = {'max_depth': [3, 5, 8, 10, 15, None], 'min_samples_split': [2, 4, 6, 8, 10]}
else:  # Linear / Ridge — limited tunable params
    grid_param_grid = {'alpha': [0.1, 1.0, 10.0]} if best_model_name == "Ridge Regression" else {}
    random_param_dist = grid_param_grid

#GRID SEARCH
if grid_param_grid:
    grid_search = GridSearchCV(
        estimator=candidate_models[best_model_name].__class__(),
        param_grid=grid_param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
    )
    start_time = time.time()
    grid_search.fit(X_train_selected, y_train)
    grid_time = time.time() - start_time
 
    grid_pred = grid_search.best_estimator_.predict(X_test_selected)
    results["Grid Search"] = {
        "R2": r2_score(y_test, grid_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, grid_pred)),
        "MAE": mean_absolute_error(y_test, grid_pred),
        "Time": grid_time,
        "Best_Params": grid_search.best_params_
    }
    print(f"Grid Search done: {grid_search.best_params_}")
 
#RANDOM SEARCH
if random_param_dist:
    random_search = RandomizedSearchCV(
        estimator=candidate_models[best_model_name].__class__(),
        param_distributions=random_param_dist, n_iter=10, cv=3,
        scoring='r2', n_jobs=-1, random_state=42, verbose=1
    )
    start_time = time.time()
    random_search.fit(X_train_selected, y_train)
    random_time = time.time() - start_time
 
    random_pred = random_search.best_estimator_.predict(X_test_selected)
    results["Random Search"] = {
        "R2": r2_score(y_test, random_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, random_pred)),
        "MAE": mean_absolute_error(y_test, random_pred),
        "Time": random_time,
        "Best_Params": random_search.best_params_
    }
    print(f"Random Search done: {random_search.best_params_}")
 

#BAYESIAN OPTIMIZATION (OPTUNA)
def objective(trial):
    if best_model_name == "Random Forest":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
        }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    elif best_model_name == "XGBoost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        }
        model = XGBRegressor(**params, random_state=42, verbosity=0)
    elif best_model_name == "Gradient Boosting":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        }
        model = GradientBoostingRegressor(**params, random_state=42)
    elif best_model_name == "Decision Tree":
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        }
        model = DecisionTreeRegressor(**params, random_state=42)
    else:  # Linear / Ridge
        alpha = trial.suggest_float('alpha', 0.01, 10.0)
        model = Ridge(alpha=alpha)
 
    return cross_val_score(model, X_train_selected, y_train, cv=3, scoring='r2').mean()
 
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
start_time = time.time()
study.optimize(objective, n_trials=15, show_progress_bar=True)
optuna_time = time.time() - start_time
 
# Rebuild best model from Optuna's winning params
best_params = study.best_params
if best_model_name == "Random Forest":
    optuna_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
elif best_model_name == "XGBoost":
    optuna_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
elif best_model_name == "Gradient Boosting":
    optuna_model = GradientBoostingRegressor(**best_params, random_state=42)
elif best_model_name == "Decision Tree":
    optuna_model = DecisionTreeRegressor(**best_params, random_state=42)
else:
    optuna_model = Ridge(**best_params)
 
optuna_model.fit(X_train_selected, y_train)
optuna_pred = optuna_model.predict(X_test_selected)
 
results["Bayesian Optimization (Optuna)"] = {
    "R2": r2_score(y_test, optuna_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, optuna_pred)),
    "MAE": mean_absolute_error(y_test, optuna_pred),
    "Time": optuna_time,
    "Best_Params": best_params
}
print(f"Optuna done: {best_params}")
 

#FINAL COMPARISON TABLE
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[["R2", "RMSE", "MAE", "Time", "Best_Params"]]

print("\n===== FINAL COMPARISON: Baseline vs Grid vs Random vs Optuna =====")
print(comparison_df[["R2", "RMSE", "MAE", "Time"]])

comparison_df.to_csv("hpo_comparison_results.csv")
print("\nSaved to hpo_comparison_results.csv")

#SAVE THE FINAL BEST MODEL FOR THE APP
with open("best_model.pkl", "wb") as f:
    pickle.dump(optuna_model, f)
 
with open("model_columns.pkl", "wb") as f:
    pickle.dump(list(selected_columns), f)
 
print("\nSaved best_model.pkl and model_columns.pkl (feature-selected columns).")
print(f"\nSUMMARY: Selected model = {best_model_name}, "
      f"Features used = {len(selected_columns)}/{X_train.shape[1]}, "
      f"Final R2 (Optuna) = {results['Bayesian Optimization (Optuna)']['R2']:.4f}")

#VISUALIZATIONS
# R2 Score comparison bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=comparison_df.index, y=comparison_df["R2"].astype(float))
plt.title("R2 Score Comparison Across Methods")
plt.ylabel("R2 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("r2_comparison.png")
plt.show()

# Time taken comparison bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=comparison_df.index, y=comparison_df["Time"].astype(float))
plt.title("Time Taken Comparison Across Methods")
plt.ylabel("Time (seconds)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("time_comparison.png")
plt.show()

#Optuna optimization history (score vs trail number)
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()
plt.savefig("optuna_optimization_history.png")
plt.show()

# Optuna hyperparameter importance
optuna.visualization.matplotlib.plot_param_importances(study)
plt.tight_layout()
plt.savefig("optuna_param_importance.png")
plt.show()

