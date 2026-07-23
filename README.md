# AutoML-Hyerparameter-Optimization
This project explores AutoML (Automated Machine Learning) and Hyperparameter Optimization (HPO) by building and tuning a regression model to predict house prices in Bengaluru, India. It compares multiple hyperparameter search strategies — Grid Search, Random Search, and Bayesian Optimization (Optuna) — against a baseline model, evaluating each on accuracy and computation time.

Objectives:

1. Build a baseline regression model using default hyperparameters.
2. Implement and compare Grid Search, Random Search, and Bayesian Optimization (Optuna).
3. Evaluate trade-offs between model performance and computation time.
4. Identify the most influential hyperparameters using Optuna's importance analysis.
5. Visualize results across all methods.
   

Dataset source: Kaggle - Bengaluru House Price Data

Tech Stack:

1. Python 3.x - Core Language
2. Visual Studio Code - Development environment
3. pandas, NumPyData - loading & preprocessing
4. scikit-learn - Baseline model, Grid Search, Random Search
5. Optuna - Bayesian Optimization
6. matplotlib, seaborn - Visualization
7. Streamlit - Working prototype / web application interface


Future Scope:

1. Extend to Neural Architecture Search (NAS) for deep learning models.
2. Integrate with an MLOps pipeline for automated retraining.
3. Use distributed tuning (Ray Tune) for larger search spaces.
4. Benchmark against a full AutoML framework (e.g., AutoGluon).

Akshitha Penakacherla,
Internship Project - Navodita Infotech,
akshitha.penakacherlla06@gmail.com
