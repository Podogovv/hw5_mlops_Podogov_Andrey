import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn
import yaml
import pickle
import os

# Загружаем параметры
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Загружаем данные
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Инициализируем MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_classification")

with mlflow.start_run():
    # Логируем параметры
    mlflow.log_params(params['train'])
    
    # Выбираем модель
    if params['train']['model_type'] == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=params['train']['n_estimators'],
            max_depth=params['train']['max_depth'],
            random_state=params['train']['random_state']
        )
    else:
        model = LogisticRegression(
            random_state=params['train']['random_state']
        )
    
    # Обучаем модель
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Логируем метрики
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Логируем модель
    mlflow.sklearn.log_model(model, "model")
    
    # Сохраняем модель локально
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    mlflow.log_artifact("model.pkl")
    
    print(f"Model: {params['train']['model_type']}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")