import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
import yaml
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем параметры
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Загружаем данные
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Features: {list(X_train.columns)}")

# Инициализируем MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic_survival_prediction")

with mlflow.start_run():
    # Логируем параметры
    mlflow.log_params(params['train'])

    # Выбираем модель
    model_type = params['train']['model_type']

    if model_type == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=params['train']['n_estimators'],
            max_depth=params['train']['max_depth'],
            min_samples_split=params['train']['min_samples_split'],
            random_state=params['train']['random_state']
        )
        mlflow.log_param("model_class", "RandomForestClassifier")

    elif model_type == "GradientBoosting":
        model = GradientBoostingClassifier(
            n_estimators=params['train']['n_estimators'],
            max_depth=params['train']['max_depth'],
            learning_rate=params['train']['learning_rate'],
            random_state=params['train']['random_state']
        )
        mlflow.log_param("model_class", "GradientBoostingClassifier")

    else:  # LogisticRegression
        model = LogisticRegression(
            random_state=params['train']['random_state'],
            max_iter=1000
        )
        mlflow.log_param("model_class", "LogisticRegression")

    # Обучаем модель
    print(f"\nTraining {model_type}...")
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if hasattr(model, 'predict_proba') else None

    # Логируем метрики
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    if roc_auc:
        mlflow.log_metric("roc_auc", roc_auc)

    # Логируем confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_type}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Сохраняем confusion matrix
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()

    # Логируем feature importance для деревьев
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
        plt.xlabel('Importance')
        plt.title(f'Top 10 Feature Importances - {model_type}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        plt.close()

        # Логируем важность признаков
        for idx, row in importance_df.iterrows():
            mlflow.log_metric(f"importance_{row['feature']}", row['importance'])

    # Сохраняем модель локально
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    mlflow.log_artifact("model.pkl")

    # Сохраняем метрики в JSON
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc if roc_auc else None
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Выводим результаты
    print(f"\n{'=' * 50}")
    print(f"Model: {model_type}")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    if roc_auc:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"{'=' * 50}")

    # Логируем дополнительную информацию
    mlflow.log_artifact('metrics.json')