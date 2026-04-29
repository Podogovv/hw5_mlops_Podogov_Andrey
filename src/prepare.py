import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
import os
import pickle


def load_and_clean_data(df):

    # Создаем копию
    df_clean = df.copy()

    # Удаляем ненужные колонки
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

    # Обработка возраста
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    if params['prepare']['fillna_strategy'] == 'median':
        age_median = df_clean['Age'].median()
        df_clean['Age'].fillna(age_median, inplace=True)
    elif params['prepare']['fillna_strategy'] == 'mean':
        age_mean = df_clean['Age'].mean()
        df_clean['Age'].fillna(age_mean, inplace=True)
    elif params['prepare']['fillna_strategy'] == 'mode':
        age_mode = df_clean['Age'].mode()[0]
        df_clean['Age'].fillna(age_mode, inplace=True)

    # Обработка порта посадки
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)

    # Обработка количества родственников
    df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
    df_clean['IsAlone'] = (df_clean['FamilySize'] == 1).astype(int)

    # Кодируем категориальные признаки
    le_sex = LabelEncoder()
    df_clean['Sex'] = le_sex.fit_transform(df_clean['Sex'])  # male=1, female=0
    le_embarked = LabelEncoder()
    df_clean['Embarked'] = le_embarked.fit_transform(df_clean['Embarked'])

    # Стандартизируем числовые признаки
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    scaler = StandardScaler()
    df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

    return df_clean


# Загружаем параметры
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Загружаем данные Titanic
print("Loading Titanic dataset...")
data = pd.read_csv('data/raw/titanic.csv')

print(f"Original data shape: {data.shape}")
print(f"Missing values:\n{data.isnull().sum()}")

# Очищаем данные
data_clean = load_and_clean_data(data)

print(f"Cleaned data shape: {data_clean.shape}")

# Разделяем на признаки и целевую переменную
X = data_clean.drop('Survived', axis=1)
y = data_clean['Survived']

# Сплит данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params['prepare']['test_size'],
    random_state=params['prepare']['random_state'],
    stratify=y
)

# Сохраняем данные
os.makedirs('data/processed', exist_ok=True)
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

# Сохраняем метаинформацию
metadata = {
    'original_shape': data.shape,
    'cleaned_shape': data_clean.shape,
    'train_shape': X_train.shape,
    'test_shape': X_test.shape,
    'survival_rate': y.mean(),
    'train_survival_rate': y_train.mean(),
    'test_survival_rate': y_test.mean()
}

with open('data/processed/metadata.json', 'w') as f:
    import json

    json.dump(metadata, f, indent=2)

print("\nData preparation completed!")
print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size: {X_test.shape[0]} samples")
print(f"Survival rate in train: {y_train.mean():.2%}")
print(f"Survival rate in test: {y_test.mean():.2%}")