import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os
import pickle

# Загружаем параметры
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Загружаем данные
data = pd.read_csv('data/raw/iris.csv')

# Базовая очистка (удаляем дубликаты)
data = data.drop_duplicates()

# Разделяем на признаки и целевую переменную
# Для Iris dataset
X = data.drop('species', axis=1)
y = data['species']

# Сплит данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params['prepare']['test_size'],
    random_state=params['prepare']['random_state'],
    stratify=y
)

# Сохраняем данные
os.makedirs('data/processed', exist_ok=True)
pd.DataFrame(X_train).to_csv('data/processed/X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('data/processed/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)

print("Data preparation completed!")