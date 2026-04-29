Структура:

├── data/ # данные

├── src/ # скрипты

├── dvc.yaml # описание пайплайна

├── params.yaml # гиперпараметры

├── requirements.txt # зависимости

└── README.md # документация      

________________________________________________

Требования

\- Python 3.8+

\- Git

\- DVC
________________________________________________
Шаги для воспроизведения:

1. Клонирование репозитория
&#x20; git clone <repo-url>
&#x20; cd mlops\_project

2. Установка зависимостей:
&#x20;pip install -r requirements.txt

3. Получение данных:
&#x20;dvc pull

4. Запуск пайплайна:
&#x20;dvc repro

5. Просмотр метрик в MLflow:
&#x20;mlflow ui --backend-store-uri sqlite:///mlflow.db

