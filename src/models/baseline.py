import os
from idlelib.colorizer import prog_group_name_to_tag

import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def run_baseline(input_path: str):
    print('\n---- Запуск Baseline ML-модели')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Файл данных не найден: {input_path}')
    df = pd.read_csv(input_path)

    #1 Отбираем базовые фичи для модели (Текст и Id убираем, чтобы модель не переобучилась)
    #нам нужны только те колонки, которые можно быстро перевести в числа
    features = ['SETTLEMENT_AMOUNT', 'SALES_AMOUNT', 'TRX_TYPE', 'MCC', 'NATIONAL_CODE']
    target = 'IS_FRAUD'

    X = df[features].copy()
    y = df[target].copy()

    #2 Простейшее кодирование категориальных признаков (One-Hot Encoding)
    X = pd.get_dummies(X, columns = ['TRX_TYPE', 'MCC', 'NATIONAL_CODE'], drop_first = True)

    #3 Разделение на Train/Test (80на20) с сохранением пропорции фрода (stratify)
    X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

    print(f'Размер обучающей выборки: {X_train.shape[0]:,}')
    print(f'Размер тестовой выборки: {X_test.shape[0]:,}')
    print(f'Доля фрода в тесте: {y_test.mean()*100:.2f}')

    #4 ОБучение модели (Random Forest как классический и стабильный бейзлайн)
    print('Обучение RandomForestClassifier...')
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    #5 Предсказание и оценка
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,-1]

    print('\n[Результаты на тестовой выборке]')
    print(classification_report(y_test,preds))
    print(f'ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}')

    print('\n Confusion matrix')
    print(confusion_matrix(y_test, preds))


if __name__ == '__main__':
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'transactions_labeled.csv')
    )
    run_baseline(path)

