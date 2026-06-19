import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_generator import generate_data
from src.rules.fraud_rules import apply_rules_based_engine
from src.models.baseline import run_baseline

def main():
    start_time = time.time()
    print('Старт единого оркестратора фрод-мониторинга')
    print('-'*45)

    #Опрделяем пути к файлам
    base_dir = os.path.dirname(__file__)
    raw_data_path = os.path.join(base_dir, 'data', 'processed', 'transactions_labeled.csv')


    #data generation
    print('Запуск генератора синтетических транзакций')
    df_raw = generate_data(num_records = 50000, fraud_ratio = 0.02, seed = 42)

    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    df_raw.to_csv(raw_data_path, index = False)
    print(f'данные сгенерированы и сохранены в {raw_data_path}')

    def pd_read_safely(path):
        import pandas as pd
        return pd.read_csv(path)
    def pd_crosstab_summary(df):
        import pandas as pd
        return  pd.crosstab(df['IS_FRAUD'], df['RULE_PREDICTION'], rownames=['Реальный фрод'], colnames=['Предсказаине правил'])

    # проверка rule_based движок
    print('прогон транзакций через rule-based движок')
    df_for_rules = pd_read_safely(raw_data_path)
    df_with_rules = apply_rules_based_engine(df_for_rules)

    #просто краткая аналитика правил
    print('Результаты детекции правил против реального фрода')
    print(pd_crosstab_summary(df_with_rules))


    #обучение baseline ML-модели
    print('передача данных в пайплайн машинного обучения')
    run_baseline(raw_data_path)

    end_time = time.time()
    print('-'*45)
    print(f'пайплайн выполнен за {end_time-start_time:.2f} сек')


if __name__ == '__main__':
    main()