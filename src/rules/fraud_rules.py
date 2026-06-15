import pandas as pd
import numpy as np


'''
2 часть проекта rule-based движок

10 правил для детекции фрода:
1.Крупный P2P перевод — MCC 6012 + сумма больше 400,000 тг
2.Повторные крупные снятия с банкомата — ATM + сумма больше 300,000 тг + 2 и более раз за день
3.Мелкие транзакции за короткое время — 5+ транзакций меньше 2,000 тг в течение 30 минут
4.Ночная транзакция (0:00 — 5:00)
5.Дроблинг — один и тот же мерчант проводит 3+ транзакции за день
6.Международная транзакция через WWW
7.Ручной ввод карты (Keyin) с суммой больше 200,000 тг
8.Транзакция в нерабочее время за рубежом — страна не KZ + ночное время
9.Одинаковая сумма повторяется у одного мерчанта 2+ раза подряд
10. Транзакция в новой стране сразу после транзакции в KZ — один клиент платит в KZ и через короткое время (менее 1 часа)
уже платит за рубежом (физически невоз  можно)
11.Высокорисковый MCC (азартные игры, ломбарды, и т.д.) с суммой больше 200,000 тг
'''

def apply_rules_based_engine(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()

    df['TRANSACTION_DATE'] = pd.to_datetime(df['TRANSACTION_DATE'])
    df['HOUR'] = df['TRANSACTION_DATE'].dt.hour
    df['DATE_STR'] = df['TRANSACTION_DATE'].dt.strftime('%Y-%m-%d')
    df['MCC'] = df['MCC'].astype(str)

    df = df.sort_values(['MEMBER_NO', 'TRANSACTION_DATE']).reset_index(drop=True)

    high_risk_mccs = [
        '4829', '5966', '5967', '5968', '6012', '6051', '6211',
        '6536', '6537', '6538', '6540', '7273', '7800', '7801', '7802', '7995'
    ]

#1 правило
    df['R1_large_p2p'] = ((df['MCC']=='6012')&(df['SETTLEMENT_AMOUNT']>400_000)).astype(int)

#2 правило
    atm_large = (df['TRX_TYPE']=='ATM') & (df['SETTLEMENT_AMOUNT']>300_000)
    df['R2_large_atm'] = df[atm_large].groupby(['MEMBER_NO', 'DATE_STR'])['TRX_ID'].transform('count')
    df['R2_large_atm'] = (df['R2_large_atm'] >=2).astype(int)

# 3 правило
    df['is_small'] = (df['SETTLEMENT_AMOUNT'] < 2000).astype(int)

    def count_in_window(group):
        return group.rolling('30min', on='TRANSACTION_DATE')['TRX_ID'].count()

    small_df = df[df['is_small'] == 1]

    if not small_df.empty:
        # include_groups=False убирает назойливый DeprecationWarning
        df.loc[df['is_small'] == 1, 'R3_small_trans'] = small_df.groupby('MEMBER_NO', group_keys=False, include_groups=False).apply(
            lambda x: count_in_window(x)
        )
    else:
        df['R3_small_trans'] = 0

    df['R3_small_trans'] = (df['R3_small_trans'].fillna(0) >= 5).astype(int)
    
#4 правило
    df['R4_night_trans'] = ((df['HOUR']>=0)& (df['HOUR'] <= 5) & (df['SETTLEMENT_AMOUNT'] >= 50_000)).astype(int)

#5 правило
    df['R5_structuring'] = df.groupby(['MEMBER_NO', 'MERCHANT_NO', 'DATE_STR'])['TRX_ID'].transform('count')
    df['R5_structuring'] = (df['R5_structuring'] >= 3).astype(int)

#6 правило
    df['R6_inter_www'] = ((df['NATIONAL_CODE'] != 'KZ')& (df['TRX_TYPE'] == 'WWW')).astype(int)

#7 правило
    df['R7_manual_card'] = ((df['POS_MODE'].str.contains('Keyin', case = False, na = False)) & (df['SETTLEMENT_AMOUNT'] > 200_000)).astype(int)

#8 правило
    df['R8_foreign_night'] = ((df['NATIONAL_CODE'] != 'KZ') & (df['HOUR']>=0)& (df['HOUR'] <= 5)).astype(int)
#9 правило
    df['prev_amount'] = df.groupby('MERCHANT_NO')['SETTLEMENT_AMOUNT'].shift(1)
    df['R9_same_amount'] = ((df['SETTLEMENT_AMOUNT'] == df['prev_amount'])).astype(int)

#10 правило
    df['prev_country'] = df.groupby('MEMBER_NO')['NATIONAL_CODE'].shift(1)
    df['prev_time'] = df.groupby('MEMBER_NO')['TRANSACTION_DATE'].shift(1)
    time_diff_hours = (df['TRANSACTION_DATE'] - df['prev_time']).dt.total_seconds() / 3600.0

    df['R10_impossible'] = ((df['NATIONAL_CODE'] != 'KZ') &
                            (df['prev_country'] == 'KZ') &
                            (time_diff_hours < 0.5)).astype(int)

# 11 правило
    df['R11_high_risk_mcc'] = ((df['MCC'].isin(high_risk_mccs)) & (df['SETTLEMENT_AMOUNT'] > 200_000)).astype(int)


    # Для красоты пересоберем список колонок правил:
    exact_rule_cols = ['R1_large_p2p', 'R2_large_atm', 'R3_small_trans', 'R4_night_trans',
                       'R5_structuring', 'R6_inter_www', 'R7_manual_card',
                       'R8_foreign_night', 'R9_same_amount', 'R10_impossible', 'R11_high_risk_mcc']



    df[exact_rule_cols] = df[exact_rule_cols].fillna(0).astype(int)

    # Итоговый вердикт rule-based системы
    df['RULE_PREDICTION'] = (df[exact_rule_cols].sum(axis=1) > 0).astype(int)

    return df




if __name__ == "__main__":
    # Загружаем размеченные данные из Части 1
    df_labeled = pd.read_csv('data/processed/transactions_labeled.csv')

    print("Запуск rule-based движка...")
    result_df = apply_rules_based_engine(df_labeled)

    # Считаем матрицу ошибок (Confusion Matrix)
    print("\n--- Результаты детекции правил против реального фрода (IS_FRAUD) ---")
    print(pd.crosstab(result_df['IS_FRAUD'], result_df['RULE_PREDICTION'],
                      rownames=['Реальный фрод'], colnames=['Предсказание правил']))

    exact_rule_cols = ['R1_large_p2p', 'R2_large_atm', 'R3_small_trans', 'R4_night_trans',
                       'R5_structuring', 'R6_inter_www', 'R7_manual_card',
                       'R8_foreign_night', 'R9_same_amount', 'R10_impossible', 'R11_high_risk_mcc']

    print("\n--- Сколько раз сработало каждое правило ---")
    for col in exact_rule_cols:
        count = result_df[col].sum()
        print(f"{col}: {count}")






