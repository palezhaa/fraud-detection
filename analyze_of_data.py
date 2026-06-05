import pandas as pd

df = pd.read_csv('data/processed/transactions_labeled.csv')

print(list(df.columns))

fraud_df = (df[df['IS_FRAUD']=='1'][['TRX_ID', 'MEMBER_NO', 'TRANSACTION_DATE', 'AUTHORIZATION_DATE', 'MERCHANT_NO', 'MERCHANT_NAME', 'CITY_NAME', 'NATIONAL_CODE', 'MCC', 'POS_MODE', 'PRODUCT_NAME', 'SETTLEMENT_AMOUNT', 'SALES_AMOUNT', 'TRX_TYPE', 'IS_FRAUD']])

fraud_df.to_csv('data/processed/only_fraud_transactions.csv', index=False,  encoding = 'utf-8')
