import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

'''
Генератор синтетических банковских транзакций для детекции фрода
Seed фиксирован - данные воспроизводимы при любом запуске
'''


#------КОНФИГ------
SEED = 42
N_CLIENTS = 500
N_ROWS = 50000
FRAUD_RATE = 0.02
OUTPUT_RAW = 'data/raw/transactions.csv'
OUTPUT_PROC = 'data/processed/transactions_labeled.csv'

rng = np.random.default_rng(SEED)

#-----Справочники-----
CITY_COUNTRY = {
    'KZ': ['ALMATY', 'ASTANA', 'AKTAU', 'ATYRAU', 'TALDYKORGAN', 'SEMEY', 'SHYMKENT', 'TARAZ', 'KARAGANDA', 'KOSTANAY', 'PAVLODAR'],
    'RU': ['MOSCOW', 'SAINT PETERSBURG', 'NOVOSIBIRSK', 'YEKATERINBURG'],
    'US': ['NEW YORK', 'LOS ANGELES', 'CHICAGO', 'MIAMI', 'SAN FRANCISCO'],
    'CN': ['BEIJING', 'SHANGHAI', 'SHENZHEN', 'GUANGZHOU'],
    'IN': ['DELHI', 'MUMBAI', 'BANGALORE'],
    'BR': ['SAO PAULO', 'RIO DE JANEIRO', 'BRASILIA'],
    'DE': ['BERLIN', 'MUNICH', 'FRANKFURT'],
    'FR': ['PARIS', 'LYON', 'MARSEILLE'],
    'GB': ['LONDON', 'MANCHESTER', 'EDINBURGH'],
    'JP': ['TOKYO', 'OSAKA', 'KYOTO'],
    'KR': ['SEOUL', 'BUSAN'],
    'AU': ['SYDNEY', 'MELBOURNE'],
    'CA': ['TORONTO', 'VANCOUVER', 'MONTREAL'],
    'MX': ['MEXICO CITY', 'CANCUN'],
    'IT': ['ROME', 'MILAN', 'VENICE'],
    'ES': ['MADRID', 'BARCELONA'],
    'NL': ['AMSTERDAM', 'ROTTERDAM'],
    'SE': ['STOCKHOLM', 'GOTHENBURG'],
    'CH': ['ZURICH', 'GENEVA'],
    'AE': ['DUBAI', 'ABU DHABI'],
    'TR': ['ISTANBUL', 'ANTALYA', 'ANKARA'],
    'UZ': ['TASHKENT', 'SAMARKAND', 'BUKHARA'],
    'KG': ['BISHKEK', 'OSH'],
    'TH': ['BANGKOK', 'PHUKET'],
    'GE': ['TBILISI', 'BATUMI'],
    'MY': ['KUALA LUMPUR'],
    'SGP': ['SINGAPORE'],
    'EG': ['CAIRO', 'SHARM EL SHEIKH'],
    'PL': ['WARSAW', 'KRAKOW']
}

kz_weight = 0.65
other_countries = [c for c in CITY_COUNTRY.keys() if c != 'KZ']
other_weight = (1.0 - kz_weight) / len(other_countries)
MERCHANTS = {
    #MCC : (name_template, avg_amount, std_amount)
    5411:('Supermarket {}', 5000, 3000), #продукты
    5812:('Cafe and restaraunts {}', 3000, 2000), #рестораны
    5912:('Pharmacy {}', 2000, 1500), #Аптеки
    4111:('Transport {}', 1000, 500), # транспорт
    5999:('Online store {}', 8000, 6000), # разное онлайн
    6011:('ATM Withdrawal', 20000, 15000), # баноматы
    7011:('Hotel {}', 30000, 20000), #отели
    4816:('Digital Service {}', 5000, 4000), #подписки/IT
    5944:('Jewelry {}', 50000, 40000), # ювелирные (риск)
    5065:('Elecrtronics {}', 40000, 30000) #Электроника
}

mccs = ['5691', '5812', '5814', '5815', '5816', '5817', '5818', '5819', '5820', '5821',
            '5912', '5921', '5931', '5932', '5933', '5940', '5941', '5942', '5943', '5944',
            '5945', '5946', '5947', '5948', '5949', '5950', '5960', '5961', '5962', '5963',
            '5964', '5965', '5966', '5967', '5968', '5969', '5970', '5971', '5972', '5973',
            '5974', '5975', '5976', '5977', '5978', '5979', '5980', '5981', '5982', '5983',
            '5984', '5985', '5986', '5987', '5988', '5989', '5990', '5991', '5992', '5993',
            '5994', '5995', '5996', '5997', '5998', '5999', '6010', '6011', '6012', '6013',
            '6014', '6015', '6016', '6017', '6018', '6019', '6020', '6021', '6022', '6023',
            '6024', '6025', '6026', '6027', '6028', '6029', '6030', '6031', '6032', '6033',
            '6034', '6035', '6036', '6037', '6038', '6039', '6040', '6041', '6042', '6043',
            '6044', '6045', '6046', '6047', '6048', '6049', '6050', '6051', '6052', '6053',
            '6054', '6055', '6056', '6057', '6058', '6059', '6060', '6061', '6062', '6063',
            '6064', '6065', '6066', '6067', '6068', '6069', '6070', '6071', '6072', '6073',
            '6074', '6075', '6076', '6077', '6078', '6079', '6080', '6081', '6082', '6083',
            '6084', '6085', '6086', '6087', '6088', '6089', '6090', '6091', '6092', '6093',
            '6094', '6095', '6096', '6097', '6098', '6099', '6100', '6101', '6102', '6103']


POS_MODES = ['0710 - Contactless - VSDC chip',
        '0120 - Keyin',
        '0730 - Contactless - VSDC chip',
        '1000 - Credential on file',
        '0720 - Contactless - VSDC magstripe',
        '0110 - Swipe',
        '0800 - Contactless - Magstripe',
        '0000 - Unknown',
        '0510 - Contactless - Magstripe',
        '0001 - E-commerce',
        '0002 - Mail/Phone order',
        '0003 - Recurring',
        '0004 - Installment',
        '0005 - Preauthorized',
        '0006 - Cash',
        '0007 - Other',
        '0008 - Not used' ]

PRODUCT_NAMES = [
        '1308112001: Debit Purchase Transaction(Domestic)',
        '1308122001: Debit Purchase Transaction(International)',
        '1308132001: Credit Purchase Transaction(Domestic)',
        '1308142001: Credit Purchase Transaction(International)',
        '1308152001: Prepaid Purchase Transaction(Domestic)',
        '1308162001: Prepaid Purchase Transaction(International)',
        '1308172001: Cash Withdrawal Transaction(Domestic)',
        '1308182001: Cash Withdrawal Transaction(International)',
        '1308192001: Balance Inquiry Transaction(Domestic)',
        '1308202001: Balance Inquiry Transaction(International)',
        '1308212001: Refund Transaction(Domestic)',
        '1308222001: Refund Transaction(International)',
        '1308232001: Transfer Transaction(Domestic)',
        '1308242001: Transfer Transaction(International)',
        '1308252001: Bill Payment Transaction(Domestic)',
        '1308262001: Bill Payment Transaction(International)',
        '1308272001: Other Transaction(Domestic)',
        '1308282001: Other Transaction(International)'
    ]

TRX_TYPES = ['PURCHASE', 'WITHDRAWAL', 'REFUND', 'TRANSFER']

#----Генерация клиентов-----
def make_clients(n:int) -> pd.DataFrame:
    '''Базовый профиль каждого клиента: средний чек, любимый МСС, продукт'''
    client_ids = [f"C{str(i).zfill(6)}" for i in range(1, n + 1)]
    avg_amounts = rng.lognormal(mean=8.5, sigma=0.8, size=n)  # ~5 000 тг
    fav_mcc = rng.choice(mccs, size=n)
    product = rng.choice(PRODUCT_NAMES, size=n)
    return pd.DataFrame({
        "client_id": client_ids,
        "avg_amount": avg_amounts,
        "fav_mcc": fav_mcc,
        "product": product,
    })


#-----Генерация одной «нормальной» транзакции---------
def normal_transaction(client: pd.Series, trx_date: datetime) -> dict:
    mcc = int(rng.choice(list(MERCHANTS.keys())))
    tmpl, avg, std = MERCHANTS[mcc]
    amount    = max(100, rng.normal(avg, std))
    nat_code = rng.choice(list(CITY_COUNTRY.keys()),
                          p=[kz_weight] + [other_weight] * len(other_countries))
    city = rng.choice(CITY_COUNTRY[nat_code])
    pos_mode  = rng.choice(POS_MODES)
    merchant_no = f"M{rng.integers(10000, 99999)}"
    merchant_name = tmpl.format(rng.integers(1, 999)) if "{}" in tmpl else tmpl

    auth_date = trx_date + timedelta(seconds=int(rng.integers(1, 120)))

    return {
        "MEMBER_NO":          client["client_id"],
        "TRANSACTION_DATE":   trx_date.strftime("%Y-%m-%d %H:%M:%S"),
        "AUTHORIZATION_DATE": auth_date.strftime("%Y-%m-%d %H:%M:%S"),
        "MERCHANT_NO":        merchant_no,
        "MERCHANT_NAME":      merchant_name,
        "CITY_NAME":          city,
        "NATIONAL_CODE":      nat_code,
        "MCC":                mcc,
        "POS_MODE":           pos_mode,
        "PRODUCT_NAME":       client["product"],
        "SETTLEMENT_AMOUNT":  round(amount, 2),
        "SALES_AMOUNT":       round(amount * rng.uniform(0.97, 1.00), 2),
        "TRX_TYPE":           rng.choice(TRX_TYPES, p=[0.75, 0.15, 0.05, 0.05]),
        "IS_FRAUD":           0,
    }

#-----Фродовые паттерны------
FRAUD_PATTERNS = ["high_amount", "foreign_location", "unusual_hour",
                  "card_testing", "mcc_mismatch"]

def fraud_transaction(client: pd.Series, trx_date: datetime) -> dict:
    """Берём нормальную транзакцию и «ломаем» её одним из паттернов."""
    row = normal_transaction(client, trx_date)
    pattern = rng.choice(FRAUD_PATTERNS)

    if pattern == "high_amount":
        # Аномально большой чек
        row["SETTLEMENT_AMOUNT"] = round(float(client["avg_amount"]) * rng.uniform(10, 50), 2)
        row["SALES_AMOUNT"]      = row["SETTLEMENT_AMOUNT"]

    elif pattern == "foreign_location":
        # Транзакция из подозрительной страны
        suspicious = [c for c in CITY_COUNTRY.keys() if c != 'KZ']
        nat_code = rng.choice(suspicious)
        row["CITY_NAME"] = rng.choice(CITY_COUNTRY[nat_code])
        row["NATIONAL_CODE"] = nat_code
        row["POS_MODE"]       = "ECOMMERCE"

    elif pattern == "unusual_hour":
        # Транзакция в 0–5      ночи
        night_hour = int(rng.integers(0, 5))
        d = trx_date.replace(hour=night_hour,
                             minute=int(rng.integers(0, 59)))
        row["TRANSACTION_DATE"]   = d.strftime("%Y-%m-%d %H:%M:%S")
        row["AUTHORIZATION_DATE"] = (d + timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S")
        row["SETTLEMENT_AMOUNT"]  = round(float(client["avg_amount"]) * rng.uniform(5, 20), 2)

    elif pattern == "card_testing":
        # Много мелких транзакций (имитируется суммой < 100)
        row["SETTLEMENT_AMOUNT"] = round(rng.uniform(1, 99), 2)
        row["SALES_AMOUNT"]      = row["SETTLEMENT_AMOUNT"]
        row["MCC"]               = '4816'  # цифровые сервисы

    elif pattern == "mcc_mismatch":
        # Ювелирка или электроника с необычным POS
        row["MCC"]               = rng.choice(['5944', '5065'])
        row["SETTLEMENT_AMOUNT"] = round(rng.uniform(80_000, 300_000), 2)
        row["POS_MODE"]          = "MAGNETIC"
        row["MERCHANT_NAME"]     = "Unknown Merchant"

    row["IS_FRAUD"] = 1
    return row
#-----Основной генератор-------
def generate(n_rows: int = N_ROWS) -> pd.DataFrame:
    clients  = make_clients(N_CLIENTS)
    start_dt = datetime(2024, 1, 1)
    end_dt   = datetime(2024, 12, 31)
    delta_s  = int((end_dt - start_dt).total_seconds())

    rows = []
    n_fraud  = int(n_rows * FRAUD_RATE)
    n_normal = n_rows - n_fraud

    fraud_indices  = set(rng.choice(n_rows, size=n_fraud, replace=False))

    for i in range(n_rows):
        client   = clients.iloc[int(rng.integers(0, N_CLIENTS))]
        trx_date = start_dt + timedelta(seconds=int(rng.integers(0, delta_s)))

        if i in fraud_indices:
            row = fraud_transaction(client, trx_date)
        else:
            row = normal_transaction(client, trx_date)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("TRANSACTION_DATE").reset_index(drop=True)
    df.insert(0, "TRX_ID", [f"T{str(j).zfill(8)}" for j in range(len(df))])
    return df


#-----Сохранение------
def save(df: pd.DataFrame):
    os.makedirs("data/raw",       exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # RAW — без метки (как будто реальные данные)
    df.drop(columns=["IS_FRAUD"]).to_csv(OUTPUT_RAW,  index=False)

    # PROCESSED — с меткой (для обучения модели)
    df.to_csv(OUTPUT_PROC, index=False)

    print(f"   Сгенерировано {len(df):,} транзакций")
    print(f"   Фрод:   {df['IS_FRAUD'].sum():,}  ({df['IS_FRAUD'].mean()*100:.1f}%)")
    print(f"   Норма:  {(~df['IS_FRAUD'].astype(bool)).sum():,}")
    print(f"   RAW  → {OUTPUT_RAW}")
    print(f"   PROC → {OUTPUT_PROC}")


if __name__ == "__main__":
    print("Генерация данных (seed=42)...")
    df = generate()
    save(df)
    print("\nПервые 5 строк:")
    print(df.head().to_string(index=False))

