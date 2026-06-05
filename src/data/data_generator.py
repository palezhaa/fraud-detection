import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data(num_records = 20000, fraud_ratio = 0.015, seed = 42):
    np.random.seed(seed)
    random.seed(seed)

    legit_mccs_with_weights = {
        '5411': 0.40,
        '5499': 0.10,
        '5814': 0.15,
        '4121': 0.12,
        '5331': 0.08,
        '5812': 0.05,
        '5912': 0.04,
        '5541': 0.03,
        '5691': 0.02,
        '5977': 0.01,
    }

    gambling_mccs = ['7995', '7800', '7801', '7802']
    crypto_invest_mccs = ['6051', '6211', '6282', '6529']
    card_testing_mccs = ['5815', '5816', '5817', '5818', '5734', '5942', '5999']
    cashout_mccs = ['4829', '6011', '6536', '6537', '6540']
    
    pos_modes = [
        '0710 - Contactless - VSDC chip',
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
        '0008 - Not used'
    ]
    
    product_names = [
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
    
    trx_types = [
        'POS',
        'ATM',
        'WWW'
    ]

    country_city_map = {
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

    num_other_countries = len(country_city_map) - 1
    kz_weight = 0.65
    other_weight = (1.0 - kz_weight) / num_other_countries
    nat_codes = list(country_city_map.keys())
    nat_weights = [kz_weight] + [other_weight] * num_other_countries

    num_fraud = int(num_records*fraud_ratio)
    num_normal = num_records-num_fraud

    data = []

    base_date = datetime(2026, 5, 20)

    mccs_list = list(legit_mccs_with_weights.keys())
    mccs_weights = list(legit_mccs_with_weights.values())

    for _ in range(num_normal):
        user_id = f"{random.choice(['P', 'C'])}{random.randint(1, 5000):09d}"

        days_offset = random.randint(0, 30)
        hours_offset = random.randint(0, 23)
        minutes_offset = random.randint(0, 59)
        seconds_offset = random.randint(0, 59)
        trx_datetime = base_date + timedelta(
            days=days_offset, 
            hours=hours_offset, 
            minutes=minutes_offset, 
            seconds=seconds_offset
        )

        auth_datetime = trx_datetime + timedelta(seconds=random.randint(0, 3))

        trx_time = trx_datetime.strftime("%Y-%m-%d %H:%M:%S")
        auth_time = auth_datetime.strftime("%Y-%m-%d %H:%M:%S")

        selected_mcc = random.choices(mccs_list, weights=mccs_weights, k=1)[0]

        if selected_mcc in ['5411', '5499', '5814', '4121']:
            sales_amt = round(float(np.random.lognormal(mean=7.5, sigma=0.8)), 2) # мелкие/средние чеки
            sales_amt = max(500.0, min(sales_amt, 15000.0))
        else:
            sales_amt = round(float(np.random.lognormal(mean=9.5, sigma=1.0)), 2) # крупные чеки
            sales_amt = max(3000.0, min(sales_amt, 100000.0))

        nat_code = random.choices(nat_codes, weights=nat_weights, k=1)[0]
        
        city = random.choice(country_city_map[nat_code])
        prod = product_names[0] if nat_code == 'KZ' else random.choice(product_names)

        data.append({
            'MEMBER_NAME': user_id,
            'TRANSACTION_DATE': trx_time,
            'AUTHORIZATION_DATE': auth_time,
            'MERCHANT_NO': str(random.randint(1000000000, 9999999999)),
            'MERCHANT_NAME': random.choice(['WALMART', 'AMAZON', 'STARBUCKS', 'APPLE STORE', 
                                            'SAMSUNG', 'NIKE', 'ADIDAS', 'TARGET', 
                                            'BEST BUY', 'COSTCO', 'HOME DEPOT', 'MACY\'S', 
                                            'SEPHORA', 'GAP', 'H&M', 'ZARA', 
                                            'UNIQLO', 'LOWE\'S', 'WAYFAIR', 'EBAY']),
            'CITY_NAME': city,
            'NATIONAL_CODE': nat_code,
            'MCC': selected_mcc,
            'POS_MODE': random.choice(pos_modes),
            'PRODUCT_NAME': prod,
            'SETTLEMENT_AMOUNT': sales_amt,
            'SALES_AMOUNT': sales_amt,
            'TRX_TYPE': random.choice(trx_types),
            'IS_FRAUD': 0
        })

    fraud_patterns = ['card_testing', 'gambling_scam', 'cashout', 'crypto_drain']
    for _ in range(num_fraud):
        user_id = f"{random.choice(['P', 'C'])}{random.randint(1, 5000):09d}"
        fraud_hour = random.choice([0, 1, 2, 3, 4, 5]) 
        base_fraud_time = base_date + timedelta(days=random.randint(0, 30), hours=fraud_hour, minutes=random.randint(0, 45))
        time_lag_from_compromise = timedelta(seconds=random.randint(1, 120)) 
        trx_datetime = base_fraud_time + time_lag_from_compromise

        auth_datetime = trx_datetime + timedelta(seconds=1) 

        trx_time = trx_datetime.strftime("%Y-%m-%d %H:%M:%S")
        auth_time = auth_datetime.strftime("%Y-%m-%d %H:%M:%S")

        pattern = random.choice(fraud_patterns)

        nat_code = random.choices(nat_codes, weights=nat_weights, k=1)[0]
        city = random.choice(country_city_map[nat_code])

        if pattern == 'card_testing':
            # правило F017: малые суммы <1000 KZT для проверки карты
            selected_mcc = random.choice(card_testing_mccs)
            sales_amt = round(random.uniform(100.0, 950.0), 2)
            merchant = 'STEAM GAMES' if selected_mcc == '5816' else 'MICROSOFT_STORE'
            pos_mode = '0001 - E-commerce'
            trx_type = 'WWW'

        elif pattern == 'gambling_scam':
            # правило F003 / High Risk MCC (гемблинг)
            selected_mcc = random.choice(gambling_mccs)
            sales_amt = round(random.uniform(50000.0, 250000.0), 2)
            merchant = 'CASINO_ONLINE_XYZ'
            pos_mode = '0001 - E-commerce'
            trx_type = 'WWW'

        elif pattern == 'cashout':
            # правило F038: быстрый вывод/снятие наличных или P2P переводы
            selected_mcc = random.choice(cashout_mccs)
            sales_amt = round(random.uniform(200000.0, 500000.0), 2)
            merchant = 'P2P_TRANSFER_DROP'
            pos_mode = '0110 - Swipe' if selected_mcc == '6011' else '0001 - E-commerce'
            trx_type = 'ATM' if selected_mcc == '6011' else 'WWW'

        else:
            # пополнение крипты/брокеров с высокими MCC для инвестиций/крипты
            selected_mcc = random.choice(crypto_invest_mccs)
            sales_amt = float(random.choice([400000, 800000, 1190000]))
            merchant = 'BINANCE CRYPTO'
            pos_mode = '1000 - Credential on file'
            trx_type = 'WWW'

        data.append({
            'MEMBER_NAME': user_id,
            'TRANSACTION_DATE': trx_time,
            'AUTHORIZATION_DATE': auth_time,
            'MERCHANT_NO': str(random.randint(1000000000, 9999999999)),
            'MERCHANT_NAME': merchant,
            'CITY_NAME': city,
            'NATIONAL_CODE': nat_code,
            'MCC': selected_mcc,
            'POS_MODE': pos_mode,
            'PRODUCT_NAME': random.choice(product_names),
            'SETTLEMENT_AMOUNT': sales_amt,
            'SALES_AMOUNT': sales_amt,
            'TRX_TYPE': trx_type,
            'IS_FRAUD': 1
        })

    df = pd.DataFrame(data)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    string_cols = ['MEMBER_NAME', 'MERCHANT_NO', 'MERCHANT_NAME', 'CITY_NAME', 'NATIONAL_CODE', 
                   'MCC', 'POS_MODE', 'PRODUCT_NAME', 'TRX_TYPE']
    df[string_cols] = df[string_cols].astype(str)
    return df


if __name__ == "__main__":
    df = generate_data()
    df.to_csv("transactions.csv", index=False)
