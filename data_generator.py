import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data(num_records = 20000, fraud_ratio = 0.015, seed = 42):
    np.random.seed(seed)
    random.seed(seed)

    legit_mccs_with_weights = {
        '5411': 0.40, '5499': 0.10, '5814': 0.15, '4121': 0.12, '5331': 0.08,
        '5812': 0.05, '5912': 0.04, '5541': 0.03, '5691': 0.02, '5977': 0.01,
        '7995': 0.01, '6051': 0.01, '5816': 0.02, '5999': 0.01
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
    kz_weight = 0.80
    other_weight = (1.0 - kz_weight) / num_other_countries
    nat_codes = list(country_city_map.keys())
    nat_weights = [kz_weight] + [other_weight] * num_other_countries

    num_fraud = int(num_records*fraud_ratio)
    num_normal = num_records-num_fraud

    data = []

    base_date = datetime(2026, 6, 9)

    user_profiles = {}
    for prefix in ['P', 'C']:
        for i in range(1, 5001):
            uid = f"{prefix}{i:09d}"
            user_profiles[uid] = {
                'home_country': 'KZ' if random.random() < 0.9 else random.choice(nat_codes),
                'avg_amount': float(np.random.lognormal(mean=8.0, sigma=0.5))
            }

    mccs_list = list(legit_mccs_with_weights.keys())
    mccs_weights = list(legit_mccs_with_weights.values())

    for _ in range(num_normal):
        user_id = f"{random.choice(['P', 'C'])}{random.randint(1, 5000):09d}"
        profile = user_profiles[user_id]

        days_offset = random.randint(0, 30)
        hours_offset = random.choices(list(range(24)), weights=[1,1,1,1,2,3,5,10,15,20,25,30,35,35,30,25,30,35,40,35,25,15,10,5], k=1)[0]
        minutes_offset = random.randint(0, 59)
        seconds_offset = random.randint(0, 59)
        trx_datetime = base_date + timedelta(
            days=days_offset, 
            hours=hours_offset, 
            minutes=minutes_offset, 
            seconds=seconds_offset
        )

        auth_datetime = trx_datetime + timedelta(seconds=random.randint(1, 3))

        trx_time = trx_datetime.strftime("%Y-%m-%d %H:%M:%S")
        auth_time = auth_datetime.strftime("%Y-%m-%d %H:%M:%S")

        selected_mcc = random.choices(mccs_list, weights=mccs_weights, k=1)[0]

        sales_amt = round(float(np.random.normal(loc=profile['avg_amount'], scale=profile['avg_amount']*0.3)), 2)
        sales_amt = max(200.0, sales_amt)

        if selected_mcc in gambling_mccs + crypto_invest_mccs:
            sales_amt *= random.uniform(1.5, 3.0)
        
        nat_code = profile['home_country'] if random.random() < 0.95 else random.choice(nat_codes)
        
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
                                            'UNIQLO', 'LOWE\'S', 'WAYFAIR', 'EBAY', 
                                            'MAGNUM', 'YANDEX TAXI', 'AMAZON', 'KFC', 
                                            'WILD BERRIES', 'OLIMP_KZ']),
            'CITY_NAME': city,
            'NATIONAL_CODE': nat_code,
            'MCC': selected_mcc,
            'POS_MODE': random.choice(pos_modes),
            'PRODUCT_NAME': prod,
            'SETTLEMENT_AMOUNT': sales_amt,
            'SALES_AMOUNT': sales_amt,
            'TRX_TYPE': 'WWW' if 'E-commerce' in pos_modes else 'POS',
            'IS_FRAUD': 0
        })

    fraud_patterns = ['F005_atypical_amount', 'F007_card_testing', 'F038_cashout', 'F020_velocity_countries']
    for _ in range(num_fraud):
        user_id = f"{random.choice(['P', 'C'])}{random.randint(1, 5000):09d}"
        profile = user_profiles[user_id]

        days_offset = random.randint(0, 30)
        fraud_pattern = random.choice(fraud_patterns)

        if fraud_pattern == 'F005_atypical_amount':
            # Нетипичная сумма в High-Risk MCC или обычном супермаркете 
            selected_mcc = random.choice(['5411', '6051', '7995']) 
            sales_amt = profile['avg_amount'] * random.uniform(6.0, 10.0) # Правило F005: Amount > Avg * 5 
            
            fraud_hour = random.choice([1, 2, 3, 4, 5]) # Ночная активность (F009) 
            trx_datetime = base_date + timedelta(days=days_offset, hours=fraud_hour, minutes=random.randint(0,59))
            auth_datetime = trx_datetime + timedelta(seconds=1) # Мгновенная атака после компрометации
            
            merchant = 'BINANCE' if selected_mcc == '6051' else 'MAGNUM_VIP'
            nat_code = profile['home_country']
            pos_mode = '0001 - E-commerce'
            trx_type = 'WWW'

        elif fraud_pattern == 'F007_card_testing':
            # Card Testing (F017/F007) — серия мелких транзакций в Digital/Games [cite: 2, 4]
            selected_mcc = random.choice(card_testing_mccs)
            sales_amt = round(random.uniform(150.0, 900.0), 2) # Мелкие суммы < 1000 KZT (F017) [cite: 4]
            
            trx_datetime = base_date + timedelta(days=days_offset, hours=random.randint(10,22))
            # Симулируем, что транзакции бьют в одну и ту же секунду/минуту
            auth_datetime = trx_datetime + timedelta(seconds=random.randint(0, 2))
            
            merchant = 'STEAM_GAMES'
            nat_code = 'US' # Мошенники часто тестируют карты на зарубежных мерчантах (F001) 
            pos_mode = '0001 - E-commerce'
            trx_type = 'WWW'

        elif fraud_pattern == 'F038_cashout':
            # Быстрый вывод средств (Cash-out / P2P) [cite: 8]
            selected_mcc = random.choice(cashout_mccs)
            sales_amt = round(random.uniform(150000.0, 450000.0), 2) # Крупный вывод [cite: 11, 13]
            
            # Аномалия дат: TRANSACTION_DATE и AUTHORIZATION_DATE имеют жесткий тайминг "на поражение"
            trx_datetime = base_date + timedelta(days=days_offset, hours=random.choice([23, 0, 1, 2]))
            auth_datetime = trx_datetime + timedelta(milliseconds=200) # Процессинг моментально подтверждает P2P
            
            merchant = 'P2P_TRANSFER_DROPPER'
            nat_code = profile['home_country']
            pos_mode = '1000 - Credential on file'
            trx_type = 'WWW'

        elif fraud_pattern == 'F020_velocity_countries':
            # Невозможное перемещение / Несколько стран (F020) [cite: 4]
            selected_mcc = '5411' # Мошенник использует карту в обычном супермаркете за рубежом!
            sales_amt = round(float(np.random.lognormal(mean=8.5, sigma=0.5)), 2)
            
            trx_datetime = base_date + timedelta(days=days_offset, hours=random.randint(12, 18))
            auth_datetime = trx_datetime + timedelta(seconds=2)
            
            merchant = 'WALMART'
            # Страна гарантированно отличается от домашней
            nat_code = random.choice([c for c in nat_codes if c != profile['home_country']])
            pos_mode = '0110 - Swipe' # Пытаются прокатать дубликат магнитной полосы (F028) [cite: 6]
            trx_type = 'POS'

        city = random.choice(country_city_map[nat_code])
        prod = product_names[3] if nat_code != 'KZ' else product_names[2]

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
