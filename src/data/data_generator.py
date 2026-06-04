import types

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data(num_records = 20000, fraud_ratio = 0.015, seed = 42):
    np.random.seed(seed)
    random.seed(seed)

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
    num_normal = num_records - num_fraud

    data = []

    base_data = datetime(2026, 5, 20)

    for _ in range(num_normal):
        user_id = f"{random.choice(['P', 'C'])}{random.randint(1, 5000):09d}"

        days_offset = random.randint(0, 30)
        trx_time = (base_data + timedelta(days=days_offset)).strftime("%Y%m%d")

        sales_amt = round(float(np.random.lognormal(mean=9.5, sigma=1.2)), 2)
        sales_amt = max(1000.0, min(sales_amt, 150000.0))

        nat_code = random.choices(nat_codes, weights=nat_weights, k=1)[0]
        
        city = random.choice(country_city_map[nat_code])
        prod = product_names[0] if nat_code == 'KZ' else random.choice(product_names)

        data.append({
            'MEMBER_NAME': user_id,
            'TRANSACTION_DATE': trx_time,
            'AUTHORIZATION_DATE': trx_time,
            'MERCHANT_NO': str(random.randint(1000000000, 9999999999)),
            'MERCHANT_NAME': random.choice(['WALMART', 'AMAZON', 'STARBUCKS', 'APPLE STORE', 
                                            'SAMSUNG', 'NIKE', 'ADIDAS', 'TARGET', 
                                            'BEST BUY', 'COSTCO', 'HOME DEPOT', 'MACY\'S', 
                                            'SEPHORA', 'GAP', 'H&M', 'ZARA', 
                                            'UNIQLO', 'LOWE\'S', 'WAYFAIR', 'EBAY']),
            'CITY_NAME': city,
            'NATIONAL_CODE': nat_code,
            'MCC': random.choice([m for m in mccs if m != '6012']),
            'POS_MODE': random.choice(pos_modes),
            'PRODUCT_NAME': prod,
            'SETTLEMENT_AMOUNT': sales_amt,
            'SALES_AMOUNT': sales_amt,
            'TRX_TYPE': random.choice(trx_types),
            'IS_FRAUD': 0
        })

    for _ in range(num_fraud):
        user_id = f"{random.choice(['P', 'C'])}{random.randint(1, 5000):09d}"
        trx_time = (base_data + timedelta(days=random.randint(0, 30))).strftime("%Y%m%d")
        sales_amt = float(random.choice([300000, 400000, 500000, 800000, 1190000]))

        nat_code = random.choices(nat_codes, weights=nat_weights, k=1)[0]
        city = random.choice(country_city_map[nat_code])

        data.append({
            'MEMBER_NAME': user_id,
            'TRANSACTION_DATE': trx_time,
            'AUTHORIZATION_DATE': trx_time,
            'MERCHANT_NO': str(random.randint(1000000000, 9999999999)),
            'MERCHANT_NAME': random.choice(['WALMART', 'AMAZON',
                                            'STARBUCKS', 'APPLE STORE', 
                                            'SAMSUNG', 'NIKE', 'ADIDAS', 'TARGET', 
                                            'BEST BUY', 'COSTCO', 'HOME DEPOT', 'MACY\'S', 
                                            'SEPHORA', 'GAP', 'H&M', 'ZARA', 
                                            'UNIQLO', 'LOWE\'S', 'WAYFAIR', 'EBAY']),
            'CITY_NAME': city,
            'NATIONAL_CODE': nat_code,
            'MCC': '6012',
            'POS_MODE': random.choice(pos_modes),
            'PRODUCT_NAME': random.choice(product_names),
            'SETTLEMENT_AMOUNT': sales_amt,
            'SALES_AMOUNT': sales_amt,
            'TRX_TYPE': random.choice(trx_types),
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
