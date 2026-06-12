import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_data(num_records: int = 20_000,
                  fraud_ratio: float = 0.015,
                  seed: int = 42) -> pd.DataFrame:

    np.random.seed(seed)
    random.seed(seed)

    legit_mccs_with_weights = {
        "5411": 0.40, "5499": 0.10, "5814": 0.15, "4121": 0.12,
        "5331": 0.08, "5812": 0.05, "5912": 0.04, "5541": 0.03,
        "5691": 0.02, "5977": 0.01, "7995": 0.01, "6051": 0.01,
        "5816": 0.02, "5999": 0.01,
        "6011": 0.05,}
    gambling_mccs      = ["7995", "7800", "7801", "7802"]
    crypto_invest_mccs = ["6051", "6211", "6282", "6529"]
    card_testing_mccs  = ["5815", "5816", "5817", "5818", "5734", "5942", "5999"]
    # 6011 намеренно исключен: для него TRX_TYPE='ATM', а P2P-паттерн
    # ниже всегда использует TRX_TYPE='WWW'. ATM вывод покрыт отдельным
    # паттерном F038_atm_cashout
    cashout_mccs       = ["4829", "6536", "6537", "6540"]

    MCC_MERCHANTS: dict[str, list[str]] = {
        "5411": ["MAGNUM", "WILD BERRIES", "SMALL", "GALMART", "RAMSTORE"],
        "5499": ["OLIMP_KZ", "MAGNUM", "FRESH_MARKET", "WILD BERRIES"],
        "5814": ["KFC", "BURGER_KING", "HARDEE'S", "DODO_PIZZA"],
        "4121": ["YANDEX TAXI", "INDRIVE", "CITYMOBIL"],
        "5331": ["TARGET", "WAYFAIR", "H&M", "FIX_PRICE"],
        "5812": ["STARBUCKS", "COSTA_COFFEE", "KFC"],
        "5912": ["APTEKA_EVROPA", "PHARMACY_24", "GREEN_APTEKA"],
        "5541": ["KAZMUNAYGAS", "SHELL", "HELIOS_PETROLEUM"],
        "5691": ["ZARA", "H&M", "UNIQLO", "GAP", "NIKE", "ADIDAS"],
        "5977": ["SEPHORA", "RIVE_GAUCHE", "L'ETOILE"],
        "7995": ["1XBET", "OLIMP_BET", "PARIMATCH"],
        "6051": ["BINANCE", "BYBIT", "CRYPTO_EXCHANGE"],
        "5816": ["STEAM_GAMES", "GOOGLE_PLAY", "APPLE STORE", "PLAYSTATION_STORE"],
        "5999": ["AMAZON", "EBAY", "WAYFAIR"],
        "6011": ["ATM_HALYK_KZ", "ATM_KASPI_KZ", "ATM_SBERBANK_KZ", "ATM_FORTEBANK"],
    }
    GENERIC_MERCHANTS = ["WALMART", "AMAZON", "TARGET", "COSTCO"]  # резерв для непредусмотренных MCC

    pos_modes_ecom = [
        "0001 - E-commerce",
        "0002 - Mail/Phone order",
        "0003 - Recurring",
        "0004 - Installment",
        "0005 - Preauthorized",
        "1000 - Credential on file",
    ]
    pos_modes_pos = [
        "0710 - Contactless - VSDC chip",
        "0730 - Contactless - VSDC chip",
        "0720 - Contactless - VSDC magstripe",
        "0110 - Swipe",
        "0120 - Keyin",
        "0800 - Contactless - Magstripe",
        "0510 - Contactless - Magstripe",
        "0000 - Unknown",
        "0007 - Other",
    ]
    pos_modes_atm = [
        "0006 - Cash",
        "0110 - Swipe",
        "0000 - Unknown",
    ]

    product_names_domestic = [
        "1308112001: Debit Purchase Transaction(Domestic)",
        "1308132001: Credit Purchase Transaction(Domestic)",
        "1308152001: Prepaid Purchase Transaction(Domestic)",
        "1308172001: Cash Withdrawal Transaction(Domestic)",
        "1308252001: Bill Payment Transaction(Domestic)",
    ]
    product_names_intl = [
        "1308122001: Debit Purchase Transaction(International)",
        "1308142001: Credit Purchase Transaction(International)",
        "1308162001: Prepaid Purchase Transaction(International)",
        "1308182001: Cash Withdrawal Transaction(International)",
        "1308262001: Bill Payment Transaction(International)",
    ]

    country_city_map = {
        "KZ":  ["ALMATY", "ASTANA", "AKTAU", "ATYRAU", "TALDYKORGAN",
                "SEMEY", "SHYMKENT", "TARAZ", "KARAGANDA", "KOSTANAY", "PAVLODAR"],
        "RU":  ["MOSCOW", "SAINT PETERSBURG", "NOVOSIBIRSK", "YEKATERINBURG"],
        "US":  ["NEW YORK", "LOS ANGELES", "CHICAGO", "MIAMI", "SAN FRANCISCO"],
        "CN":  ["BEIJING", "SHANGHAI", "SHENZHEN", "GUANGZHOU"],
        "IN":  ["DELHI", "MUMBAI", "BANGALORE"],
        "BR":  ["SAO PAULO", "RIO DE JANEIRO", "BRASILIA"],
        "DE":  ["BERLIN", "MUNICH", "FRANKFURT"],
        "FR":  ["PARIS", "LYON", "MARSEILLE"],
        "GB":  ["LONDON", "MANCHESTER", "EDINBURGH"],
        "JP":  ["TOKYO", "OSAKA", "KYOTO"],
        "KR":  ["SEOUL", "BUSAN"],
        "AU":  ["SYDNEY", "MELBOURNE"],
        "CA":  ["TORONTO", "VANCOUVER", "MONTREAL"],
        "MX":  ["MEXICO CITY", "CANCUN"],
        "IT":  ["ROME", "MILAN", "VENICE"],
        "ES":  ["MADRID", "BARCELONA"],
        "NL":  ["AMSTERDAM", "ROTTERDAM"],
        "SE":  ["STOCKHOLM", "GOTHENBURG"],
        "CH":  ["ZURICH", "GENEVA"],
        "AE":  ["DUBAI", "ABU DHABI"],
        "TR":  ["ISTANBUL", "ANTALYA", "ANKARA"],
        "UZ":  ["TASHKENT", "SAMARKAND", "BUKHARA"],
        "KG":  ["BISHKEK", "OSH"],
        "TH":  ["BANGKOK", "PHUKET"],
        "GE":  ["TBILISI", "BATUMI"],
        "MY":  ["KUALA LUMPUR"],
        "SGP": ["SINGAPORE"],
        "EG":  ["CAIRO", "SHARM EL SHEIKH"],
        "PL":  ["WARSAW", "KRAKOW"],
    }

    nat_codes  = list(country_city_map.keys())
    nat_weights = [0.80] + [0.20 / (len(nat_codes) - 1)] * (len(nat_codes) - 1)

    user_profiles: dict[str, dict] = {}
    for prefix in ["P", "C"]:
        for i in range(1, 5001):
            uid = f"{prefix}{i:09d}"
            home = "KZ" if random.random() < 0.9 else random.choice(nat_codes)
            user_profiles[uid] = {
                "home_country":   home,
                "avg_amount":     float(np.random.lognormal(mean=8.0, sigma=0.5)),
                "seen_countries": {home},
                "seen_merchants": set(),
            }

    base_date  = datetime(2026, 6, 9)
    num_fraud  = int(num_records * fraud_ratio)
    num_normal = num_records - num_fraud

    def _pos_mode(trx_type: str) -> str:
        if trx_type == "WWW":
            return random.choice(pos_modes_ecom)
        if trx_type == "ATM":
            return random.choice(pos_modes_atm)
        return random.choice(pos_modes_pos)

    def _product_name(nat_code: str, trx_type: str = "POS") -> str:
        pool = product_names_domestic if nat_code == "KZ" else product_names_intl
        if trx_type == "ATM":
            return pool[3]
        return random.choice(pool)

    def _merchant_name(mcc: str) -> str:
        return random.choice(MCC_MERCHANTS.get(mcc, GENERIC_MERCHANTS))

    def _trx_type_by_mcc(mcc: str) -> str:
        if mcc in {"5816", "5815", "5817", "5818", "6051", "6211"}:
            return "WWW"
        if mcc == "6011":
            return "ATM"
        return random.choices(["POS", "WWW"], weights=[0.75, 0.25], k=1)[0]

    def _build(
        user_id:       str,
        trx_dt:        datetime,
        auth_dt:       datetime,
        merchant_no:   str,
        merchant_name: str,
        city:          str,
        nat_code:      str,
        mcc:           str,
        pos_mode:      str,
        product_name:  str,
        amount:        float,
        trx_type:      str,
        is_fraud:      int = 0,
    ) -> dict:
        profile = user_profiles[user_id]
        profile["seen_countries"].add(nat_code)
        profile["seen_merchants"].add(merchant_no)
        amt = round(amount, 2)
        return {
            "MEMBER_NAME":        user_id,
            "TRANSACTION_DATE":   trx_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "AUTHORIZATION_DATE": auth_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "MERCHANT_NO":        merchant_no,
            "MERCHANT_NAME":      merchant_name,
            "CITY_NAME":          city,
            "NATIONAL_CODE":      nat_code,
            "MCC":                mcc,
            "POS_MODE":           pos_mode,
            "PRODUCT_NAME":       product_name,
            "SETTLEMENT_AMOUNT":  amt,
            "SALES_AMOUNT":       amt,
            "TRX_TYPE":           trx_type,
            "IS_FRAUD":           is_fraud,
        }

    data: list[dict] = []

    mccs_list    = list(legit_mccs_with_weights.keys())
    mccs_weights = list(legit_mccs_with_weights.values())
    hour_weights = [1,1,1,1,2,3,5,10,15,20,25,30,35,35,30,25,30,35,40,35,25,15,10,5]

    for _ in range(num_normal):
        uid     = f"{random.choice(['P','C'])}{random.randint(1, 5000):09d}"
        profile = user_profiles[uid]

        trx_dt = base_date + timedelta(
            days    = random.randint(0, 30),
            hours   = random.choices(range(24), weights=hour_weights, k=1)[0],
            minutes = random.randint(0, 59),
            seconds = random.randint(0, 59),
        )
        auth_dt  = trx_dt + timedelta(seconds=random.randint(1, 3))
        mcc      = random.choices(mccs_list, weights=mccs_weights, k=1)[0]
        trx_type = _trx_type_by_mcc(mcc)

        if trx_type == "ATM":
            amt = round(float(np.random.lognormal(mean=np.log(15_000.0), sigma=0.6)), -3)
            amt = max(1_000.0, amt)
        else:
            amt = max(100.0, float(np.random.lognormal(
                mean  = np.log(profile["avg_amount"]),
                sigma = 1.2,
            )))
            if mcc in gambling_mccs + crypto_invest_mccs:
                amt *= random.uniform(1.5, 3.0)

        nat_code = (
            profile["home_country"]
            if random.random() < 0.95
            else random.choices(nat_codes, weights=nat_weights, k=1)[0]
        )

        data.append(_build(
            uid, trx_dt, auth_dt,
            str(random.randint(1_000_000_000, 9_999_999_999)),
            _merchant_name(mcc),
            random.choice(country_city_map[nat_code]),
            nat_code, mcc, _pos_mode(trx_type), _product_name(nat_code, trx_type),
            amt, trx_type,
        ))

    fraud_patterns = [
        "F005_atypical_amount",
        "F007_card_testing",
        "F038_cashout",
        "F020_velocity_countries",
        "F006_velocity_cluster",
        "F038_atm_cashout",
    ]
    fraud_weights = [0.30, 0.20, 0.20, 0.10, 0.10, 0.10]

    generated_fraud = 0

    while generated_fraud < num_fraud:
        uid     = f"{random.choice(['P','C'])}{random.randint(1, 5000):09d}"
        profile = user_profiles[uid]
        d_off   = random.randint(0, 30)
        pattern = random.choices(fraud_patterns, weights=fraud_weights, k=1)[0]

        if pattern == "F005_atypical_amount":
            mcc    = random.choice(["5411", "6051", "7995"])
            amt    = profile["avg_amount"] * random.uniform(6.0, 10.0)
            trx_dt = base_date + timedelta(
                days=d_off, hours=random.choice([1, 2, 3, 4, 5]),
                minutes=random.randint(0, 59), seconds=random.randint(0, 59),
            )
            auth_dt      = trx_dt + timedelta(seconds=1)
            merchant_name = "BINANCE" if mcc == "6051" else "MAGNUM_VIP"
            nat_code      = profile["home_country"]
            data.append(_build(
                uid, trx_dt, auth_dt,
                str(random.randint(1_000_000_000, 9_999_999_999)),
                merchant_name,
                random.choice(country_city_map[nat_code]),
                nat_code, mcc, "0001 - E-commerce", _product_name(nat_code),
                amt, "WWW", is_fraud=1,
            ))
            generated_fraud += 1

        elif pattern == "F007_card_testing":
            mcc          = random.choice(card_testing_mccs)
            merchant_no  = str(random.randint(1_000_000_000, 9_999_999_999))
            nat_code     = "US"
            city         = random.choice(country_city_map[nat_code])
            cluster_start = base_date + timedelta(
                days=d_off, hours=random.randint(10, 22),
                minutes=random.randint(0, 59),
            )
            for j in range(random.randint(4, 7)):
                if generated_fraud >= num_fraud:
                    break
                dt = cluster_start + timedelta(seconds=j * random.randint(15, 90))
                data.append(_build(
                    uid, dt, dt + timedelta(seconds=1),
                    merchant_no, "STEAM_GAMES", city,
                    nat_code, mcc, "0001 - E-commerce", _product_name(nat_code),
                    round(random.uniform(100.0, 900.0), 2), "WWW", is_fraud=1,
                ))
                generated_fraud += 1

        elif pattern == "F038_cashout":
            mcc    = random.choice(cashout_mccs)
            amt    = random.uniform(150_000.0, 450_000.0)
            trx_dt = base_date + timedelta(
                days=d_off, hours=random.choice([23, 0, 1, 2]),
                minutes=random.randint(0, 59), seconds=random.randint(0, 59),
            )
            auth_dt  = trx_dt + timedelta(milliseconds=200)
            nat_code = profile["home_country"]
            data.append(_build(
                uid, trx_dt, auth_dt,
                str(random.randint(1_000_000_000, 9_999_999_999)),
                "P2P_TRANSFER_DROPPER",
                random.choice(country_city_map[nat_code]),
                nat_code, mcc, "1000 - Credential on file", _product_name(nat_code),
                amt, "WWW", is_fraud=1,
            ))
            generated_fraud += 1

        elif pattern == "F020_velocity_countries":
            foreign = random.choice([c for c in nat_codes if c != profile["home_country"]])
            amt     = float(np.random.lognormal(mean=8.5, sigma=0.5))
            trx_dt  = base_date + timedelta(
                days=d_off, hours=random.randint(12, 18),
                minutes=random.randint(0, 59), seconds=random.randint(0, 59),
            )
            auth_dt = trx_dt + timedelta(seconds=2)
            data.append(_build(
                uid, trx_dt, auth_dt,
                str(random.randint(1_000_000_000, 9_999_999_999)),
                "WALMART",
                random.choice(country_city_map[foreign]),
                foreign, "5411", "0110 - Swipe", _product_name(foreign),
                amt, "POS", is_fraud=1,
            ))
            generated_fraud += 1

        elif pattern == "F006_velocity_cluster":
            mcc          = random.choice(["5816", "5999"])
            nat_code     = profile["home_country"]
            city         = random.choice(country_city_map[nat_code])
            cluster_start = base_date + timedelta(
                days=d_off, hours=random.randint(0, 23),
                minutes=random.randint(0, 50),
            )
            for j in range(random.randint(5, 10)):
                if generated_fraud >= num_fraud:
                    break
                dt = cluster_start + timedelta(seconds=j * random.randint(30, 90))
                data.append(_build(
                    uid, dt, dt + timedelta(seconds=random.randint(1, 2)),
                    str(random.randint(1_000_000_000, 9_999_999_999)),
                    _merchant_name(mcc),
                    city, nat_code, mcc, "0001 - E-commerce", _product_name(nat_code),
                    round(random.uniform(500.0, 5_000.0), 2), "WWW", is_fraud=1,
                ))
                generated_fraud += 1

        elif pattern == "F038_atm_cashout":
            nat_code     = profile["home_country"]
            city         = random.choice(country_city_map[nat_code])
            cluster_start = base_date + timedelta(
                days=d_off, hours=random.choice([23, 0, 1, 2]),
                minutes=random.randint(0, 20),
            )
            for j in range(random.randint(3, 5)):
                if generated_fraud >= num_fraud:
                    break
                dt = cluster_start + timedelta(minutes=j * random.randint(5, 9))
                amt = round(float(np.random.lognormal(mean=np.log(100_000.0), sigma=0.3)), -4)
                amt = min(max(amt, 50_000.0), 300_000.0)
                data.append(_build(
                    uid, dt, dt + timedelta(seconds=random.randint(1, 3)),
                    str(random.randint(1_000_000_000, 9_999_999_999)),
                    _merchant_name("6011"),
                    city, nat_code, "6011", "0006 - Cash",
                    _product_name(nat_code, "ATM"),
                    amt, "ATM", is_fraud=1,
                ))
                generated_fraud += 1

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    OUTPUT_COLS = [
        "MEMBER_NAME", "TRANSACTION_DATE", "AUTHORIZATION_DATE",
        "MERCHANT_NO", "MERCHANT_NAME", "CITY_NAME", "NATIONAL_CODE",
        "MCC", "POS_MODE", "PRODUCT_NAME",
        "SETTLEMENT_AMOUNT", "SALES_AMOUNT",
        "TRX_TYPE", "IS_FRAUD",
    ]
    str_cols = [
        "MEMBER_NAME", "MERCHANT_NO", "MERCHANT_NAME", "CITY_NAME",
        "NATIONAL_CODE", "MCC", "POS_MODE", "PRODUCT_NAME", "TRX_TYPE",
    ]
    df[str_cols] = df[str_cols].astype(str)
    return df[OUTPUT_COLS]


if __name__ == "__main__":
    df = generate_data()
    print(f"Records: {len(df):,}  |  Fraud: {df['IS_FRAUD'].sum()} ({df['IS_FRAUD'].mean()*100:.2f}%)")
    print(f"Columns: {list(df.columns)}")

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "transactions.csv"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("transactions.csv saved to:", output_path)
