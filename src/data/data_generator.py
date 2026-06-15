import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_data(num_records: int = 50_000,  # Вернули объем как в первой версии
                  fraud_ratio: float = 0.02,  # Установили 2% фрода под конфиг правил
                  seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)

    legit_mccs_with_weights = {
        "5411": 0.35, "5499": 0.10, "5814": 0.15, "4121": 0.12,
        "5331": 0.08, "5812": 0.05, "5912": 0.04, "5541": 0.03,
        "5691": 0.02, "5977": 0.01, "7995": 0.01, "6051": 0.01,
        "5816": 0.02, "5999": 0.01, "6011": 0.05,
    }

    gambling_mccs = ["7995", "7800", "7801", "7802"]
    crypto_invest_mccs = ["6051", "6211", "6282", "6529"]
    card_testing_mccs = ["5815", "5816", "5817", "5818", "5734", "5942", "5999"]
    cashout_mccs = ["4829", "6536", "6537", "6540", "6012"]  # Добавили 6012 для Правила 1

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
    GENERIC_MERCHANTS = ["WALMART", "AMAZON", "TARGET", "COSTCO"]

    pos_modes_ecom = ["0001 - E-commerce", "0002 - Mail/Phone order", "0003 - Recurring", "1000 - Credential on file"]
    pos_modes_pos = ["0710 - Contactless - VSDC chip", "0730 - Contactless - VSDC chip", "0110 - Swipe", "0120 - Keyin"]
    pos_modes_atm = ["0006 - Cash", "0110 - Swipe"]

    product_names_domestic = [
        "1308112001: Debit Purchase Transaction(Domestic)",
        "1308132001: Credit Purchase Transaction(Domestic)",
        "1308172001: Cash Withdrawal Transaction(Domestic)",
    ]
    product_names_intl = [
        "1308122001: Debit Purchase Transaction(International)",
        "1308142001: Credit Purchase Transaction(International)",
        "1308182001: Cash Withdrawal Transaction(International)",
    ]

    country_city_map = {
        "KZ": ["ALMATY", "ASTANA", "AKTAU", "ATYRAU", "KARAGANDA", "SHYMKENT"],
        "RU": ["MOSCOW", "SAINT PETERSBURG"],
        "US": ["NEW YORK", "LOS ANGELES", "MIAMI"],
        "CN": ["BEIJING", "SHANGHAI"],
        "GB": ["LONDON", "MANCHESTER"],
        "AE": ["DUBAI", "ABU DHABI"],
        "TR": ["ISTANBUL", "ANTALYA"],
    }

    nat_codes = list(country_city_map.keys())
    nat_weights = [0.85] + [0.15 / (len(nat_codes) - 1)] * (len(nat_codes) - 1)

    # Инициализация профилей
    user_profiles: dict[str, dict] = {}
    for prefix in ["P", "C"]:
        for i in range(1, 2001):  # Сделали 2000 клиентов для плотности транзакций
            uid = f"{prefix}{i:06d}"
            home = "KZ" if random.random() < 0.92 else random.choice(nat_codes)
            user_profiles[uid] = {
                "home_country": home,
                "avg_amount": float(np.random.lognormal(mean=8.5, sigma=0.6)),
                "seen_countries": {home},
            }

    base_date = datetime(2026, 6, 9)
    num_fraud = int(num_records * fraud_ratio)
    num_normal = num_records - num_fraud

    def _pos_mode(trx_type: str) -> str:
        if trx_type == "WWW": return random.choice(pos_modes_ecom)
        if trx_type == "ATM": return random.choice(pos_modes_atm)
        return random.choice(pos_modes_pos)

    def _product_name(nat_code: str, trx_type: str = "POS") -> str:
        pool = product_names_domestic if nat_code == "KZ" else product_names_intl
        if trx_type == "ATM": return pool[2]
        return random.choice(pool[:2])

    def _merchant_name(mcc: str) -> str:
        return random.choice(MCC_MERCHANTS.get(mcc, GENERIC_MERCHANTS))

    def _trx_type_by_mcc(mcc: str) -> str:
        if mcc in {"5816", "6051", "5999"}: return "WWW"
        if mcc == "6011": return "ATM"
        return random.choices(["POS", "WWW"], weights=[0.80, 0.20], k=1)[0]

    def _build(user_id: str, trx_dt: datetime, auth_dt: datetime, merchant_no: str,
               merchant_name: str, city: str, nat_code: str, mcc: str, pos_mode: str,
               product_name: str, amount: float, trx_type: str, is_fraud: int = 0) -> dict:
        amt = round(amount, 2)
        return {
            "MEMBER_NO": user_id,  # Переименовали под логику правил
            "TRANSACTION_DATE": trx_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "AUTHORIZATION_DATE": auth_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "MERCHANT_NO": merchant_no,
            "MERCHANT_NAME": merchant_name,
            "CITY_NAME": city,
            "NATIONAL_CODE": nat_code,
            "MCC": str(mcc),
            "POS_MODE": pos_mode,
            "PRODUCT_NAME": product_name,
            "SETTLEMENT_AMOUNT": amt,
            "SALES_AMOUNT": amt,
            "TRX_TYPE": trx_type,
            "IS_FRAUD": is_fraud,
        }

    data: list[dict] = []
    mccs_list = list(legit_mccs_with_weights.keys())
    mccs_weights = list(legit_mccs_with_weights.values())
    hour_weights = [1, 1, 1, 1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 35, 30, 25, 30, 35, 40, 35, 25, 15, 10, 5]
    _h = list(range(24))

    fraud_hour_any_time = [1] * 24
    fraud_hour_night_biased = [10, 10, 10, 10, 10, 8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 10]

    # --- ГЕНЕРАЦИЯ ЛЕГИТИМНЫХ ТРАНЗАКЦИЙ ---
    print("Генерация легитимных транзакций...")
    for _ in range(num_normal):
        uid = f"{random.choice(['P', 'C'])}{random.randint(1, 2000):06d}"
        profile = user_profiles[uid]

        trx_dt = base_date + timedelta(
            days=random.randint(0, 6),  # сузили до 7 дней как в правилах
            hours=random.choices(range(24), weights=hour_weights, k=1)[0],
            minutes=random.randint(0, 59), seconds=random.randint(0, 59),
        )
        auth_dt = trx_dt + timedelta(seconds=random.randint(1, 3))
        mcc = random.choices(mccs_list, weights=mccs_weights, k=1)[0]
        trx_type = _trx_type_by_mcc(mcc)

        if trx_type == "ATM":
            amt = round(float(np.random.lognormal(mean=np.log(20_000.0), sigma=0.5)), -3)
            amt = min(max(2000.0, amt), 250_000.0)  # Ограничили, чтобы не пересекалось с фродом в банкоматах
        else:
            amt = max(150.0, float(np.random.lognormal(mean=np.log(profile["avg_amount"]), sigma=0.8)))
            if mcc in gambling_mccs + crypto_invest_mccs:
                amt *= random.uniform(1.1, 1.8)

        nat_code = profile["home_country"] if random.random() < 0.96 else \
        random.choices(nat_codes, weights=nat_weights, k=1)[0]

        data.append(_build(
            uid, trx_dt, auth_dt,
            str(random.randint(1_000_000_000, 9_999_999_999)),
            _merchant_name(mcc), random.choice(country_city_map[nat_code]),
            nat_code, mcc, _pos_mode(trx_type), _product_name(nat_code, trx_type),
            amt, trx_type,
        ))

    # --- ГЕНЕРАЦИЯ ФРОДОВЫХ ПАТТЕРНОВ ---
    print("Генерация фродовых паттернов под бизнес-правила...")
    fraud_patterns = [
        "F005_atypical_amount", "F007_card_testing", "F038_cashout",
        "F020_velocity_countries", "F006_velocity_cluster", "F038_atm_cashout"
    ]
    fraud_weights = [0.25, 0.15, 0.20, 0.15, 0.10, 0.15]
    generated_fraud = 0

    while generated_fraud < num_fraud:
        uid = f"{random.choice(['P', 'C'])}{random.randint(1, 2000):06d}"
        profile = user_profiles[uid]
        d_off = random.randint(0, 6)
        pattern = random.choices(fraud_patterns, weights=fraud_weights, k=1)[0]

        # Паттерн: Крупный чек (Триггерит Правило 1, 4 или 11)
        if pattern == "F005_atypical_amount":
            mcc = random.choice(["6012", "6051", "7995", "5411"])
            # Ставим крупную сумму для детекции правил
            amt = random.uniform(410_000.0, 550_000.0) if mcc in ["6012", "6051", "7995"] else random.uniform(60_000.0,
                                                                                                              120_000.0)

            # Смещение на ночь для Правила 4
            hour = random.choices(_h, weights=fraud_hour_night_biased, k=1)[0]
            trx_dt = base_date + timedelta(days=d_off, hours=hour, minutes=random.randint(0, 59),
                                           seconds=random.randint(0, 59))

            nat_code = profile["home_country"]
            data.append(_build(
                uid, trx_dt, trx_dt + timedelta(seconds=1),
                str(random.randint(1_000_000_000, 9_999_999_999)),
                "HIGH_RISK_MERCHANT" if mcc != "5411" else "MAGNUM_VIP",
                random.choice(country_city_map[nat_code]),
                nat_code, mcc, "0001 - E-commerce", _product_name(nat_code),
                amt, "WWW", is_fraud=1
            ))
            generated_fraud += 1

        # Паттерн: Card Testing (Триггерит Правило 3 - много мелких транзакций)
        elif pattern == "F007_card_testing":
            mcc = "5816"
            merchant_no = str(random.randint(1_000_000_000, 9_999_999_999))
            nat_code = profile["home_country"]
            cluster_start = base_date + timedelta(days=d_off, hours=random.randint(9, 21),
                                                  minutes=random.randint(0, 20))

            for j in range(random.randint(5, 7)):  # Строго от 5 транзакций
                if generated_fraud >= num_fraud: break
                dt = cluster_start + timedelta(minutes=j * random.randint(1, 4))  # в пределах 30 минут
                data.append(_build(
                    uid, dt, dt + timedelta(seconds=1),
                    merchant_no, "STEAM_GAMES", random.choice(country_city_map[nat_code]),
                    nat_code, mcc, "0001 - E-commerce", _product_name(nat_code),
                    random.uniform(500.0, 1500.0), "WWW", is_fraud=1  # Сумма < 2000
                ))
                generated_fraud += 1

        # Паттерн: Ручной ввод на крупные суммы (Триггерит Правило 7)
        elif pattern == "F038_cashout":
            mcc = random.choice(cashout_mccs)
            amt = random.uniform(210_000.0, 350_000.0)  # Сумма > 200к для правила 7
            trx_dt = base_date + timedelta(days=d_off, hours=random.randint(9, 18), minutes=random.randint(0, 59))
            nat_code = profile["home_country"]

            data.append(_build(
                uid, trx_dt, trx_dt + timedelta(seconds=2),
                str(random.randint(1_000_000_000, 9_999_999_999)),
                "MANUAL_CASH_OUT", random.choice(country_city_map[nat_code]),
                nat_code, mcc, "0120 - Keyin", _product_name(nat_code),  # Подставили Keyin!
                amt, "POS", is_fraud=1
            ))
            generated_fraud += 1

        # Паттерн: Скорость по странам (Триггерит Правило 10 и Правило 6/8)
        elif pattern == "F020_velocity_countries":
            if profile["home_country"] != "KZ": continue  # Генерируем только для резидентов KZ

            # Сначала делаем чистую легитимную транзакцию в KZ
            t_start = base_date + timedelta(days=d_off, hours=random.randint(10, 20), minutes=random.randint(0, 10))
            data.append(_build(
                uid, t_start, t_start + timedelta(seconds=2),
                str(random.randint(1_000_000_000, 9_999_999_999)),
                "KFC_ALMATY", "ALMATY", "KZ", "5814", "0710 - Contactless - VSDC chip",
                _product_name("KZ"), random.uniform(2000, 5000), "POS", is_fraud=0
            ))

            # Через 15 минут прилетает фрод из-за рубежа (физически невозможно)
            t_fraud = t_start + timedelta(minutes=15)
            foreign_country = random.choice([c for c in nat_codes if c != "KZ"])

            data.append(_build(
                uid, t_fraud, t_fraud + timedelta(seconds=1),
                str(random.randint(1_000_000_000, 9_999_999_999)),
                "FOREIGN_STORE", random.choice(country_city_map[foreign_country]),
                foreign_country, "5411", "0110 - Swipe", _product_name(foreign_country),
                random.uniform(10000, 50000), "POS", is_fraud=1
            ))
            generated_fraud += 1

        # Паттерн: Дроблинг и одинаковые суммы (Триггерит Правило 5 и Правило 9)
        elif pattern == "F006_velocity_cluster":
            mcc = "5999"
            nat_code = profile["home_country"]
            merchant_no = str(random.randint(1_000_000_000, 9_999_999_999))
            cluster_start = base_date + timedelta(days=d_off, hours=random.randint(8, 22),
                                                  minutes=random.randint(0, 30))
            fixed_amount = random.uniform(5000, 25000)  # Одинаковая сумма для правила 9!

            for j in range(3):  # Повторяем 3 раза за день на одного мерчанта (Правило 5)
                if generated_fraud >= num_fraud: break
                dt = cluster_start + timedelta(minutes=j * random.randint(5, 15))
                data.append(_build(
                    uid, dt, dt + timedelta(seconds=1),
                    merchant_no, "AMAZON_ATTACK", random.choice(country_city_map[nat_code]),
                    nat_code, mcc, "0001 - E-commerce", _product_name(nat_code),
                    fixed_amount, "WWW", is_fraud=1
                ))
                generated_fraud += 1

        # Паттерн: Серийный вынос банкомата (Триггерит Правило 2)
        elif pattern == "F038_atm_cashout":
            nat_code = profile["home_country"]
            city = random.choice(country_city_map[nat_code])
            merchant_no = str(random.randint(1_000_000_000, 9_999_999_999))
            cluster_start = base_date + timedelta(days=d_off, hours=random.randint(12, 23),
                                                  minutes=random.randint(0, 15))

            for j in range(2):  # 2 повторения крупных снятий за день
                if generated_fraud >= num_fraud: break
                dt = cluster_start + timedelta(minutes=j * random.randint(4, 10))
                # Ставим сумму строго > 300 000 тг (Правило 2)
                atm_amount = random.uniform(310_000.0, 380_000.0)
                data.append(_build(
                    uid, dt, dt + timedelta(seconds=3),
                    merchant_no, "ATM_FRAUD_LOCK", city,
                    nat_code, "6011", "0006 - Cash", _product_name(nat_code, "ATM"),
                    atm_amount, "ATM", is_fraud=1
                ))
                generated_fraud += 1

    # --- СБОРКА И ИТОГОВОЕ СОРТИРОВАНИЕ ---
    df = pd.DataFrame(data)
    df['TRANSACTION_DATE'] = pd.to_datetime(df['TRANSACTION_DATE'])
    df = df.sort_values("TRANSACTION_DATE").reset_index(drop=True)

    # Добавляем TRX_ID в самый первый столбец, чтобы не падали правила
    df.insert(0, "TRX_ID", [f"T{str(j).zfill(8)}" for j in range(len(df))])

    # Переводим обратно даты в строковый формат для сохранения в CSV
    df['TRANSACTION_DATE'] = df['TRANSACTION_DATE'].dt.strftime("%Y-%m-%d %H:%M:%S")

    OUTPUT_COLS = [
        "TRX_ID", "MEMBER_NO", "TRANSACTION_DATE", "AUTHORIZATION_DATE",
        "MERCHANT_NO", "MERCHANT_NAME", "CITY_NAME", "NATIONAL_CODE",
        "MCC", "POS_MODE", "PRODUCT_NAME", "SETTLEMENT_AMOUNT", "SALES_AMOUNT",
        "TRX_TYPE", "IS_FRAUD"
    ]
    return df[OUTPUT_COLS]


if __name__ == "__main__":
    df = generate_data()
    print(f"📊 Итого записей: {len(df):,} | Фрод: {df['IS_FRAUD'].sum()} ({df['IS_FRAUD'].mean() * 100:.2f}%)")

    # Путь сохранения в соответствии с архитектурой репозитория
    output_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "transactions_labeled.csv"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("✅ Файл сохранен в: ", output_path)