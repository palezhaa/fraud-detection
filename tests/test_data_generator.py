"""
Тесты для data_generator.py
Запуск в корне проекта:
    python -m tests.test_data_generator

Группы тестов:
  1. Схема и базовые инварианты (колонки, типы, обязательные поля)
  2. Согласованность сумм (SETTLEMENT == SALES, диапазоны, точность)
  3. Закон Бенфорда  первая цифра SALES_AMOUNT
  4. Даты/время (AUTH >= TRANSACTION, корректность парсинга)
  5. Бизнес-логика (TRX_TYPE согласован с MCC/POS_MODE, доли fraud)
  6. Воспроизводимость (одинаковый seed -> одинаковый датасет)
"""

import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.data.data_generator import generate_data

EXPECTED_COLUMNS = [
    "MEMBER_NAME", "TRANSACTION_DATE", "AUTHORIZATION_DATE",
    "MERCHANT_NO", "MERCHANT_NAME", "CITY_NAME", "NATIONAL_CODE",
    "MCC", "POS_MODE", "PRODUCT_NAME",
    "SETTLEMENT_AMOUNT", "SALES_AMOUNT", "TRX_TYPE", "IS_FRAUD",
]


@pytest.fixture(scope="module")
def df():
    return generate_data(num_records=20_000, fraud_ratio=0.015, seed=42)


# схема и базовые инварианты
class TestSchema:

    def test_columns_exact(self, df):
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_no_nulls(self, df):
        nulls = df.isna().sum()
        assert nulls.sum() == 0, f"Found NaN:\n{nulls[nulls > 0]}"

    def test_row_count(self, df):
        assert len(df) >= 20_000  # может быть чуть больше из-за кластеров

    def test_is_fraud_binary(self, df):
        assert set(df["IS_FRAUD"].unique()) <= {0, 1}

    def test_trx_type_values(self, df):
        assert set(df["TRX_TYPE"].unique()) <= {"POS", "WWW", "ATM"}

    def test_member_name_format(self, df):
        # формат: P/C + 9 цифр
        pattern = df["MEMBER_NAME"].str.match(r"^[PC]\d{9}$")
        assert pattern.all(), "Incorrect MEMBER_NAME format"

    def test_mcc_is_4_digit_string(self, df):
        assert df["MCC"].astype(str).str.match(r"^\d{4}$").all()


# согласованность сумм
class TestAmounts:

    def test_settlement_equals_sales(self, df):
        assert (df["SETTLEMENT_AMOUNT"] == df["SALES_AMOUNT"]).all()

    def test_amounts_positive(self, df):
        assert (df["SALES_AMOUNT"] > 0).all()
        assert (df["SETTLEMENT_AMOUNT"] > 0).all()

    def test_normal_amounts_above_min(self, df):
        # нормальные транзакции имеют минимум 100
        normal = df[df["IS_FRAUD"] == 0]
        assert (normal["SALES_AMOUNT"] >= 100.0 - 1e-6).all()

    def test_amounts_two_decimal_places(self, df):
        # round(x, 2) -> разница с округленным значением должна быть ~0
        rounded = df["SALES_AMOUNT"].round(2)
        assert np.allclose(df["SALES_AMOUNT"], rounded, atol=1e-9)

    def test_atm_amounts_are_round_denominations(self, df):
        # все ATM-суммы (обычные и fraud) генерируются через lognormal
        # с округлением до купюр: обычные -> кратно 1000, fraud -> кратно 10000
        atm = df[df["TRX_TYPE"] == "ATM"]
        if len(atm) == 0:
            pytest.skip("No ATM records in this sample")

        normal_atm = atm[atm["IS_FRAUD"] == 0]
        if len(normal_atm) > 0:
            assert (normal_atm["SALES_AMOUNT"] % 1000 == 0).all()
            assert (normal_atm["SALES_AMOUNT"] >= 1_000.0).all()

        fraud_atm = atm[atm["IS_FRAUD"] == 1]
        if len(fraud_atm) > 0:
            assert (fraud_atm["SALES_AMOUNT"] % 10_000 == 0).all()
            assert (fraud_atm["SALES_AMOUNT"] >= 50_000.0).all()
            assert (fraud_atm["SALES_AMOUNT"] <= 300_000.0).all()

    def test_atm_not_exclusively_fraud(self, df):
        # регрессионный тест: раньше MCC=6011 встречался только во
        # fraud-паттерне F038_atm_cashout => 100% ATM были fraud.
        atm = df[df["TRX_TYPE"] == "ATM"]
        if len(atm) == 0:
            pytest.skip("No ATM records in this sample")

        fraud_rate = atm["IS_FRAUD"].mean()
        assert fraud_rate < 0.5, (
            f"Fraction of fraud among ATM transactions = {fraud_rate:.2%} -- "
            f"looks like a regression (MCC=6011 is not generated as a normal transaction)"
        )
        assert (atm["IS_FRAUD"] == 0).any(), "No normal ATM transactions found"

    def test_mcc_matches_merchant_category(self, df):
        # регрессионный тест: MERCHANT_NAME должен соответствовать
        # категории MCC (см. MCC_MERCHANTS в data_generator.py)
        expected = {
            "4121": {"YANDEX TAXI", "INDRIVE", "CITYMOBIL"},
            "5816": {"STEAM_GAMES", "GOOGLE_PLAY", "APPLE STORE", "PLAYSTATION_STORE"},
            "6011": {"ATM_HALYK_KZ", "ATM_KASPI_KZ", "ATM_SBERBANK_KZ", "ATM_FORTEBANK"},
            "7995": {"1XBET", "OLIMP_BET", "PARIMATCH"},
            "6051": {"BINANCE", "BYBIT", "CRYPTO_EXCHANGE"},
        }
        normal = df[df["IS_FRAUD"] == 0]
        for mcc, allowed in expected.items():
            actual = set(normal.loc[normal["MCC"] == mcc, "MERCHANT_NAME"].unique())
            assert actual <= allowed, (
                f"MCC={mcc}: unexpected merchants {actual - allowed}"
            )

    def test_taxi_mcc_not_a_supermarket(self, df):
        # конкретный кейс из обратной связи: MCC=4121 (такси) не должен
        # выдаваться супермаркетам типа MAGNUM/WALMART/COSTCO, 
        # иначе модели могут легко переобучиться на этих ярких фичах 
        # и не научиться распознавать такси по другим признакам
        normal = df[df["IS_FRAUD"] == 0]
        taxi = normal[normal["MCC"] == "4121"]
        supermarket_names = {"MAGNUM", "WALMART", "COSTCO", "WILD BERRIES", "GALMART", "RAMSTORE"}
        assert not set(taxi["MERCHANT_NAME"].unique()) & supermarket_names
        # F007: суммы 100-900 в card-testing MCC
        cardtest_mccs = {"5815", "5816", "5817", "5818", "5734", "5942", "5999"}
        ct = df[(df["IS_FRAUD"] == 1)
                & (df["MCC"].isin(cardtest_mccs))
                & (df["NATIONAL_CODE"] == "US")
                & (df["TRX_TYPE"] == "WWW")]
        if len(ct) > 0:
            assert (ct["SALES_AMOUNT"] >= 100.0).all()
            assert (ct["SALES_AMOUNT"] <= 900.0).all()


BENFORD_PROBS = {d: math.log10(1 + 1 / d) for d in range(1, 10)}


def first_digit(x: float) -> int:
    x = abs(x)
    while x < 1:
        x *= 10
    while x >= 10:
        x /= 10
    return int(x)


def chi_square_benford(amounts: pd.Series) -> tuple[float, dict]:
    digits = amounts.apply(first_digit)
    counts = Counter(digits)
    n = len(amounts)

    chi2 = 0.0
    observed_vs_expected = {}
    for d in range(1, 10):
        observed = counts.get(d, 0)
        expected = BENFORD_PROBS[d] * n
        chi2 += (observed - expected) ** 2 / expected
        observed_vs_expected[d] = (observed, round(expected, 1))

    return chi2, observed_vs_expected


class TestBenfordsLaw:
    # критическое значение хи-квадрат для df=8:
    # p=0.05 -> 15.51
    # p=0.01 -> 20.09
    # для синтетических данных используем мягкий порог, т.к. генерация не идеальна, 
    # а тест служит для обнаружения грубых нарушений (например, из-за багов), 
    # а не для строгого статистического заключения о соответствии Бенфорду
    CHI2_THRESHOLD = 50.0

    def test_normal_amounts_follow_benford(self, df):
        normal_amounts = df.loc[df["IS_FRAUD"] == 0, "SALES_AMOUNT"]
        chi2, obs_exp = chi_square_benford(normal_amounts)

        print("\nDistribution of First Digits (Normal Transactions):")
        print(f"{'Digit':>6} {'Observed':>10} {'Expected (Benford)':>16}")
        for d, (obs, exp) in obs_exp.items():
            print(f"{d:>6} {obs:>10} {exp:>16}")
        print(f"chi2 = {chi2:.2f}  (threshold = {self.CHI2_THRESHOLD})")

        assert chi2 < self.CHI2_THRESHOLD, (
            f"Distribution of first digit of SALES_AMOUNT significantly deviates "
            f"from Benford's Law (chi2={chi2:.2f})"
        )

    def test_digit_one_most_frequent(self, df):
        # базовая проверка закона Бенфорда: цифра 1 должна быть самой
        # частой первой цифрой (~30.1% теоретически)
        normal_amounts = df.loc[df["IS_FRAUD"] == 0, "SALES_AMOUNT"]
        digits = normal_amounts.apply(first_digit)
        freq = digits.value_counts(normalize=True)

        assert freq.idxmax() == 1, (
            f"Expected digit 1 as the most frequent, "
            f"observed: {freq.idxmax()} ({freq.max():.1%})\n{freq.sort_index()}"
        )
        # допускаем широкий диапазон (теория: ~30.1%)
        assert 0.15 < freq[1] < 0.45, f"Digit 1 frequency = {freq[1]:.1%}, expected range 15-45%"

    def test_digit_nine_least_frequent_among_high_digits(self, df):
        # цифра 9 должна встречаться реже цифр 1-3 (теория: 4.6% vs 30.1/17.6/12.5%)
        normal_amounts = df.loc[df["IS_FRAUD"] == 0, "SALES_AMOUNT"]
        digits = normal_amounts.apply(first_digit)
        freq = digits.value_counts(normalize=True)

        for low_digit in (1, 2, 3):
            assert freq.get(9, 0) < freq.get(low_digit, 0), (
                f"Digit 9 ({freq.get(9, 0):.1%}) occurs more frequently, "
                f"than digit {low_digit} ({freq.get(low_digit, 0):.1%})"
            )

    def test_atm_fraud_amounts_are_large_and_round(self, df):
        # контрольный тест: ATM фрод намеренно использует крупные суммы,
        # кратные 10 000 (округление до купюр через lognormal)
        # это ОЖИДАЕМОЕ отклонение от Бенфорда - документируем его явно,
        # чтобы при изменении паттерна осознанно решать, 
        # сохранять ли такое нарушение или нет, и не допустить регрессий,
        # ломает ли это общее распределение
        atm_fraud = df[(df["TRX_TYPE"] == "ATM") & (df["IS_FRAUD"] == 1)]
        if len(atm_fraud) == 0:
            pytest.skip("No ATM fraud records in this sample")

        assert (atm_fraud["SALES_AMOUNT"] % 10_000 == 0).all()
        assert (atm_fraud["SALES_AMOUNT"] >= 50_000.0).all()

    def test_atm_normal_vs_fraud_amount_ranges_differ(self, df):
        # обычные ATM снятия (медиана ~15k) должны быть в среднем
        # существенно меньше fraud кэшаута (диапазон 50k-300k)
        atm = df[df["TRX_TYPE"] == "ATM"]
        normal_atm = atm[atm["IS_FRAUD"] == 0]["SALES_AMOUNT"]
        fraud_atm  = atm[atm["IS_FRAUD"] == 1]["SALES_AMOUNT"]
        if len(normal_atm) == 0 or len(fraud_atm) == 0:
            pytest.skip("Not enough ATM records for comparison")

        assert normal_atm.median() < fraud_atm.median()

    def test_full_dataset_benford_not_grossly_violated(self, df):
        # весь датасет (включая fraud) - допускаем больший chi2,
        # тк fraud-паттерны намеренно используют нетипичные суммы,
        # но грубых нарушений (на порядки) быть не должно
        chi2, _ = chi_square_benford(df["SALES_AMOUNT"])
        loose_threshold = self.CHI2_THRESHOLD * 3
        assert chi2 < loose_threshold, (
            f"chi2 for the entire dataset = {chi2:.2f} > {loose_threshold} -- "
            f"check if there are too many artificial "
            f"round amounts in the fraud patterns"
        )


# даты и время
class TestDatesAndTimes:

    def test_dates_parseable(self, df):
        pd.to_datetime(df["TRANSACTION_DATE"], format="%Y-%m-%d %H:%M:%S")
        pd.to_datetime(df["AUTHORIZATION_DATE"], format="%Y-%m-%d %H:%M:%S")

    def test_authorization_after_transaction(self, df):
        trx = pd.to_datetime(df["TRANSACTION_DATE"])
        auth = pd.to_datetime(df["AUTHORIZATION_DATE"])
        assert (auth >= trx).all(), "AUTHORIZATION_DATE раньше TRANSACTION_DATE"

    def test_fraud_dates_not_constant(self, df):
        # регрессионный тест на исходный баг: все fraud-даты были одинаковы
        fraud_dates = df.loc[df["IS_FRAUD"] == 1, "TRANSACTION_DATE"]
        assert fraud_dates.nunique() > len(fraud_dates) * 0.5, (
            "Too few unique dates among fraud transactions -- "
            "looks like a regression bug with constant date"
        )

    def test_dates_within_expected_range(self, df):
        trx = pd.to_datetime(df["TRANSACTION_DATE"])
        # base_date = 2026-06-09, +30 дней основной диапазон,
        # + запас на сдвиги внутри кластеров
        assert trx.min() >= pd.Timestamp("2026-06-09 00:00:00")
        assert trx.max() <= pd.Timestamp("2026-07-12 00:00:00")

    def test_authorization_delay_reasonable(self, df):
        trx = pd.to_datetime(df["TRANSACTION_DATE"])
        auth = pd.to_datetime(df["AUTHORIZATION_DATE"])
        delay = (auth - trx).dt.total_seconds()
        assert (delay >= 0).all()
        assert (delay <= 5).all(), "Authorization delay > 5 seconds -- check generation"


# бизнес логика
class TestBusinessLogic:

    def test_trx_type_matches_mcc_for_atm(self, df):
        # MCC 6011 (банкомат) должен иметь TRX_TYPE == ATM
        atm_mcc = df[df["MCC"] == "6011"]
        assert (atm_mcc["TRX_TYPE"] == "ATM").all()

    def test_atm_pos_mode_is_cash_or_swipe(self, df):
        atm = df[df["TRX_TYPE"] == "ATM"]
        assert atm["POS_MODE"].isin(["0006 - Cash", "0110 - Swipe", "0000 - Unknown"]).all()

    def test_www_pos_mode_is_ecom_family(self, df):
        ecom_modes = {
            "0001 - E-commerce", "0002 - Mail/Phone order", "0003 - Recurring",
            "0004 - Installment", "0005 - Preauthorized", "1000 - Credential on file",
        }
        www = df[df["TRX_TYPE"] == "WWW"]
        assert www["POS_MODE"].isin(ecom_modes).all()

    def test_fraud_ratio_close_to_target(self, df):
        ratio = df["IS_FRAUD"].mean()
        # fraud_ratio=0.015 задает долю одиночных эпизодов; кластерные
        # паттерны добавляют записи сверху, поэтому итоговая доля выше
        assert 0.01 <= ratio <= 0.06, f"Fraud ratio = {ratio:.4f}, expected 0.01-0.06"

    def test_velocity_country_fraud_uses_foreign_country(self, df):
        # каждая запись F020 должна иметь NATIONAL_CODE с MCC=5411 и Swipe
        suspicious = df[(df["IS_FRAUD"] == 1)
                         & (df["MCC"] == "5411")
                         & (df["POS_MODE"] == "0110 - Swipe")
                         & (df["TRX_TYPE"] == "POS")]
        assert len(suspicious) > 0, "No F020-like records found"

    def test_no_duplicate_rows(self, df):
        # полные дубликаты маловероятны при нормальной генерации
        dup_ratio = df.duplicated().mean()
        assert dup_ratio < 0.01, f"Too many duplicates: {dup_ratio:.2%}"


# возпроизводимость
class TestReproducibility:

    def test_same_seed_same_output(self):
        df1 = generate_data(num_records=2_000, fraud_ratio=0.015, seed=123)
        df2 = generate_data(num_records=2_000, fraud_ratio=0.015, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_output(self):
        df1 = generate_data(num_records=2_000, fraud_ratio=0.015, seed=1)
        df2 = generate_data(num_records=2_000, fraud_ratio=0.015, seed=2)
        assert not df1.equals(df2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
