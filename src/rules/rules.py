from datetime import timedelta
import pandas as pd
from src.rules.config import HIGH_RISK_MCC, GLOBAL_HIGH_AMOUNT, SMALL_AMOUNT_LIMIT

def rule_F001_new_country(tx: dict, user_history: list) -> bool:
    if not user_history:
        return False
    past_countries = {past_tx['NATIONAL_CODE'] for past_tx in user_history}
    return tx['NATIONAL_CODE'] not in past_countries


def rule_F003_high_risk_mcc(tx: dict, user_history: list) -> bool:
    return int(tx['MCC']) in HIGH_RISK_MCC


def rule_F004_high_amount(tx: dict, user_history: list) -> bool:
    return tx['SALES_AMOUNT'] > GLOBAL_HIGH_AMOUNT


def rule_F005_atypical_amount(tx: dict, user_history: list) -> bool:
    if not user_history:
        return False
    amounts = [past_tx['SALES_AMOUNT'] for past_tx in user_history]
    avg_amount = sum(amounts) / len(amounts)
    return tx['SALES_AMOUNT'] > (avg_amount * 5)


def rule_F006_high_frequency(tx: dict, user_history: list) -> bool:
    if not user_history:
        return False
    current_time = pd.to_datetime(tx['TRANSACTION_DATE'])
    ten_minutes_ago = current_time - timedelta(minutes=10)
    
    recent_count = sum(1 for past_tx in user_history 
                       if pd.to_datetime(past_tx['TRANSACTION_DATE']) >= ten_minutes_ago)
    return recent_count > 5


def rule_F016_repeated_amounts(tx: dict, user_history: list) -> bool:
    if len(user_history) < 2:
        return False
    return tx['SALES_AMOUNT'] == user_history[-1]['SALES_AMOUNT'] == user_history[-2]['SALES_AMOUNT']


def rule_F017_small_amounts_velocity(tx: dict, user_history: list) -> bool:
    if tx['SALES_AMOUNT'] >= SMALL_AMOUNT_LIMIT:
        return False
    if not user_history:
        return False
        
    small_tx_count = sum(1 for past_tx in user_history if past_tx['SALES_AMOUNT'] < SMALL_AMOUNT_LIMIT)
    return (small_tx_count + 1) >= 5


def rule_F026_multiple_cities(tx: dict, user_history: list) -> bool:
    if not user_history:
        return False
    current_time = pd.to_datetime(tx['TRANSACTION_DATE'])
    one_hour_ago = current_time - timedelta(hours=1)
    
    cities = {past_tx['CITY_NAME'] for past_tx in user_history 
              if pd.to_datetime(past_tx['TRANSACTION_DATE']) >= one_hour_ago}
    cities.add(tx['CITY_NAME'])
    return len(cities) > 2

def rule_F028_magstripe_mode(tx: dict, user_history: list) -> bool:
    return str(tx['POS_MODE']).lower() in ['02', 'magstripe', 'полоса']


def rule_F033_high_risk_mcc_new_country(tx: dict, user_history: list) -> bool:
    is_high_risk = int(tx['MCC']) in HIGH_RISK_MCC
    
    is_new_country = True
    if user_history:
        past_countries = {past_tx['NATIONAL_CODE'] for past_tx in user_history}
        is_new_country = tx['NATIONAL_CODE'] not in past_countries
        
    return is_high_risk and is_new_country
