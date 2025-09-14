"""
Детальная диагностика preprocess_and_merge для поиска потери клиентов
"""

import pandas as pd
import numpy as np
from typing import Dict

def debug_preprocess_step_by_step():
    """Пошагово отслеживает preprocess_and_merge."""
    
    print("🔬 ДЕТАЛЬНАЯ ДИАГНОСТИКА PREPROCESS_AND_MERGE")
    print("=" * 50)
    
    # Загружаем данные
    dataframes = {
        'clients': pd.read_csv('../data/clients.csv'),
        'transactions': pd.read_csv('../data/transactions.csv'),
        'transfers': pd.read_csv('../data/transfers.csv')
    }
    
    df_clients = dataframes['clients'].copy()
    df_transactions = dataframes['transactions'].copy()
    df_transfers = dataframes['transfers'].copy()
    
    print(f"📊 Исходно клиентов: {len(df_clients)}")
    print(f"📊 Исходно транзакций: {len(df_transactions)}")
    print(f"📊 Исходно переводов: {len(df_transfers)}")
    
    # Конверсия дат
    if 'date' in df_transactions.columns:
        df_transactions['date'] = pd.to_datetime(df_transactions['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    if 'date' in df_transfers.columns:
        df_transfers['date'] = pd.to_datetime(df_transfers['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    print(f"📅 После конверсии дат - transactions: {len(df_transactions)}")
    print(f"📅 После конверсии дат - transfers: {len(df_transfers)}")
    
    # Создание amount_kzt
    CURRENCY_RATES = {'KZT': 1.0, 'USD': 456.0, 'EUR': 502.0, 'RUB': 4.5}
    
    if 'currency' in df_transactions.columns and 'amount' in df_transactions.columns:
        df_transactions['amount_kzt'] = df_transactions.apply(
            lambda row: row['amount'] * CURRENCY_RATES.get(row['currency'].upper(), 1.0)
            if pd.notna(row['currency']) and pd.notna(row['amount']) else 0,
            axis=1
        )
    
    if 'currency' in df_transfers.columns and 'amount' in df_transfers.columns:
        df_transfers['amount_kzt'] = df_transfers.apply(
            lambda row: row['amount'] * CURRENCY_RATES.get(row['currency'].upper(), 1.0)
            if pd.notna(row['currency']) and pd.notna(row['amount']) else 0,
            axis=1
        )
    
    # Обработка пропущенных значений - КРИТИЧНО!
    print(f"🔍 ДО обработки NaN:")
    print(f"  - Transactions с NaN date: {df_transactions['date'].isna().sum()}")
    print(f"  - Transactions с NaN category: {df_transactions['category'].isna().sum()}")
    print(f"  - Transfers с NaN date: {df_transfers['date'].isna().sum()}")
    print(f"  - Transfers с NaN type: {df_transfers['type'].isna().sum()}")
    
    df_transactions['amount_kzt'] = df_transactions['amount_kzt'].fillna(0)
    df_transfers['amount_kzt'] = df_transfers['amount_kzt'].fillna(0)
    
    # ЗДЕСЬ МОЖЕТ БЫТЬ ПОТЕРЯ!
    before_drop_trans = len(df_transactions)
    before_drop_transfers = len(df_transfers)
    
    df_transactions = df_transactions.dropna(subset=['date', 'category'], how='any')
    df_transfers = df_transfers.dropna(subset=['date', 'type'], how='any')
    
    print(f"🚨 ПОСЛЕ dropna:")
    print(f"  - Transactions: {before_drop_trans} → {len(df_transactions)} (потеря: {before_drop_trans - len(df_transactions)})")
    print(f"  - Transfers: {before_drop_transfers} → {len(df_transfers)} (потеря: {before_drop_transfers - len(df_transfers)})")
    
    # Merge step 1: clients + transactions
    before_merge1 = df_clients['client_code'].nunique()
    df_merged = pd.merge(df_clients, df_transactions, on='client_code', how='left', suffixes=('', '_trans'))
    after_merge1 = df_merged['client_code'].nunique()
    
    print(f"🔄 MERGE 1 (clients + transactions):")
    print(f"  - Клиентов: {before_merge1} → {after_merge1}")
    print(f"  - Строк после merge: {len(df_merged)}")
    
    # Merge step 2: + transfers
    df_merged = pd.merge(df_merged, df_transfers, on='client_code', how='left', suffixes=('', '_transfer'))
    after_merge2 = df_merged['client_code'].nunique()
    
    print(f"🔄 MERGE 2 (+ transfers):")
    print(f"  - Клиентов: {after_merge1} → {after_merge2}")
    print(f"  - Строк после merge: {len(df_merged)}")
    
    # Outlier removal - может быть критично!
    def remove_outliers_by_client(group):
        if len(group) < 4:
            return group
        Q1 = group['amount_kzt'].quantile(0.25)
        Q3 = group['amount_kzt'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return group[(group['amount_kzt'] >= lower_bound) & (group['amount_kzt'] <= upper_bound)]
    
    before_outliers = df_merged['client_code'].nunique()
    if len(df_merged) > 0:
        df_merged = df_merged.groupby('client_code').apply(remove_outliers_by_client).reset_index(drop=True)
    after_outliers = df_merged['client_code'].nunique()
    
    print(f"🎯 OUTLIER REMOVAL:")
    print(f"  - Клиентов: {before_outliers} → {after_outliers} (потеря: {before_outliers - after_outliers})")
    print(f"  - Строк: {len(df_merged)}")
    
    # Финальная группировка
    client_aggregated = []
    for client_code, group in df_merged.groupby('client_code'):
        client_info = group.iloc[0]
        client_record = {
            'client_code': client_code,
            'name': client_info.get('name', ''),
            'status': client_info.get('status', ''),
            'age': client_info.get('age', 0),
            'city': client_info.get('city', ''),
            'avg_monthly_balance_KZT': client_info.get('avg_monthly_balance_KZT', 0),
        }
        client_aggregated.append(client_record)
    
    final_count = len(client_aggregated)
    print(f"✅ ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: {final_count} клиентов")
    
    lost = 60 - final_count
    if lost > 0:
        print(f"🚨 ПОТЕРЯНО {lost} клиента!")
        all_codes = set(range(1, 61))
        final_codes = set([c['client_code'] for c in client_aggregated])
        missing = sorted(all_codes - final_codes)
        print(f"🚨 Отсутствующие client_code: {missing}")
    
    return final_count

if __name__ == "__main__":
    debug_preprocess_step_by_step()