"""
Debug script для диагностики потери клиентов 60→58
"""

import pandas as pd

def debug_client_pipeline():
    """Диагностирует где теряются клиенты в pipeline."""
    
    print("🔍 ДИАГНОСТИКА ПОТЕРИ КЛИЕНТОВ")
    print("=" * 40)
    
    # 1. Проверяем исходные файлы
    clients = pd.read_csv('../data/clients.csv')
    transactions = pd.read_csv('../data/transactions.csv')
    transfers = pd.read_csv('../data/transfers.csv')
    
    print(f"📁 Clients.csv: {len(clients)} записей")
    print(f"📁 Уникальных client_code: {clients['client_code'].nunique()}")
    print(f"📁 Диапазон: {clients['client_code'].min()}-{clients['client_code'].max()}")
    
    missing = set(range(1, 61)) - set(clients['client_code'])
    print(f"📁 Отсутствующие client_code: {sorted(missing)}")
    
    # 2. Проверяем клиентов в transactions и transfers
    trans_clients = set(transactions['client_code'].unique())
    transfer_clients = set(transfers['client_code'].unique())
    
    print(f"📊 Клиенты в transactions: {len(trans_clients)}")
    print(f"📊 Клиенты в transfers: {len(transfer_clients)}")
    
    # 3. Клиенты без данных
    clients_without_trans = set(clients['client_code']) - trans_clients
    clients_without_transfers = set(clients['client_code']) - transfer_clients
    
    print(f"🚨 Клиенты БЕЗ transactions: {sorted(clients_without_trans)}")
    print(f"🚨 Клиенты БЕЗ transfers: {sorted(clients_without_transfers)}")
    
    return len(clients)

if __name__ == "__main__":
    debug_client_pipeline()