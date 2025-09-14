"""
Простое добавление 2 клиентов для достижения 60 согласно ТЗ.
"""

import pandas as pd
import numpy as np

def add_missing_clients():
    """Добавляет клиентов 34 и 45 для достижения 60 клиентов."""
    
    print("🔧 Добавляем клиентов 34 и 45...")
    
    # Загружаем существующие данные
    clients = pd.read_csv('../data/clients.csv')
    transactions = pd.read_csv('../data/transactions.csv')
    transfers = pd.read_csv('../data/transfers.csv')
    
    # Клиент 34: FX активный (для обмена валют)
    client_34 = pd.DataFrame([{
        'client_code': 34,
        'name': 'Айдар',
        'status': 'Премиальный клиент',
        'age': 35,
        'city': 'Алматы',
        'avg_monthly_balance_KZT': 800000
    }])
    
    # Клиент 45: Travel активный (для карты путешествий)
    client_45 = pd.DataFrame([{
        'client_code': 45,
        'name': 'Диана',
        'status': 'Зарплатный клиент',
        'age': 27,
        'city': 'Астана',
        'avg_monthly_balance_KZT': 400000
    }])
    
    # Простые транзакции для клиента 34 (FX профиль)
    trans_34 = []
    for i in range(50):
        trans_34.append({
            'client_code': 34,
            'name': 'Айдар',
            'product': 'Обмен валют',
            'status': 'вип',
            'city': 'Алматы',
            'date': '2025-06-15',
            'category': np.random.choice(['Кафе и рестораны', 'Такси', 'АЗС']),
            'amount': np.random.uniform(5000, 25000),
            'currency': 'KZT'
        })
    
    # Простые транзакции для клиента 45 (Travel профиль)
    trans_45 = []
    for i in range(50):
        trans_45.append({
            'client_code': 45,
            'name': 'Диана',
            'product': 'Карта для путешествий',
            'status': 'зп',
            'city': 'Астана',
            'date': '2025-06-15',
            'category': np.random.choice(['Такси', 'Путешествия', 'Отели', 'Кафе и рестораны']),
            'amount': np.random.uniform(3000, 15000),
            'currency': 'KZT'
        })
    
    # Переводы для клиента 34 (много FX)
    transfers_34 = []
    for i in range(30):
        if i < 10:  # FX операции
            transfers_34.append({
                'client_code': 34,
                'date': '2025-06-15',
                'type': np.random.choice(['fx_buy', 'fx_sell']),
                'direction': np.random.choice(['in', 'out']),
                'amount': np.random.uniform(10000, 50000),
                'currency': 'USD'
            })
        else:  # Обычные операции
            transfers_34.append({
                'client_code': 34,
                'date': '2025-06-15',
                'type': np.random.choice(['card_out', 'salary_in', 'p2p_out']),
                'direction': np.random.choice(['in', 'out']),
                'amount': np.random.uniform(5000, 30000),
                'currency': 'KZT'
            })
    
    # Переводы для клиента 45 (стандартные + немного FX)
    transfers_45 = []
    for i in range(30):
        if i < 3:  # Немного FX для travel
            transfers_45.append({
                'client_code': 45,
                'date': '2025-06-15',
                'type': 'fx_buy',
                'direction': 'out',
                'amount': np.random.uniform(5000, 15000),
                'currency': 'USD'
            })
        else:  # Обычные операции
            transfers_45.append({
                'client_code': 45,
                'date': '2025-06-15',
                'type': np.random.choice(['card_out', 'salary_in', 'atm_withdrawal']),
                'direction': np.random.choice(['in', 'out']),
                'amount': np.random.uniform(3000, 20000),
                'currency': 'KZT'
            })
    
    # Объединяем данные
    new_clients = pd.concat([clients, client_34, client_45], ignore_index=True)
    new_transactions = pd.concat([transactions, pd.DataFrame(trans_34), pd.DataFrame(trans_45)], ignore_index=True)
    new_transfers = pd.concat([transfers, pd.DataFrame(transfers_34), pd.DataFrame(transfers_45)], ignore_index=True)
    
    # Сохраняем
    new_clients.to_csv('../data/clients.csv', index=False)
    new_transactions.to_csv('../data/transactions.csv', index=False)
    new_transfers.to_csv('../data/transfers.csv', index=False)
    
    print(f"✅ Клиентов: {len(new_clients)} (было {len(clients)})")
    print(f"✅ Транзакций: {len(new_transactions)} (было {len(transactions)})")
    print(f"✅ Переводов: {len(new_transfers)} (было {len(transfers)})")

if __name__ == "__main__":
    add_missing_clients()