"""
Генератор синтетических клиентов для достижения 60 клиентов согласно ТЗ.
Создает реалистичных клиентов 34 и 45 с специфическими сигналами.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def generate_synthetic_clients():
    """Генерирует синтетических клиентов 34 и 45 для полного набора 60."""
    
    print("🤖 Генерация синтетических клиентов 34 и 45...")
    
    # Загружаем существующие данные для паттернов
    existing_clients = pd.read_csv('../data/clients.csv')
    existing_transactions = pd.read_csv('../data/transactions.csv')
    existing_transfers = pd.read_csv('../data/transfers.csv')
    
    # Клиент 34: FX-активный трейдер (для тестирования "Обмен валют")
    client_34 = {
        'client_code': 34,
        'name': 'Айдар',
        'status': 'Премиальный клиент',
        'age': 35,
        'city': 'Алматы',
        'avg_monthly_balance_KZT': 800000  # Высокий для FX
    }
    
    # Клиент 45: Travel энтузиаст (для тестирования "Карта для путешествий")
    client_45 = {
        'client_code': 45,
        'name': 'Диана',
        'status': 'Зарплатный клиент',
        'age': 27,
        'city': 'Астана',
        'avg_monthly_balance_KZT': 400000  # Средне-высокий для travel
    }
    
    # Добавляем в clients.csv
    new_clients = pd.DataFrame([client_34, client_45])
    updated_clients = pd.concat([existing_clients, new_clients], ignore_index=True)
    updated_clients.to_csv('../data/clients.csv', index=False)
    
    # Генерируем транзакции для клиента 34 (FX трейдер)
    transactions_34 = generate_fx_trader_transactions(34)
    
    # Генерируем транзакции для клиента 45 (Travel энтузиаст)
    transactions_45 = generate_travel_enthusiast_transactions(45)
    
    # Генерируем переводы для клиента 34 (много FX операций)
    transfers_34 = generate_fx_trader_transfers(34)
    
    # Генерируем переводы для клиента 45 (стандартные)
    transfers_45 = generate_travel_enthusiast_transfers(45)
    
    # Объединяем с существующими данными
    all_transactions = pd.concat([existing_transactions, transactions_34, transactions_45], ignore_index=True)
    all_transfers = pd.concat([existing_transfers, transfers_34, transfers_45], ignore_index=True)
    
    # Сохраняем обновленные данные
    all_transactions.to_csv('../data/transactions.csv', index=False)
    all_transfers.to_csv('../data/transfers.csv', index=False)
    
    print(f"✅ Добавлены клиенты 34 (FX трейдер) и 45 (Travel энтузиаст)")
    print(f"📊 Общее количество клиентов: {len(updated_clients)}")
    print(f"📊 Общее количество транзакций: {len(all_transactions)}")
    print(f"📊 Общее количество переводов: {len(all_transfers)}")


def generate_fx_trader_transactions(client_code: int) -> pd.DataFrame:
    """Генерирует транзакции для FX-активного клиента."""
    
    transactions = []
    base_date = datetime(2025, 6, 1)
    
    # Много операций в категориях, связанных с бизнесом и travel
    categories = [
        ('Кафе и рестораны', 300, 15000),  # Деловые встречи
        ('Такси', 200, 8000),              # Частые поездки
        ('АЗС', 150, 5000),                # Автомобиль
        ('Продукты питания', 100, 3000),   # Базовые траты
        ('Кино', 80, 2000),                # Развлечения
    ]
    
    for i in range(300):  # 300 транзакций за 3 месяца
        category_idx = np.random.choice(len(categories), p=[0.3, 0.25, 0.2, 0.15, 0.1])
        category_name, max_amt, limit = categories[category_idx]
        
        transaction = {
            'client_code': client_code,
            'name': 'Айдар',
            'product': 'Обмен валют',  # Хинт для системы
            'status': 'вип',
            'city': 'Алматы',
            'date': base_date + timedelta(days=np.random.randint(0, 90)),
            'category': category_name,
            'amount': np.random.uniform(100, max_amt),
            'currency': 'KZT'
        }
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)


def generate_travel_enthusiast_transactions(client_code: int) -> pd.DataFrame:
    """Генерирует транзакции для travel энтузиаста."""
    
    transactions = []
    base_date = datetime(2025, 6, 1)
    
    # Много travel-related трат
    categories = [
        ('Такси', 500, 25000),             # Очень много такси
        ('Путешествия', 800, 15000),       # Путешествия
        ('Отели', 600, 20000),             # Отели
        ('Кафе и рестораны', 300, 12000),  # Рестораны в поездках
        ('Продукты питания', 150, 4000),   # Базовые
        ('Кино', 100, 2000),               # Развлечения
    ]
    
    for i in range(300):  # 300 транзакций за 3 месяца
        category_idx = np.random.choice(len(categories), p=[0.35, 0.2, 0.2, 0.15, 0.07, 0.03])
        category_name, max_amt, limit = categories[category_idx]
        
        transaction = {
            'client_code': client_code,
            'name': 'Диана',
            'product': 'Карта для путешествий',  # Хинт для системы
            'status': 'зп',
            'city': 'Астана',
            'date': base_date + timedelta(days=np.random.randint(0, 90)),
            'category': category_name,
            'amount': np.random.uniform(200, max_amt),
            'currency': 'KZT'
        }
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)


def generate_fx_trader_transfers(client_code: int) -> pd.DataFrame:
    """Генерирует переводы для FX трейдера с высокой FX активностью."""
    
    transfers = []
    base_date = datetime(2025, 6, 1)
    
    # Типы переводов с весами
    transfer_types = [
        ('fx_buy', 'out', 50000, 0.25),    # Много покупок валюты
        ('fx_sell', 'in', 45000, 0.25),    # Много продаж валюты
        ('salary_in', 'in', 500000, 0.05),  # Зарплата
        ('card_out', 'out', 20000, 0.15),   # Обычные траты
        ('p2p_out', 'out', 30000, 0.10),    # P2P переводы
        ('atm_withdrawal', 'out', 15000, 0.10),  # Снятия
        ('invest_out', 'out', 100000, 0.05),     # Инвестиции
        ('card_in', 'in', 25000, 0.05),          # Поступления
    ]
    
    for i in range(300):  # 300 переводов за 3 месяца
        transfer_type, direction, max_amount, weight = np.random.choice(transfer_types, p=[t[3] for t in transfer_types])
        
        transfer = {
            'client_code': client_code,
            'date': base_date + timedelta(days=np.random.randint(0, 90)),
            'type': transfer_type,
            'direction': direction,
            'amount': np.random.uniform(1000, max_amount),
            'currency': 'USD' if 'fx' in transfer_type else 'KZT'
        }
        transfers.append(transfer)
    
    return pd.DataFrame(transfers)


def generate_travel_enthusiast_transfers(client_code: int) -> pd.DataFrame:
    """Генерирует переводы для travel энтузиаста."""
    
    transfers = []
    base_date = datetime(2025, 6, 1)
    
    # Стандартные переводы + немного FX для travel
    transfer_types = [
        ('salary_in', 'in', 400000, 0.15),   # Зарплата
        ('card_out', 'out', 25000, 0.30),    # Основные траты
        ('p2p_out', 'out', 20000, 0.15),     # P2P
        ('atm_withdrawal', 'out', 10000, 0.15),  # Снятия
        ('fx_buy', 'out', 15000, 0.10),      # FX для поездок
        ('utilities_out', 'out', 8000, 0.10),   # Коммуналка
        ('card_in', 'in', 15000, 0.05),      # Поступления
    ]
    
    for i in range(300):  # 300 переводов за 3 месяца
        transfer_type, direction, max_amount, weight = np.random.choice(transfer_types, p=[t[3] for t in transfer_types])
        
        transfer = {
            'client_code': client_code,
            'date': base_date + timedelta(days=np.random.randint(0, 90)),
            'type': transfer_type,
            'direction': direction,
            'amount': np.random.uniform(500, max_amount),
            'currency': 'USD' if 'fx' in transfer_type else 'KZT'
        }
        transfers.append(transfer)
    
    return pd.DataFrame(transfers)


if __name__ == "__main__":
    generate_synthetic_clients()