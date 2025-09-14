"""
Модуль обработки данных системы персонализации банковских предложений.
Содержит функции для загрузки, очистки и объединения данных из CSV файлов.
"""

import pandas as pd
import logging
from typing import Dict
from config import CONFIG

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_datasets(config: Dict[str, any]) -> Dict[str, pd.DataFrame]:
    """
    Загружает все наборы данных из CSV файлов.
    
    Args:
        config: Словарь конфигурации с путями к файлам
        
    Returns:
        Словарь с DataFrame для каждого набора данных
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если обязательные столбцы отсутствуют
    """
    dataframes = {}
    
    try:
        # Загрузка клиентов
        df_clients = pd.read_csv(config['DATA_PATHS']['clients'])
        if 'client_code' not in df_clients.columns:
            raise ValueError("Столбец 'client_code' отсутствует в файле clients.csv")
        dataframes['clients'] = df_clients
        logger.info(f"Загружено {len(df_clients)} записей клиентов")
        
        # Загрузка транзакций
        df_transactions = pd.read_csv(config['DATA_PATHS']['transactions'])
        if 'client_code' not in df_transactions.columns:
            raise ValueError("Столбец 'client_code' отсутствует в файле transactions.csv")
        dataframes['transactions'] = df_transactions
        logger.info(f"Загружено {len(df_transactions)} транзакций")
        
        # Загрузка переводов
        df_transfers = pd.read_csv(config['DATA_PATHS']['transfers'])
        if 'client_code' not in df_transfers.columns:
            raise ValueError("Столбец 'client_code' отсутствует в файле transfers.csv")
        dataframes['transfers'] = df_transfers
        logger.info(f"Загружено {len(df_transfers)} переводов")
        
    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        raise
        
    return dataframes


def preprocess_and_merge(dataframes: Dict[str, pd.DataFrame], config: Dict[str, any]) -> pd.DataFrame:
    """
    Выполняет предобработку и объединение всех наборов данных.
    
    Args:
        dataframes: Словарь с DataFrame
        config: Словарь конфигурации
        
    Returns:
        Объединенный и предобработанный DataFrame
    """
    logger.info("Начало предобработки и объединения данных")
    
    df_clients = dataframes['clients'].copy()
    df_transactions = dataframes['transactions'].copy()
    df_transfers = dataframes['transfers'].copy()
    
    # 1. Конверсия дат
    if 'date' in df_transactions.columns:
        df_transactions['date'] = pd.to_datetime(df_transactions['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    if 'date' in df_transfers.columns:
        df_transfers['date'] = pd.to_datetime(df_transfers['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # 2. Унификация валют для транзакций
    if 'currency' in df_transactions.columns and 'amount' in df_transactions.columns:
        df_transactions['amount_kzt'] = df_transactions.apply(
            lambda row: row['amount'] * config['CURRENCY_RATES'].get(row['currency'].upper(), 1.0)
            if pd.notna(row['currency']) and pd.notna(row['amount']) else 0,
            axis=1
        )
    else:
        df_transactions['amount_kzt'] = 0
    
    # 3. Унификация валют для переводов
    if 'currency' in df_transfers.columns and 'amount' in df_transfers.columns:
        df_transfers['amount_kzt'] = df_transfers.apply(
            lambda row: row['amount'] * config['CURRENCY_RATES'].get(row['currency'].upper(), 1.0)
            if pd.notna(row['currency']) and pd.notna(row['amount']) else 0,
            axis=1
        )
    else:
        df_transfers['amount_kzt'] = 0
    
    # 4. Обработка пропущенных значений
    # Для транзакций
    df_transactions['amount_kzt'] = df_transactions['amount_kzt'].fillna(0)
    df_transactions = df_transactions.dropna(subset=['date', 'category'], how='any')
    
    # Для переводов
    df_transfers['amount_kzt'] = df_transfers['amount_kzt'].fillna(0)
    df_transfers = df_transfers.dropna(subset=['date', 'type'], how='any')
    
    # Для клиентов
    if 'avg_monthly_balance_KZT' in df_clients.columns:
        df_clients['avg_monthly_balance_KZT'] = df_clients['avg_monthly_balance_KZT'].fillna(0)
    
    # 5. Объединение данных
    # Сначала объединяем клиентов с транзакциями
    df_merged = pd.merge(df_clients, df_transactions, on='client_code', how='left', suffixes=('', '_trans'))
    
    # Затем объединяем с переводами
    df_merged = pd.merge(df_merged, df_transfers, on='client_code', how='left', suffixes=('', '_transfer'))
    
    # 6. Улучшенное удаление выбросов (КРИТИЧЕСКИЙ ФИКС: сохраняем всех клиентов!)
    def remove_outliers_by_client_safe(group):
        """Удаляет выбросы по IQR методу, но ВСЕГДА сохраняет хотя бы несколько записей клиента."""
        if len(group) < 4:  # Недостаточно данных для расчета IQR
            return group
            
        Q1 = group['amount_kzt'].quantile(0.25)
        Q3 = group['amount_kzt'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Если IQR = 0 (все суммы одинаковые), возвращаем как есть
        if IQR == 0:
            return group
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filtered = group[(group['amount_kzt'] >= lower_bound) & (group['amount_kzt'] <= upper_bound)]
        
        # КРИТИЧЕСКАЯ ЗАЩИТА: если все записи клиента считаются выбросами,
        # сохраняем хотя бы медианные записи
        if len(filtered) == 0:
            median_amount = group['amount_kzt'].median()
            # Берем записи ближайшие к медиане
            distances = abs(group['amount_kzt'] - median_amount)
            # Сохраняем минимум 3 записи или 25% от общего числа
            keep_count = max(3, len(group) // 4)
            keep_indices = distances.nsmallest(keep_count).index
            return group.loc[keep_indices]
        
        return filtered
    
    if len(df_merged) > 0:
        # КРИТИЧЕСКИЙ ФИКС: сохраняем client_code после apply
        df_merged = df_merged.groupby('client_code', group_keys=False).apply(remove_outliers_by_client_safe).reset_index(drop=True)
    
    # 7. Добавление месяца для NLG
    if 'date' in df_merged.columns:
        month_mapping = {
            1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
            5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
            9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
        }
        df_merged['month'] = df_merged['date'].dt.month.map(month_mapping)
    
    # 8. Группировка по client_code для получения агрегированных данных
    client_aggregated = []
    
    for client_code, group in df_merged.groupby('client_code'):
        # Основная информация о клиенте
        client_info = group.iloc[0]
        
        # Агрегация транзакций по категориям
        spend_by_category = {}
        if 'category' in group.columns:
            category_spending = group.groupby('category')['amount_kzt'].sum()
            for category, amount in category_spending.items():
                spend_by_category[f'spend_{category}'] = amount
        
        # Агрегация переводов по типам
        transfer_by_type = {}
        if 'type' in group.columns:
            type_transfers = group.groupby('type')['amount_kzt'].sum()
            for transfer_type, amount in type_transfers.items():
                transfer_by_type[f'transfer_{transfer_type}'] = amount
        
        # Создание записи клиента
        client_record = {
            'client_code': client_code,
            'name': client_info.get('name', ''),
            'status': client_info.get('status', ''),
            'age': client_info.get('age', 0),
            'city': client_info.get('city', ''),
            'avg_monthly_balance_KZT': client_info.get('avg_monthly_balance_KZT', 0),
            'month': client_info.get('month', 'Сентябрь'),
            **spend_by_category,
            **transfer_by_type
        }
        
        client_aggregated.append(client_record)
    
    result_df = pd.DataFrame(client_aggregated)
    
    # Заполнение пропущенных значений нулями для spending категорий
    spending_cols = [col for col in result_df.columns if col.startswith('spend_') or col.startswith('transfer_')]
    for col in spending_cols:
        result_df[col] = result_df[col].fillna(0)
    
    logger.info(f"Предобработка завершена. Итоговый DataFrame: {len(result_df)} записей")
    
    return result_df


def test_load_datasets():
    """Тест функции загрузки данных."""
    try:
        # Создаем тестовые данные
        test_config = {
            'DATA_PATHS': {
                'clients': 'test_clients.csv',
                'transactions': 'test_transactions.csv',
                'transfers': 'test_transfers.csv'
            }
        }
        
        # Создаем тестовые CSV файлы
        test_clients = pd.DataFrame({
            'client_code': [1, 2, 3],
            'name': ['Иван Иванов', 'Анна Петрова', 'Петр Сидоров'],
            'avg_monthly_balance_KZT': [100000, 200000, 150000]
        })
        
        test_transactions = pd.DataFrame({
            'client_code': [1, 1, 2],
            'date': ['2025-01-15 10:30:00', '2025-02-20 14:20:00', '2025-01-10 09:15:00'],
            'category': ['Такси', 'Путешествия', 'Ресторан'],
            'amount': [5000, 50000, 15000],
            'currency': ['KZT', 'KZT', 'KZT']
        })
        
        test_transfers = pd.DataFrame({
            'client_code': [1, 2, 3],
            'date': ['2025-01-15 10:30:00', '2025-02-20 14:20:00', '2025-01-10 09:15:00'],
            'type': ['fx_buy', 'fx_sell', 'transfer_out'],
            'amount': [10000, 20000, 5000],
            'currency': ['USD', 'EUR', 'KZT']
        })
        
        # Сохраняем тестовые файлы
        test_clients.to_csv('test_clients.csv', index=False)
        test_transactions.to_csv('test_transactions.csv', index=False)
        test_transfers.to_csv('test_transfers.csv', index=False)
        
        # Тестируем загрузку
        dataframes = load_datasets(test_config)
        assert len(dataframes) == 3
        assert 'clients' in dataframes
        assert 'transactions' in dataframes
        assert 'transfers' in dataframes
        
        print("Тест load_datasets пройден успешно!")
        
        # Очистка тестовых файлов
        import os
        os.remove('test_clients.csv')
        os.remove('test_transactions.csv') 
        os.remove('test_transfers.csv')
        
    except Exception as e:
        print(f"Тест load_datasets не пройден: {e}")


if __name__ == "__main__":
    test_load_datasets()