"""
Глубокий анализ данных для оптимизации системы персонализации.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def analyze_client_patterns():
    """Анализирует паттерны клиентов для оптимизации рекомендаций."""
    
    # Загружаем данные
    clients = pd.read_csv('../data/clients.csv')
    transactions = pd.read_csv('../data/transactions.csv') 
    transfers = pd.read_csv('../data/transfers.csv')
    
    print('=== АНАЛИЗ КЛИЕНТСКОЙ БАЗЫ ===')
    print(f'Клиентов: {len(clients)}')
    print(f'Транзакций: {len(transactions)}')
    print(f'Переводов: {len(transfers)}')
    
    # Объединяем данные по клиентам
    client_data = []
    for client_code in clients['client_code'].unique():
        client_info = clients[clients['client_code'] == client_code].iloc[0]
        client_trans = transactions[transactions['client_code'] == client_code]
        client_transfers = transfers[transfers['client_code'] == client_code]
        
        # Агрегируем траты по категориям
        spending = client_trans.groupby('category')['amount'].sum()
        
        # Агрегируем переводы по типам  
        transfer_sums = client_transfers.groupby('type')['amount'].sum()
        
        # Рассчитываем ключевые метрики согласно ТЗ
        travel_spend = (spending.get('Путешествия', 0) + 
                       spending.get('Отели', 0) + 
                       spending.get('Такси', 0))
        
        restaurant_spend = spending.get('Кафе и рестораны', 0)
        
        online_spend = (spending.get('Едим дома', 0) + 
                       spending.get('Смотрим дома', 0) + 
                       spending.get('Играем дома', 0))
        
        jewelry_spend = spending.get('Ювелирные украшения', 0)
        cosmetics_spend = spending.get('Косметика и Парфюмерия', 0)
        
        fx_activity = transfer_sums.get('fx_buy', 0) + transfer_sums.get('fx_sell', 0)
        
        balance = client_info['avg_monthly_balance_KZT']
        
        outflows = client_transfers[client_transfers['direction'] == 'out']['amount'].sum()
        inflows = client_transfers[client_transfers['direction'] == 'in']['amount'].sum()
        
        client_data.append({
            'client_code': client_code,
            'name': client_info['name'],
            'balance': balance,
            'travel_spend': travel_spend,
            'restaurant_spend': restaurant_spend,
            'online_spend': online_spend,
            'jewelry_spend': jewelry_spend,
            'cosmetics_spend': cosmetics_spend,
            'fx_activity': fx_activity,
            'total_spend': spending.sum(),
            'outflows': outflows,
            'inflows': inflows,
            'cash_flow': inflows - outflows,
            'spend_categories': len(spending[spending > 0]),
            'has_loan_payments': transfer_sums.get('loan_payment_out', 0) > 0,
            'has_fx_activity': fx_activity > 0
        })
    
    df = pd.DataFrame(client_data)
    
    print('\n=== КЛЮЧЕВЫЕ ПАТТЕРНЫ ===')
    print(f'Средний баланс: {df["balance"].mean():,.0f} ₸')
    print(f'Медианный баланс: {df["balance"].median():,.0f} ₸')
    print(f'Travel активных клиентов (>10k): {(df["travel_spend"] > 10000).sum()}')
    print(f'FX активных клиентов (>5k): {(df["fx_activity"] > 5000).sum()}')
    print(f'Онлайн активных (>15k): {(df["online_spend"] > 15000).sum()}')
    print(f'С дефицитом cash flow: {(df["outflows"] > df["inflows"]).sum()}')
    print(f'Премиальных (ресторан+ювелирка+косметика >10k): {((df["restaurant_spend"] + df["jewelry_spend"] + df["cosmetics_spend"]) > 10000).sum()}')
    
    print('\n=== РАСПРЕДЕЛЕНИЕ БАЛАНСОВ ===')
    for threshold in [100000, 500000, 1000000, 2000000]:
        count = (df['balance'] > threshold).sum()
        print(f'Баланс > {threshold:,}: {count} клиентов ({count/len(df)*100:.1f}%)')
    
    print('\n=== ТОП КАТЕГОРИИ ТРАТ ===')
    all_spending = transactions.groupby('category')['amount'].sum().sort_values(ascending=False)
    for cat, amount in all_spending.head(10).items():
        print(f'{cat}: {amount:,.0f} ₸')
    
    print('\n=== ТОП ТИПЫ ПЕРЕВОДОВ ===')
    all_transfers = transfers.groupby('type')['amount'].sum().sort_values(ascending=False)
    for transfer_type, amount in all_transfers.head(10).items():
        print(f'{transfer_type}: {amount:,.0f} ₸')
    
    # Анализ потенциала для каждого продукта
    print('\n=== ПОТЕНЦИАЛ ПРОДУКТОВ ===')
    
    # Карта для путешествий
    travel_potential = (df['travel_spend'] > 5000).sum()
    print(f'Карта для путешествий: {travel_potential} клиентов ({travel_potential/len(df)*100:.1f}%)')
    
    # Премиальная карта
    premium_potential = ((df['balance'] > 800000) & 
                        ((df['restaurant_spend'] + df['jewelry_spend'] + df['cosmetics_spend']) > 5000)).sum()
    print(f'Премиальная карта: {premium_potential} клиентов ({premium_potential/len(df)*100:.1f}%)')
    
    # Кредитная карта
    credit_potential = (df['online_spend'] > 10000).sum()
    print(f'Кредитная карта: {credit_potential} клиентов ({credit_potential/len(df)*100:.1f}%)')
    
    # Обмен валют
    fx_potential = (df['fx_activity'] > 3000).sum()
    print(f'Обмен валют: {fx_potential} клиентов ({fx_potential/len(df)*100:.1f}%)')
    
    # Кредит наличными
    cash_credit_potential = ((df['outflows'] > df['inflows']) & 
                            (df['balance'] < 300000)).sum()
    print(f'Кредит наличными: {cash_credit_potential} клиентов ({cash_credit_potential/len(df)*100:.1f}%)')
    
    # Инвестиции
    investment_potential = ((df['balance'] > 1000000) & 
                           (df['cash_flow'] > 50000)).sum()
    print(f'Инвестиции: {investment_potential} клиентов ({investment_potential/len(df)*100:.1f}%)')
    
    return df


def suggest_optimization_parameters(df):
    """Предлагает оптимальные параметры на основе анализа данных."""
    
    print('\n=== РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ ===')
    
    # Анализ порогов
    print('\nОптимальные пороги:')
    print(f'Travel threshold: {df["travel_spend"].quantile(0.8):.0f} ₸ (топ 20%)')
    print(f'Restaurant threshold: {df["restaurant_spend"].quantile(0.8):.0f} ₸ (топ 20%)')
    print(f'Online threshold: {df["online_spend"].quantile(0.8):.0f} ₸ (топ 20%)')
    print(f'FX threshold: {df["fx_activity"].quantile(0.8):.0f} ₸ (топ 20%)')
    print(f'Premium balance threshold: {df["balance"].quantile(0.6):.0f} ₸ (топ 40%)')
    
    # Анализ корреляций
    print('\nКорреляции:')
    correlations = df[['balance', 'travel_spend', 'restaurant_spend', 'online_spend', 'fx_activity']].corr()
    print(correlations)
    
    return {
        'travel_threshold': df["travel_spend"].quantile(0.8),
        'restaurant_threshold': df["restaurant_spend"].quantile(0.8),
        'online_threshold': df["online_spend"].quantile(0.8),
        'fx_threshold': df["fx_activity"].quantile(0.8),
        'premium_balance_threshold': df["balance"].quantile(0.6)
    }


if __name__ == "__main__":
    df = analyze_client_patterns()
    params = suggest_optimization_parameters(df)