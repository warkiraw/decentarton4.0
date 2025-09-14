"""
Комплексная диагностика проблем системы персонализации.
Senior Data Scientist уровень - выявляем все критические баги.
"""

import pandas as pd
import numpy as np
import os

def comprehensive_diagnosis():
    """Проводит полную диагностику системы."""
    
    print("🔍 === COMPREHENSIVE SYSTEM DIAGNOSIS ===")
    
    # 1. Диагностика данных
    print("\n📊 ДИАГНОСТИКА ДАННЫХ:")
    clients = pd.read_csv('../data/clients.csv')
    transactions = pd.read_csv('../data/transactions.csv')
    transfers = pd.read_csv('../data/transfers.csv')
    
    print(f"Клиентов в системе: {len(clients)} (ожидается 60)")
    print(f"Транзакций: {len(transactions):,}")
    print(f"Переводов: {len(transfers):,}")
    
    # 2. Анализ балансов (критический для benefit calculation)
    print(f"\n💰 АНАЛИЗ БАЛАНСОВ:")
    print(f"Средний баланс: {clients['avg_monthly_balance_KZT'].mean():,.0f} ₸")
    print(f"Медиана баланса: {clients['avg_monthly_balance_KZT'].median():,.0f} ₸")
    print(f"Макс баланс: {clients['avg_monthly_balance_KZT'].max():,.0f} ₸")
    print(f"Мин баланс: {clients['avg_monthly_balance_KZT'].min():,.0f} ₸")
    
    # 3. Диагностика результатов
    if os.path.exists('../data/output_extended.csv'):
        output = pd.read_csv('../data/output_extended.csv')
        print(f"\n🎯 ДИАГНОСТИКА РЕЗУЛЬТАТОВ:")
        print(f"Клиентов обработано: {len(output)}")
        print(f"Уникальных продуктов: {output['product'].nunique()}/10")
        
        if '_benefit' in output.columns:
            print(f"Средняя выгода: {output['_benefit'].mean():.1f} ₸ (КРИТИЧНО: должно быть >1000)")
            print(f"Макс выгода: {output['_benefit'].max():.1f} ₸")
            print(f"Мин выгода: {output['_benefit'].min():.1f} ₸")
        
        print(f"\n📈 РАСПРЕДЕЛЕНИЕ ПРОДУКТОВ:")
        product_dist = output['product'].value_counts()
        missing_products = []
        for product in ['Карта для путешествий', 'Премиальная карта', 'Кредитная карта', 
                       'Обмен валют', 'Кредит наличными', 'Депозит мультивалютный', 
                       'Депозит сберегательный', 'Депозит накопительный', 'Инвестиции', 'Золотые слитки']:
            count = product_dist.get(product, 0)
            if count == 0:
                missing_products.append(product)
            print(f"  {product}: {count} ({count/len(output)*100:.1f}%)")
        
        if missing_products:
            print(f"\n⚠️ ОТСУТСТВУЮЩИЕ ПРОДУКТЫ: {missing_products}")
    
    # 4. Анализ данных для missing клиентов
    print(f"\n🔍 АНАЛИЗ MISSING КЛИЕНТОВ:")
    case1_files = os.listdir('../case 1')
    trans_files = [f for f in case1_files if 'transactions' in f]
    client_ids = [int(f.split('_')[1]) for f in trans_files]
    
    all_expected = set(range(1, 61))
    available = set(client_ids)
    missing = all_expected - available
    
    print(f"Ожидается клиентов: 60")
    print(f"Доступно файлов транзакций: {len(client_ids)}")
    print(f"Missing клиенты: {sorted(missing)}")
    
    # 5. Анализ категорий трат
    print(f"\n🛍️ АНАЛИЗ КАТЕГОРИЙ ТРАТ:")
    category_totals = transactions.groupby('category')['amount'].sum().sort_values(ascending=False)
    print("Топ-5 категорий по объему:")
    for cat, amount in category_totals.head().items():
        print(f"  {cat}: {amount:,.0f} ₸")
    
    # 6. Анализ FX активности
    print(f"\n💱 АНАЛИЗ FX АКТИВНОСТИ:")
    fx_transfers = transfers[transfers['type'].isin(['fx_buy', 'fx_sell'])]
    fx_volume = fx_transfers['amount'].sum()
    fx_clients = fx_transfers['client_code'].nunique()
    print(f"Общий FX объем: {fx_volume:,.0f} ₸")
    print(f"FX активных клиентов: {fx_clients}/58 ({fx_clients/58*100:.1f}%)")
    
    return {
        'clients_count': len(clients),
        'missing_clients': missing,
        'fx_active_ratio': fx_clients/58,
        'avg_balance': clients['avg_monthly_balance_KZT'].mean()
    }

if __name__ == "__main__":
    results = comprehensive_diagnosis()
    print(f"\n🎯 КЛЮЧЕВЫЕ FINDINGS:")
    print(f"- Нужно добавить {60 - results['clients_count']} клиентов")
    print(f"- FX активность: {results['fx_active_ratio']*100:.1f}% клиентов")
    print(f"- Средний баланс: {results['avg_balance']:,.0f} ₸")