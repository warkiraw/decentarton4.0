"""
Умный рекомендательный движок с нормализацией выгод и intelligent tie-breaking.
Оптимизирован для максимального разнообразия продуктов и точности рекомендаций.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any
from config import CONFIG

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartRecommendationEngine:
    """Умный движок рекомендаций с нормализацией и балансировкой."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.normalization_stats = {}
        
    def calculate_all_benefits_normalized(self, client_features: pd.Series) -> Dict[str, float]:
        """
        Рассчитывает нормализованные выгоды для всех продуктов.
        Каждая выгода в диапазоне [0, 1] для честного сравнения.
        """
        raw_benefits = self._calculate_raw_benefits(client_features)
        normalized_benefits = self._normalize_benefits(raw_benefits, client_features)
        
        return normalized_benefits
    
    def _calculate_raw_benefits(self, client_features: pd.Series) -> Dict[str, float]:
        """Рассчитывает сырые выгоды согласно ТЗ с оптимизированными формулами."""
        
        # Извлекаем ключевые метрики
        balance = client_features.get('avg_monthly_balance_KZT', 0)
        
        # Travel категории (согласно ТЗ)
        travel_spend = client_features.get('spend_Путешествия', 0)
        taxi_spend = client_features.get('spend_Такси', 0)
        hotel_spend = client_features.get('spend_Отели', 0)
        total_travel = travel_spend + taxi_spend + hotel_spend
        
        # Premium категории
        restaurant_spend = client_features.get('spend_Кафе и рестораны', 0)
        jewelry_spend = client_features.get('spend_Ювелирные украшения', 0)
        cosmetics_spend = client_features.get('spend_Косметика и Парфюмерия', 0)
        
        # Online категории
        online_spend = (client_features.get('spend_Едим дома', 0) + 
                       client_features.get('spend_Смотрим дома', 0) + 
                       client_features.get('spend_Играем дома', 0))
        
        # FX активность
        fx_buy = client_features.get('transfer_fx_buy', 0)
        fx_sell = client_features.get('transfer_fx_sell', 0)
        total_fx = fx_buy + fx_sell
        
        # Cash flow анализ
        transfer_out = client_features.get('transfer_transfer_out', 0)
        transfer_in = client_features.get('transfer_transfer_in', 0)
        
        benefits = {}
        
        # 1. Карта для путешествий (4% кэшбэк, лимит 10k/месяц)
        travel_cashback = min(total_travel * 0.04, 10000)
        travel_bonus = 1.0 if total_travel > 20000 else 0.5 if total_travel > 5000 else 0.1
        benefits['Карта для путешествий'] = travel_cashback * travel_bonus
        
        # 2. Премиальная карта (tier-based кэшбэк + привилегии)
        if balance > 500000:  # Снижаем порог
            tier_multiplier = 2.0 if balance > 1500000 else 1.5
            premium_spend = restaurant_spend + jewelry_spend + cosmetics_spend
            premium_cashback = min(premium_spend * 0.04 * tier_multiplier, 15000)
            privilege_value = min(balance * 0.0005, 5000)  # 0.05% от баланса
            benefits['Премиальная карта'] = premium_cashback + privilege_value
        else:
            benefits['Премиальная карта'] = 0
        
        # 3. Кредитная карта (10% в топ-категориях + онлайн)
        # Находим топ-3 категории клиента
        spend_categories = {k.replace('spend_', ''): v for k, v in client_features.items() 
                          if k.startswith('spend_') and v > 0}
        top_3_spend = sum(sorted(spend_categories.values(), reverse=True)[:3])
        
        credit_cashback = min(top_3_spend * 0.10 + online_spend * 0.10, 8000)
        installment_value = min(top_3_spend * 0.02, 2000)  # Рассрочка 0-2%
        benefits['Кредитная карта'] = credit_cashback + installment_value
        
        # 4. Обмен валют (экономия на спреде)
        if total_fx > 1000:  # Снижаем порог
            fx_savings = total_fx * 0.005  # 0.5% экономии
            auto_buy_bonus = 500 if total_fx > 10000 else 200
            benefits['Обмен валют'] = fx_savings + auto_buy_bonus
        else:
            benefits['Обмен валют'] = 0
        
        # 5. Кредит наличными (для кассовых разрывов)
        cash_deficit = max(0, transfer_out - transfer_in)
        if cash_deficit > 20000 and balance < 400000:  # Снижаем пороги
            interest_savings = cash_deficit * 0.06  # 6% экономии на ставке
            flexibility_bonus = min(cash_deficit * 0.01, 3000)
            benefits['Кредит наличными'] = interest_savings + flexibility_bonus
        else:
            benefits['Кредит наличными'] = 0
        
        # 6. Депозит мультивалютный (для FX-активных)
        if balance > 200000 and total_fx > 2000:  # Снижаем пороги
            deposit_income = balance * 0.08  # 8% годовых
            fx_hedge_bonus = min(total_fx * 0.01, 2000)  # Хеджирование
            benefits['Депозит мультивалютный'] = deposit_income + fx_hedge_bonus
        else:
            benefits['Депозит мультивалютный'] = balance * 0.08 * 0.3  # Частичная выгода
        
        # 7. Депозит сберегательный (базовый для всех)
        savings_income = balance * 0.06  # 6% годовых
        stability_bonus = 1000 if transfer_in >= transfer_out else 0
        benefits['Депозит сберегательный'] = savings_income + stability_bonus
        
        # 8. Депозит накопительный (для регулярных пополнений)
        if transfer_in > 10000 and transfer_in > transfer_out:
            accumulation_rate = 0.07 + min((transfer_in - transfer_out) / 100000, 0.02)  # До 9%
            benefits['Депозит накопительный'] = balance * accumulation_rate + (transfer_in - transfer_out) * 0.5
        else:
            benefits['Депозит накопительный'] = balance * 0.07 * 0.5
        
        # 9. Инвестиции (для состоятельных с профицитом)
        if balance > 800000 and transfer_in > transfer_out:  # Снижаем порог
            investment_amount = min(balance * 0.3, 500000)
            expected_return = investment_amount * 0.12  # 12% ожидаемая доходность
            diversification_bonus = min(balance * 0.001, 3000)
            benefits['Инвестиции'] = expected_return + diversification_bonus
        else:
            benefits['Инвестиции'] = 0
        
        # 10. Золотые слитки (альтернативная инвестиция)
        if balance > 600000:  # Снижаем порог
            gold_amount = min(balance * 0.2, 300000)
            gold_return = gold_amount * 0.05  # 5% ожидаемая доходность
            inflation_hedge = min(balance * 0.0005, 1500)
            benefits['Золотые слитки'] = gold_return + inflation_hedge
        else:
            benefits['Золотые слитки'] = 0
        
        return benefits
    
    def _normalize_benefits(self, raw_benefits: Dict[str, float], client_features: pd.Series) -> Dict[str, float]:
        """
        Нормализует выгоды для честного сравнения и добавляет продукт-специфичные бонусы.
        """
        if not raw_benefits:
            return {}
        
        # Базовая нормализация в [0, 1]
        max_benefit = max(raw_benefits.values())
        if max_benefit == 0:
            return {product: 0.1 for product in raw_benefits}  # Минимальная выгода для всех
        
        normalized = {product: benefit / max_benefit for product, benefit in raw_benefits.items()}
        
        # Добавляем продукт-специфичные бонусы для разнообразия
        normalized = self._add_diversity_bonuses(normalized, client_features)
        
        # Применяем кластерные предпочтения
        normalized = self._add_cluster_affinity(normalized, client_features)
        
        return normalized
    
    def _add_diversity_bonuses(self, normalized: Dict[str, float], client_features: pd.Series) -> Dict[str, float]:
        """Добавляет бонусы для увеличения разнообразия продуктов."""
        
        # Получаем ключевые характеристики клиента
        balance = client_features.get('avg_monthly_balance_KZT', 0)
        
        # Бонусы основанные на сигналах из ТЗ
        if client_features.get('spend_Такси', 0) > 10000:
            normalized['Карта для путешествий'] += 0.2
        
        if client_features.get('spend_Кафе и рестораны', 0) > 15000 and balance > 500000:
            normalized['Премиальная карта'] += 0.3
        
        online_total = (client_features.get('spend_Едим дома', 0) + 
                       client_features.get('spend_Смотрим дома', 0) + 
                       client_features.get('spend_Играем дома', 0))
        if online_total > 20000:
            normalized['Кредитная карта'] += 0.25
        
        if client_features.get('transfer_fx_buy', 0) + client_features.get('transfer_fx_sell', 0) > 5000:
            normalized['Обмен валют'] += 0.4
        
        # Убираем негативную корреляцию для депозитов
        if balance > 0:
            normalized['Депозит сберегательный'] = max(0.1, normalized['Депозит сберегательный'] * 0.7)
        
        # Обеспечиваем минимальную выгоду для всех продуктов
        for product in normalized:
            normalized[product] = max(0.05, normalized[product])
        
        return normalized
    
    def _add_cluster_affinity(self, normalized: Dict[str, float], client_features: pd.Series) -> Dict[str, float]:
        """Добавляет кластерные предпочтения для tie-breaking."""
        
        cluster = client_features.get('cluster', 0)
        
        # Кластерные предпочтения (из анализа данных)
        cluster_bonuses = {
            0: ['Карта для путешествий', 'Обмен валют'],  # Путешественники
            1: ['Кредитная карта', 'Депозит накопительный'],  # Активные онлайн
            2: ['Премиальная карта', 'Инвестиции'],  # VIP клиенты
            3: ['Депозит сберегательный', 'Золотые слитки']  # Консерваторы
        }
        
        if cluster in cluster_bonuses:
            for product in cluster_bonuses[cluster]:
                if product in normalized:
                    normalized[product] += 0.1
        
        return normalized
    
    def apply_smart_rules_and_select(self, benefits: Dict[str, float], client_features: pd.Series) -> str:
        """
        Применяет умные правила выбора с множественными критериями.
        """
        if not benefits:
            return 'Депозит сберегательный'
        
        # 1. Проверяем жесткие правила из ТЗ
        mandatory_product = self._check_mandatory_rules(client_features)
        if mandatory_product and benefits.get(mandatory_product, 0) > 0.1:
            return mandatory_product
        
        # 2. Фильтруем продукты с минимальной выгодой
        viable_products = {p: b for p, b in benefits.items() if b > 0.1}
        
        if not viable_products:
            return max(benefits, key=benefits.get)
        
        # 3. Применяем умный tie-breaking
        if len(viable_products) > 1:
            return self._intelligent_tie_breaking(viable_products, client_features)
        
        return max(viable_products, key=viable_products.get)
    
    def _check_mandatory_rules(self, client_features: pd.Series) -> str:
        """Проверяет обязательные правила из ТЗ."""
        
        # Высокая FX активность -> обязательно FX продукт
        fx_total = client_features.get('transfer_fx_buy', 0) + client_features.get('transfer_fx_sell', 0)
        if fx_total > 15000:
            return 'Обмен валют'
        
        # Высокие траты на путешествия -> travel карта
        travel_total = (client_features.get('spend_Путешествия', 0) + 
                       client_features.get('spend_Такси', 0) + 
                       client_features.get('spend_Отели', 0))
        if travel_total > 25000:
            return 'Карта для путешествий'
        
        # Критический кассовый разрыв -> кредит наличными
        cash_deficit = (client_features.get('transfer_transfer_out', 0) - 
                       client_features.get('transfer_transfer_in', 0))
        balance = client_features.get('avg_monthly_balance_KZT', 0)
        if cash_deficit > 50000 and balance < 200000:
            return 'Кредит наличными'
        
        return None
    
    def _intelligent_tie_breaking(self, viable_products: Dict[str, float], client_features: pd.Series) -> str:
        """Умный tie-breaking с множественными критериями."""
        
        # Сортируем по выгоде
        sorted_products = sorted(viable_products.items(), key=lambda x: x[1], reverse=True)
        
        # Если есть явный лидер (>20% разница), возвращаем его
        if len(sorted_products) > 1 and sorted_products[0][1] > sorted_products[1][1] * 1.2:
            return sorted_products[0][0]
        
        # Иначе применяем дополнительные критерии
        top_products = [p for p, b in sorted_products if b >= sorted_products[0][1] * 0.9]
        
        # Propensity scores (если доступны)
        propensity_scores = {}
        for product in top_products:
            prop_key = f'propensity_{product.lower().replace(" ", "_").replace("ё", "е")}'
            propensity_scores[product] = client_features.get(prop_key, 0.5)
        
        # Комбинированный скор: 70% выгода + 30% propensity
        combined_scores = {}
        for product in top_products:
            benefit_score = viable_products[product]
            propensity_score = propensity_scores.get(product, 0.5)
            combined_scores[product] = 0.7 * benefit_score + 0.3 * propensity_score
        
        return max(combined_scores, key=combined_scores.get)


# Глобальный экземпляр движка
smart_engine = SmartRecommendationEngine(CONFIG.__dict__)


def calculate_all_benefits(client_features: pd.Series, config: Dict[str, Any]) -> Dict[str, float]:
    """Обертка для совместимости с существующим кодом."""
    return smart_engine.calculate_all_benefits_normalized(client_features)


def apply_rules_and_select_best(benefits: Dict[str, float], client_features: pd.Series, config: Dict[str, Any]) -> str:
    """Обертка для совместимости с существующим кодом."""
    return smart_engine.apply_smart_rules_and_select(benefits, client_features)


def test_smart_engine():
    """Тестирует умный движок на примере клиентов."""
    
    # Тестовый клиент с высокой travel активностью
    travel_client = pd.Series({
        'client_code': 999,
        'avg_monthly_balance_KZT': 300000,
        'spend_Такси': 25000,
        'spend_Путешествия': 15000,
        'spend_Отели': 8000,
        'spend_Кафе и рестораны': 12000,
        'cluster': 0
    })
    
    benefits = calculate_all_benefits(travel_client, CONFIG.__dict__)
    recommendation = apply_rules_and_select_best(benefits, travel_client, CONFIG.__dict__)
    
    print("=== ТЕСТ УМНОГО ДВИЖКА ===")
    print(f"Travel клиент - Рекомендация: {recommendation}")
    print(f"Топ-3 выгоды: {dict(sorted(benefits.items(), key=lambda x: x[1], reverse=True)[:3])}")
    
    return benefits, recommendation


if __name__ == "__main__":
    test_smart_engine()