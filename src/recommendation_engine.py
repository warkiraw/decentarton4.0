"""
Модуль рекомендательного движка для системы персонализации банковских предложений.
Содержит функции для расчета выгоды продуктов и применения бизнес-правил.
"""

import pandas as pd
import json
import logging
from typing import Dict, Any
from config import CONFIG

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_all_benefits(client_features: pd.Series, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Рассчитывает выгоду для всех банковских продуктов для конкретного клиента.
    
    Args:
        client_features: Серия с характеристиками клиента
        config: Словарь конфигурации
        
    Returns:
        Словарь с выгодой для каждого продукта
    """
    benefits = {}
    
    # Получаем основные характеристики клиента
    balance = client_features.get('avg_monthly_balance_KZT', 0)
    
    # Траты по категориям согласно ТЗ
    travel_spend = client_features.get('spend_Путешествия', 0)
    taxi_spend = client_features.get('spend_Такси', 0)
    hotel_spend = client_features.get('spend_Отели', 0)
    restaurant_spend = client_features.get('spend_Кафе и рестораны', 0)
    jewelry_spend = client_features.get('spend_Ювелирные украшения', 0)
    cosmetics_spend = client_features.get('spend_Косметика и Парфюмерия', 0)
    entertainment_spend = client_features.get('spend_Развлечения', 0)
    
    # Онлайн категории
    online_spend = (client_features.get('spend_Едим дома', 0) + 
                   client_features.get('spend_Смотрим дома', 0) + 
                   client_features.get('spend_Играем дома', 0))
    
    # Топ-3 категории для кредитной карты
    all_spend_categories = {k: v for k, v in client_features.items() if k.startswith('spend_')}
    top_categories = sorted(all_spend_categories.items(), key=lambda x: x[1], reverse=True)[:3]
    top_categories_spend = sum([amount for _, amount in top_categories])
    
    # Переводы
    fx_buy = client_features.get('transfer_fx_buy', 0)
    fx_sell = client_features.get('transfer_fx_sell', 0)
    transfer_out = client_features.get('transfer_transfer_out', 0)
    transfer_in = client_features.get('transfer_transfer_in', 0)
    
    # 1. Карта для путешествий
    benefits['Карта для путешествий'] = _calc_travel_benefit(
        travel_spend + hotel_spend, taxi_spend, balance, config
    )
    
    # 2. Премиальная карта
    benefits['Премиальная карта'] = _calc_premium_benefit(
        balance, restaurant_spend, jewelry_spend, cosmetics_spend, config
    )
    
    # 3. Кредитная карта
    benefits['Кредитная карта'] = _calc_credit_benefit(
        restaurant_spend, entertainment_spend, entertainment_spend, 
        online_spend, top_categories_spend, balance, config
    )
    
    # 4. Обмен валют
    benefits['Обмен валют'] = _calc_fx_benefit(
        fx_buy, fx_sell, config
    )
    
    # 5. Кредит наличными
    benefits['Кредит наличными'] = _calc_cash_credit_benefit(
        transfer_out, transfer_in, balance, config
    )
    
    # 6. Депозит мультивалютный
    benefits['Депозит мультивалютный'] = _calc_multi_deposit_benefit(
        balance, fx_buy, fx_sell, config
    )
    
    # 7. Депозит сберегательный
    benefits['Депозит сберегательный'] = _calc_savings_deposit_benefit(
        balance, transfer_in, transfer_out, config
    )
    
    # 8. Депозит накопительный
    benefits['Депозит накопительный'] = _calc_accumulation_deposit_benefit(
        balance, transfer_in, config
    )
    
    # 9. Инвестиции
    benefits['Инвестиции'] = _calc_investment_benefit(
        balance, transfer_in, transfer_out, config
    )
    
    # 10. Золотые слитки
    benefits['Золотые слитки'] = _calc_gold_benefit(
        balance, fx_buy, fx_sell, config
    )
    
    return benefits


def _calc_travel_benefit(travel_spend: float, taxi_spend: float, balance: float, config: Dict[str, Any]) -> float:
    """Расчет выгоды для карты путешествий согласно ТЗ."""
    # 4% кэшбэк на путешествия, отели, такси, транспорт
    travel_categories_spend = travel_spend + taxi_spend
    cashback = 0.04 * travel_categories_spend
    
    # Бонус за высокий баланс
    if balance > config['RFMD_THRESHOLDS']['high_balance']:
        cashback *= 1.2
    
    # Максимальный кэшбэк 10,000 тенге в месяц
    return min(cashback, 10000)


def _calc_premium_benefit(balance: float, restaurant_spend: float, jewelry_spend: float, cosmetics_spend: float, config: Dict[str, Any]) -> float:
    """Расчет выгоды для премиальной карты согласно ТЗ."""
    if balance < config['RFMD_THRESHOLDS']['high_balance']:
        return 0  # Премиальная карта только для VIP клиентов
    
    # Tier-based cashback: 2-4% базовый кэшбэк
    tier_multiplier = 2.0 if balance > config['RFMD_THRESHOLDS']['high_balance'] * 2 else 1.5
    base_cashback = 0.02 * tier_multiplier * (restaurant_spend + jewelry_spend + cosmetics_spend)
    
    # Повышенный кэшбэк 4% на ювелирку, косметику, рестораны
    premium_cashback = 0.04 * (jewelry_spend + cosmetics_spend + restaurant_spend)
    
    # Экономия на комиссиях (ATM, переводы) - условная выгода
    saved_fees = min(balance * 0.0005, 3000)  # 0.05% от баланса, максимум 3000
    
    return base_cashback + premium_cashback + saved_fees


def _calc_credit_benefit(restaurant_spend: float, shopping_spend: float, entertainment_spend: float, 
                        online_spend: float, top_categories_spend: float, balance: float, config: Dict[str, Any]) -> float:
    """Расчет выгоды для кредитной карты согласно ТЗ."""
    total_spend = restaurant_spend + shopping_spend + entertainment_spend
    
    # До 10% в 3 "любимых" категориях + 10% на онлайн-услуги
    top_categories_cashback = 0.10 * top_categories_spend
    online_cashback = 0.10 * online_spend
    
    # Льготный период до 2 месяцев - экономия процентов
    # Предполагаем 18% годовых * 60/365 дней
    interest_saved = total_spend * 0.18 * (60/365)
    
    # Рассрочка 3-24 месяца - условная выгода
    installment_benefit = min(total_spend * 0.05, 2000)  # 5% от трат, максимум 2000
    
    return top_categories_cashback + online_cashback + interest_saved + installment_benefit


def _calc_fx_benefit(fx_buy: float, fx_sell: float, config: Dict[str, Any]) -> float:
    """Расчет выгоды для обмена валют."""
    total_fx_volume = fx_buy + fx_sell
    
    if total_fx_volume < config['RFMD_THRESHOLDS']['fx_volume_threshold']:
        return 0
    
    # Выгодный курс - экономия 0.5% от суммы операций
    fx_savings = total_fx_volume * 0.005
    
    # Бонус за большой объем
    if total_fx_volume > config['RFMD_THRESHOLDS']['fx_volume_threshold'] * 2:
        fx_savings *= 1.5
    
    return fx_savings


def _calc_cash_credit_benefit(transfer_out: float, transfer_in: float, balance: float, config: Dict[str, Any]) -> float:
    """Расчет выгоды для кредита наличными."""
    # Рассчитываем потребность в дополнительных средствах
    cash_flow_gap = transfer_out - transfer_in
    
    if cash_flow_gap <= 0:
        return 0  # Нет потребности в кредите
    
    # Выгода от низкой процентной ставки vs альтернативы
    # Предполагаем наш кредит 12% vs рыночный 18%
    interest_savings = cash_flow_gap * 0.06 * (12/12)  # Экономия 6% годовых
    
    # Учитываем кредитоспособность через баланс
    creditworthiness_bonus = min(balance * 0.0001, 1000)
    
    return interest_savings + creditworthiness_bonus


def _calc_multi_deposit_benefit(balance: float, fx_buy: float, fx_sell: float, config: Dict[str, Any]) -> float:
    """Расчет выгоды для мультивалютного депозита."""
    if balance < config['RFMD_THRESHOLDS']['high_balance'] * 0.5:
        return 0  # Минимальный баланс для депозита
    
    # Высокая доходность для валютного депозита
    base_return = balance * 0.08 * (12/12)  # 8% годовых
    
    # Бонус за валютную активность
    fx_bonus = 0
    if fx_buy + fx_sell > config['RFMD_THRESHOLDS']['fx_volume_threshold']:
        fx_bonus = base_return * 0.2  # +20% к доходности
    
    return base_return + fx_bonus


def _calc_savings_deposit_benefit(balance: float, transfer_in: float, transfer_out: float, config: Dict[str, Any]) -> float:
    """Расчет выгоды для сберегательного депозита."""
    if balance < 50000:  # Минимальная сумма
        return 0
    
    # Консервативный клиент (больше поступлений, чем трат)
    savings_potential = max(0, transfer_in - transfer_out)
    deposit_amount = min(balance, balance + savings_potential * 0.5)
    
    # 6% годовых
    return deposit_amount * 0.06 * (12/12)


def _calc_accumulation_deposit_benefit(balance: float, transfer_in: float, config: Dict[str, Any]) -> float:
    """Расчет выгоды для накопительного депозита."""
    if transfer_in < 10000:  # Минимальные регулярные поступления
        return 0
    
    # Прогрессивная ставка для накопительного депозита
    annual_accumulation = transfer_in * 12
    progressive_rate = 0.05 + min(annual_accumulation / 1000000, 0.03)  # 5-8% в зависимости от суммы
    
    return annual_accumulation * progressive_rate


def _calc_investment_benefit(balance: float, transfer_in: float, transfer_out: float, config: Dict[str, Any]) -> float:
    """Расчет выгоды для инвестиций."""
    if balance < config['RFMD_THRESHOLDS']['high_balance']:
        return 0  # Инвестиции для состоятельных клиентов
    
    # Свободные средства для инвестирования
    free_cash = balance + transfer_in - transfer_out
    investment_amount = max(0, min(free_cash * 0.3, balance * 0.5))  # До 30% от оборота или 50% от баланса
    
    # Потенциальная доходность 12% годовых (с учетом рисков)
    expected_return = investment_amount * 0.12 * (12/12)
    
    # Бонус за большую сумму инвестирования
    if investment_amount > config['RFMD_THRESHOLDS']['high_balance']:
        expected_return *= 1.2
    
    return expected_return


def _calc_gold_benefit(balance: float, fx_buy: float, fx_sell: float, config: Dict[str, Any]) -> float:
    """Расчет выгоды для золотых слитков."""
    if balance < config['RFMD_THRESHOLDS']['high_balance'] * 0.5:
        return 0
    
    # Альтернативная инвестиция для клиентов с валютной активностью
    fx_activity = fx_buy + fx_sell
    
    # Потенциальная доходность от золота 5% годовых + защита от инфляции
    gold_investment = min(balance * 0.2, 500000)  # До 20% портфеля в золото
    base_return = gold_investment * 0.05
    
    # Бонус за валютную диверсификацию
    if fx_activity > config['RFMD_THRESHOLDS']['fx_volume_threshold']:
        base_return *= 1.3
    
    return base_return


def apply_rules_and_select_best(benefits: Dict[str, float], client_features: pd.Series, config: Dict[str, Any]) -> str:
    """
    Применяет бизнес-правила и выбирает лучший продукт для клиента.
    
    Args:
        benefits: Словарь с выгодами для каждого продукта
        client_features: Характеристики клиента
        config: Конфигурация
        
    Returns:
        Название рекомендуемого продукта
    """
    try:
        # Загружаем правила из JSON
        rules = _load_business_rules(config['RULES_JSON_PATH'])
        
        # Подготавливаем переменные для правил
        variables = _prepare_rule_variables(client_features, benefits)
        
        # Применяем правила
        recommended_product = _apply_business_rules(rules, variables)
        
        if recommended_product:
            logger.debug(f"Продукт выбран по правилам: {recommended_product}")
            return recommended_product
            
    except Exception as e:
        logger.warning(f"Ошибка при применении правил: {e}")
    
    # Если правила не сработали, выбираем продукт с максимальной выгодой
    if not benefits:
        return 'Депозит сберегательный'  # Продукт по умолчанию
    
    best_product = max(benefits, key=benefits.get)
    
    # Tie-breaker: если несколько продуктов с одинаковой выгодой
    max_benefit = benefits[best_product]
    tied_products = [product for product, benefit in benefits.items() if benefit == max_benefit]
    
    if len(tied_products) > 1:
        best_product = _resolve_tie(tied_products, client_features, config)
    
    logger.debug(f"Продукт выбран по максимальной выгоде: {best_product} ({max_benefit:.0f})")
    return best_product


def _load_business_rules(rules_path: str) -> list:
    """Загружает бизнес-правила из JSON файла."""
    try:
        with open(rules_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Файл правил не найден: {rules_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка при парсинге JSON правил: {e}")
        return []


def _prepare_rule_variables(client_features: pd.Series, benefits: Dict[str, float]) -> Dict[str, Any]:
    """Подготавливает переменные для применения правил."""
    variables = {
        # Характеристики клиента
        'balance': client_features.get('avg_monthly_balance_KZT', 0),
        'age': client_features.get('age', 0),
        'status': client_features.get('status', ''),
        'city': client_features.get('city', ''),
        
        # Траты по категориям
        'travel_spend': client_features.get('spend_Путешествия', 0),
        'taxi_spend': client_features.get('spend_Такси', 0),
        'restaurant_spend': client_features.get('spend_Ресторан', 0),
        'shopping_spend': client_features.get('spend_Покупки', 0),
        'entertainment_spend': client_features.get('spend_Развлечения', 0),
        
        # Переводы
        'fx_buy': client_features.get('transfer_fx_buy', 0),
        'fx_sell': client_features.get('transfer_fx_sell', 0),
        'transfer_out': client_features.get('transfer_transfer_out', 0),
        'transfer_in': client_features.get('transfer_transfer_in', 0),
        
        # RFM-D метрики
        'recency': client_features.get('recency', 0.5),
        'frequency': client_features.get('frequency', 0),
        'monetary': client_features.get('monetary', 0),
        'diversity': client_features.get('diversity', 0),
        'cluster': client_features.get('cluster', 0),
        
        # Выгоды
        **{f'benefit_{product.lower().replace(" ", "_")}': benefit for product, benefit in benefits.items()}
    }
    
    return variables


def _apply_business_rules(rules: list, variables: Dict[str, Any]) -> str:
    """Применяет бизнес-правила и возвращает рекомендуемый продукт."""
    for rule in rules:
        try:
            if _evaluate_rule_condition(rule.get('condition', ''), variables):
                return rule.get('product', '')
        except Exception as e:
            logger.warning(f"Ошибка при оценке правила: {e}")
            continue
    
    return ''


def _evaluate_rule_condition(condition: str, variables: Dict[str, Any]) -> bool:
    """Оценивает условие правила."""
    if not condition:
        return False
    
    try:
        # Простая реализация условий
        # В продакшене следует использовать более безопасную библиотеку правил
        return eval(condition, {"__builtins__": {}}, variables)
    except Exception:
        return False


def _resolve_tie(tied_products: list, client_features: pd.Series, config: Dict[str, Any]) -> str:
    """Разрешает ничью между продуктами с одинаковой выгодой."""
    cluster = client_features.get('cluster', 0)
    
    # Приоритеты по кластерам
    cluster_preferences = {
        0: ['Карта для путешествий', 'Премиальная карта'],  # Путешественники
        1: ['Инвестиции', 'Депозит мультивалютный'],        # Инвесторы
        2: ['Кредитная карта', 'Кредит наличными'],         # Активные заемщики
        3: ['Депозит сберегательный', 'Депозит накопительный']  # Консерваторы
    }
    
    preferences = cluster_preferences.get(cluster, tied_products)
    
    # Выбираем первый продукт из предпочтений, который есть в tied_products
    for preferred_product in preferences:
        if preferred_product in tied_products:
            return preferred_product
    
    # Если ничего не найдено, возвращаем первый
    return tied_products[0]


def test_calculate_all_benefits():
    """Тест функции расчета выгод."""
    try:
        # Создаем тестового клиента
        test_client = pd.Series({
            'client_code': 1,
            'avg_monthly_balance_KZT': 500000,
            'spend_Путешествия': 30000,
            'spend_Такси': 10000,
            'spend_Ресторан': 20000,
            'spend_Покупки': 25000,
            'transfer_fx_buy': 15000,
            'transfer_fx_sell': 5000,
            'transfer_transfer_out': 50000,
            'transfer_transfer_in': 80000
        })
        
        test_config = CONFIG.__dict__
        benefits = calculate_all_benefits(test_client, test_config)
        
        # Проверяем, что выгоды рассчитаны для всех продуктов
        expected_products = CONFIG.PRODUCTS
        for product in expected_products:
            assert product in benefits, f"Выгода не рассчитана для {product}"
            assert isinstance(benefits[product], (int, float)), f"Выгода для {product} не является числом"
            assert benefits[product] >= 0, f"Выгода для {product} отрицательная"
        
        print("Тест calculate_all_benefits пройден успешно!")
        print(f"Пример выгод: {dict(list(benefits.items())[:3])}")
        
    except Exception as e:
        print(f"Тест calculate_all_benefits не пройден: {e}")


if __name__ == "__main__":
    test_calculate_all_benefits()