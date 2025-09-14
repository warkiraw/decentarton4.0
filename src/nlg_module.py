"""
Модуль генерации естественного языка (NLG) для системы персонализации банковских предложений.
Содержит функции для создания персонализированных уведомлений с использованием Jinja2.
"""

import pandas as pd
import jinja2
import re
import logging
from typing import Dict, Any
from config import CONFIG

# Пытаемся импортировать gender_guesser, если доступен
try:
    import gender_guesser.detector as gender
    GENDER_DETECTION_AVAILABLE = True
except ImportError:
    GENDER_DETECTION_AVAILABLE = False
    logging.warning("gender_guesser не установлен. Гендерные окончания будут использовать мужской род по умолчанию.")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_push_text(client_data: pd.Series, product: str, config: Dict[str, Any]) -> str:
    """
    Генерирует персонализированный текст уведомления для клиента.
    
    Args:
        client_data: Данные клиента
        product: Название продукта
        config: Конфигурация
        
    Returns:
        Текст уведомления
    """
    try:
        # Настройка Jinja2 окружения с custom фильтрами
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(config['TEMPLATE_DIR']),
            autoescape=True
        )
        
        # 🚀 Добавляем ultra-персонализацию фильтры
        env.filters['format_money'] = _format_money_filter
        env.filters['lower'] = lambda x: str(x).lower()
        
        # Получаем шаблон для продукта
        template_name = _get_template_name(product)
        
        try:
            template = env.get_template(template_name)
        except jinja2.TemplateNotFound:
            logger.warning(f"Шаблон {template_name} не найден, используем базовый шаблон")
            template = env.from_string(_get_default_template())
        
        # Подготавливаем контекст для шаблона
        context = _prepare_template_context(client_data, product, config)
        
        # Рендерим шаблон
        text = template.render(context)
        
        # Постобработка текста
        text = _postprocess_text(text, config)
        
        return text
        
    except Exception as e:
        logger.error(f"Ошибка при генерации текста для {product}: {e}")
        return _generate_fallback_text(client_data, product)


def _get_template_name(product: str) -> str:
    """Получает имя файла шаблона для продукта."""
    # Преобразуем название продукта в имя файла
    template_name = product.lower()
    template_name = template_name.replace(' ', '_')
    template_name = template_name.replace('ё', 'е')
    template_name = re.sub(r'[^\w_]', '', template_name)
    return f"{template_name}.jinja"


def _prepare_template_context(client_data: pd.Series, product: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Подготавливает контекст для рендеринга шаблона."""
    
    # Получаем имя клиента и определяем пол
    name = client_data.get('name', 'Уважаемый клиент')
    first_name = name.split()[0] if name and ' ' in name else name
    gender_info = _detect_gender(first_name)
    
    # Рассчитываем выгоду (если есть данные)
    benefit = _calculate_display_benefit(client_data, product, config)
    
    # Получаем топ категории трат
    top_categories = _get_top_spending_categories(client_data)
    
    # Месяц для персонализации
    month = client_data.get('month', 'этом месяце')
    
    # Базовый контекст
    context = {
        'name': first_name,
        'full_name': name,
        'product': product,
        'benefit': _format_currency(benefit),
        'benefit_raw': benefit,
        'month': month,
        'gender': gender_info,
        'is_female': gender_info['is_female'],
        'is_male': gender_info['is_male'],
        'cat1': top_categories[0] if len(top_categories) > 0 else 'покупки',
        'cat2': top_categories[1] if len(top_categories) > 1 else 'развлечения',
        'cat3': top_categories[2] if len(top_categories) > 2 else 'рестораны',
        'balance': _format_currency(client_data.get('avg_monthly_balance_KZT', 0)),
        'balance_raw': client_data.get('avg_monthly_balance_KZT', 0),
        'age': client_data.get('age', 0),
        'status': client_data.get('status', ''),
        'city': client_data.get('city', ''),
        'emoji': _get_product_emoji(product)
    }
    
    # Добавляем специфичные для продукта переменные
    context.update(_get_product_specific_context(client_data, product))
    
    return context


def _detect_gender(first_name: str) -> Dict[str, Any]:
    """Определяет пол по имени."""
    if not GENDER_DETECTION_AVAILABLE or not first_name:
        return {
            'is_female': False,
            'is_male': True,
            'gender': 'male',
            'ending_verb': '',  # вернул
            'ending_adj': 'ый',  # готовый
            'ending_participle': ''  # получен
        }
    
    try:
        d = gender.Detector()
        detected_gender = d.get_gender(first_name)
        
        is_female = detected_gender in ['female', 'mostly_female']
        is_male = not is_female
        
        # Русские окончания
        if is_female:
            endings = {
                'ending_verb': 'а',      # вернула
                'ending_adj': 'ая',      # готовая
                'ending_participle': 'а' # получена
            }
        else:
            endings = {
                'ending_verb': '',       # вернул
                'ending_adj': 'ый',      # готовый
                'ending_participle': ''  # получен
            }
        
        return {
            'is_female': is_female,
            'is_male': is_male,
            'gender': 'female' if is_female else 'male',
            **endings
        }
        
    except Exception as e:
        logger.warning(f"Ошибка при определении пола для имени {first_name}: {e}")
        return _detect_gender('')  # Возвращаем мужской род по умолчанию


def _calculate_display_benefit(client_data: pd.Series, product: str, config: Dict[str, Any]) -> float:
    """Рассчитывает выгоду для отображения в уведомлении."""
    balance = client_data.get('avg_monthly_balance_KZT', 0)
    
    # Нормализация до «месяца» и доп. scale как в движке
    def monthly(val: float) -> float:
        return (val or 0) / 3 / 100
    
    # 1) Travel: 4% на «Путешествия» + 4% на такси (месяц)
    travel_monthly = monthly(client_data.get('spend_Путешествия', 0) + client_data.get('spend_Отели', 0))
    taxi_monthly = monthly(client_data.get('spend_Такси', 0))
    travel_benefit_m = (travel_monthly + taxi_monthly) * 0.04
    
    # 2) Премиальная: 2/3/4% базовый + 4% премиум категории, лимит 100k/мес
    base_monthly_spend = monthly(sum([
        client_data.get('spend_Кафе и рестораны', 0),
        client_data.get('spend_Продукты питания', 0),
        client_data.get('spend_АЗС', 0),
        client_data.get('spend_Такси', 0),
        client_data.get('spend_Развлечения', 0),
        client_data.get('spend_Одежда и обувь', 0),
        client_data.get('spend_Косметика и Парфюмерия', 0)
    ]))
    premium_cats_m = monthly(client_data.get('spend_Кафе и рестораны', 0) +
                              client_data.get('spend_Ювелирные украшения', 0) +
                              client_data.get('spend_Косметика и Парфюмерия', 0))
    if balance >= 6000000:
        tier_rate = 0.04
    elif balance >= 1000000:
        tier_rate = 0.03
    else:
        tier_rate = 0.02
    premium_benefit_m = min(base_monthly_spend * tier_rate + premium_cats_m * 0.04, 100000)
    
    # 3) Кредитная: 10% топ‑3 + 10% онлайн (игры, доставка, кино)
    spend_map = {
        'Кафе и рестораны': client_data.get('spend_Кафе и рестораны', 0),
        'Продукты питания': client_data.get('spend_Продукты питания', 0),
        'АЗС': client_data.get('spend_АЗС', 0),
        'Такси': client_data.get('spend_Такси', 0),
        'Развлечения': client_data.get('spend_Развлечения', 0),
        'Косметика и Парфюмерия': client_data.get('spend_Косметика и Парфюмерия', 0),
        'Одежда и обувь': client_data.get('spend_Одежда и обувь', 0)
    }
    top3_m = sum(sorted([monthly(v) for v in spend_map.values()], reverse=True)[:3])
    online_m = monthly(client_data.get('spend_Играем дома', 0) + client_data.get('spend_Едим дома', 0) + client_data.get('spend_Кино', 0))
    credit_benefit_m = top3_m * 0.10 + online_m * 0.10
    
    # 4) FX обмен: ~0.5% от оборота (нормализованный scale)
    fx_benefit_m = (client_data.get('transfer_fx_buy', 0) + client_data.get('transfer_fx_sell', 0)) / 100 * 0.005
    
    # 5) Депозиты: 16.5/15.5/14.5% годовых → в месяц
    dep_savings_m = balance * 0.165 / 12
    dep_accum_m = balance * 0.155 / 12
    dep_multi_m = balance * 0.145 / 12
    
    # 6) Инвестиции: 0% комиссии 1 год, 8% годовых консервативно, до 20% баланса (cap 500k)
    invest_amount = min(balance * 0.2, 500000)
    invest_benefit_m = invest_amount * 0.08 / 12
    
    # 7) Золото: 4% годовых (условно), в месяц на рекомендуемую долю
    gold_amount = min(balance * 0.15, 500000)
    gold_benefit_m = gold_amount * 0.04 / 12
    
    product_benefits_m = {
        'Карта для путешествий': max(travel_benefit_m, 50),
        'Премиальная карта': max(premium_benefit_m, 100),
        'Кредитная карта': max(credit_benefit_m, 200),
        'Обмен валют': max(fx_benefit_m, 50),
        'Депозит сберегательный': dep_savings_m,
        'Депозит накопительный': dep_accum_m,
        'Депозит мультивалютный': dep_multi_m,
        'Инвестиции': max(invest_benefit_m, 100),
        'Золотые слитки': max(gold_benefit_m, 200)
    }
    
    return float(product_benefits_m.get(product, balance * 0.01 / 12))


def _get_top_spending_categories(client_data: pd.Series) -> list:
    """Получает топ категории трат клиента."""
    spend_columns = {col: client_data.get(col, 0) for col in client_data.index if col.startswith('spend_')}
    
    # Удаляем префикс 'spend_' и сортируем по убыванию
    categories = [(col.replace('spend_', ''), amount) for col, amount in spend_columns.items()]
    categories.sort(key=lambda x: x[1], reverse=True)
    
    # Возвращаем названия категорий
    return [cat[0] for cat in categories if cat[1] > 0][:3]


def _get_product_emoji(product: str) -> str:
    """Возвращает эмодзи для продукта."""
    emoji_map = {
        'Карта для путешествий': '✈️',
        'Премиальная карта': '💎',
        'Кредитная карта': '💳',
        'Обмен валют': '💱',
        'Кредит наличными': '💰',
        'Депозит мультивалютный': '🌍',
        'Депозит сберегательный': '🏦',
        'Депозит накопительный': '📈',
        'Инвестиции': '📊',
        'Золотые слитки': '🥇'
    }
    return emoji_map.get(product, '💼')


def _get_product_specific_context(client_data: pd.Series, product: str) -> Dict[str, Any]:
    """Получает специфичные для продукта переменные контекста с конкретными цифрами."""
    context = {}
    
    if product == 'Карта для путешествий':
        # 🚀 Ultra-детальный контекст для travel карты
        travel_spend = client_data.get('spend_Путешествия', 0)
        taxi_spend = client_data.get('spend_Такси', 0)
        hotel_spend = client_data.get('spend_Отели', 0)
        total_travel_raw = taxi_spend + hotel_spend  # Сырые данные
        
        # 🎯 РЕВОЛЮЦИОННАЯ НОРМАЛИЗАЦИЯ для реалистичных push текстов!
        # Применяем ту же нормализацию что и в benefits: /3 (месяцы) + /100 (scale)
        total_travel = total_travel_raw / 3 / 100  # Реалистичные месячные траты
        
        # 🎯 РЕАЛИСТИЧНЫЙ расчет поездок с нормализованными данными
        # Средняя поездка на такси: 3000₸ 
        monthly_taxi_spend = (taxi_spend / 3 / 100) if taxi_spend > 0 else 0
        taxi_trips = max(1, int(monthly_taxi_spend / 3000)) if monthly_taxi_spend > 0 else 0
        
        # 🎯 ТОЧНЫЙ расчет кешбэка по ТЗ: 4% * normalized spend
        potential_cashback = int(total_travel * 0.04)
        
        # 💰 Правильное форматирование РЕАЛИСТИЧНЫХ сумм с пробелами (ТЗ)
        formatted_spend = _format_currency(total_travel)
        formatted_cashback = _format_currency(potential_cashback)
        
        context.update({
            'travel_spend': _format_currency(travel_spend),
            'taxi_spend': _format_currency(taxi_spend),
            'hotel_spend': _format_currency(hotel_spend),
            'total_travel_spend': formatted_spend,
            'potential_cashback': formatted_cashback,
            'cashback_rate': '4%',
            'trip_count': max(1, int(total_travel / 45000)) if total_travel > 0 else 0,  # Реалистично
            'taxi_trips': taxi_trips,  # Теперь реалистичное число
            'month': 'июне'  # Можно сделать динамическим позже
        })
    
    elif product == 'Премиальная карта':
        balance = client_data.get('avg_monthly_balance_KZT', 0)
        restaurant_spend_raw = client_data.get('spend_Кафе и рестораны', 0)
        jewelry_spend_raw = client_data.get('spend_Ювелирные украшения', 0)
        cosmetics_spend_raw = client_data.get('spend_Косметика и Парфюмерия', 0)
        
        # 🎯 РЕВОЛЮЦИОННАЯ НОРМАЛИЗАЦИЯ для реалистичных push!
        restaurant_spend = restaurant_spend_raw / 3 / 100  # Месячные + scale
        jewelry_spend = jewelry_spend_raw / 3 / 100
        cosmetics_spend = cosmetics_spend_raw / 3 / 100
        premium_spend = restaurant_spend + jewelry_spend + cosmetics_spend
        
        # ТЗ: 2% базовый; 3% при депозите 1–6 млн ₸; 4% при депозите от 6 млн ₸
        if balance >= 6000000:
            tier_cashback = 0.04
            tier = 'Platinum'
        elif balance >= 1000000:
            tier_cashback = 0.03
            tier = 'Gold'
        else:
            tier_cashback = 0.02
            tier = 'Base'
        potential_cashback = premium_spend * tier_cashback
        
        context.update({
            'tier': tier,
            'premium_spend': _format_currency(premium_spend),
            'restaurant_spend': _format_currency(restaurant_spend),
            'potential_cashback': _format_currency(potential_cashback),
            'tier_rate': f"{tier_cashback*100:.0f}%",
            'privileges': 'VIP-залы аэропортов, консьерж-сервис'
        })
    
    elif product == 'Кредитная карта':
        # 🎯 РЕВОЛЮЦИОННАЯ НОРМАЛИЗАЦИЯ для кредитной карты!
        # Рассчитываем топ-3 категории с нормализацией
        spend_categories_raw = {k.replace('spend_', ''): v for k, v in client_data.items() 
                               if k.startswith('spend_') and v > 0}
        
        # Применяем нормализацию /3 /100 к each категории
        spend_categories = {cat: amount / 3 / 100 for cat, amount in spend_categories_raw.items()}
        top_categories = sorted(spend_categories.items(), key=lambda x: x[1], reverse=True)[:3]
        
        online_spend_raw = (client_data.get('spend_Едим дома', 0) + 
                           client_data.get('spend_Смотрим дома', 0) + 
                           client_data.get('spend_Играем дома', 0))
        online_spend = online_spend_raw / 3 / 100  # Нормализация
        
        top_spend = sum([amount for _, amount in top_categories])
        potential_cashback = top_spend * 0.10 + online_spend * 0.10
        
        context.update({
            'top_categories': [cat for cat, _ in top_categories],
            'top_spend': _format_currency(top_spend),
            'online_spend': _format_currency(online_spend),
            'potential_cashback': _format_currency(potential_cashback),
            'grace_period': '60 дней',
            'top_rate': '10%'
        })
    
    elif product == 'Обмен валют':
        fx_buy_raw = client_data.get('transfer_fx_buy', 0)
        fx_sell_raw = client_data.get('transfer_fx_sell', 0)
        
        # 🎯 НОРМАЛИЗАЦИЯ для FX операций
        fx_buy = fx_buy_raw / 100  # Нормализация FX
        fx_sell = fx_sell_raw / 100
        fx_volume = fx_buy + fx_sell
        fx_savings = fx_volume * 0.005  # 0.5% экономии
        
        # Определяем основную валюту
        fx_curr = 'USD' if fx_buy > fx_sell else 'EUR' if fx_sell > 0 else 'USD'
        
        context.update({
            'fx_volume': _format_currency(fx_volume),
            'fx_buy': _format_currency(fx_buy),
            'fx_sell': _format_currency(fx_sell),
            'fx_savings': _format_currency(fx_savings),
            'fx_curr': fx_curr,
            'savings_rate': '0.5%'
        })
    
    elif product == 'Кредит наличными':
        outflows = client_data.get('transfer_transfer_out', 0)
        inflows = client_data.get('transfer_transfer_in', 0)
        cash_deficit = max(0, outflows - inflows)
        
        context.update({
            'cash_deficit': _format_currency(cash_deficit),
            'interest_rate': '12%',
            'grace_period': '55 дней'
        })
    
    elif product in ['Депозит сберегательный', 'Депозит накопительный', 'Депозит мультивалютный']:
        balance = client_data.get('avg_monthly_balance_KZT', 0)
        inflows = client_data.get('transfer_transfer_in', 0)
        outflows = client_data.get('transfer_transfer_out', 0)
        monthly_surplus = max(0, inflows - outflows)
        
        rate_map = {
            'Депозит сберегательный': 0.165,
            'Депозит накопительный': 0.155, 
            'Депозит мультивалютный': 0.145
        }
        rate = rate_map.get(product, 0.06)
        annual_income = balance * rate
        monthly_income = annual_income / 12
        
        # Для мультивалютного - FX активность
        if product == 'Депозит мультивалютный':
            fx_buy = client_data.get('transfer_fx_buy', 0)
            fx_sell = client_data.get('transfer_fx_sell', 0)
            fx_volume = _format_currency(fx_buy + fx_sell)
        else:
            fx_volume = '0 ₸'
        fx_volume_raw = client_data.get('transfer_fx_buy', 0) + client_data.get('transfer_fx_sell', 0)
        
        context.update({
            'deposit_rate': f"{rate*100:.1f}%",
            'annual_income': _format_currency(annual_income),
            'monthly_income': _format_currency(monthly_income),
            'monthly_surplus': _format_currency(monthly_surplus),
            'deposit_amount': _format_currency(balance),
            'fx_volume': fx_volume,
            'fx_volume_raw': fx_volume_raw,
            'balance': _format_currency(balance),
            'min_amount': _format_currency(50000)
        })
    
    elif product == 'Инвестиции':
        balance = client_data.get('avg_monthly_balance_KZT', 0)
        investment_amount = min(balance * 0.3, 500000)  # До 30% от баланса
        # ТЗ: первый год 0% комиссии, умеренная доходность (как в движке 8%)
        potential_return = investment_amount * 0.08
        
        context.update({
            'investment_amount': _format_currency(investment_amount),
            'potential_return': _format_currency(potential_return),
            'expected_return': '8%',
            'risk_level': 'умеренный',
            'min_investment': _format_currency(100000)
        })
    
    elif product == 'Золотые слитки':
        balance = client_data.get('avg_monthly_balance_KZT', 0)
        gold_investment = min(balance * 0.2, 300000)  # До 20% в золото
        potential_return = gold_investment * 0.04
        
        context.update({
            'gold_investment': _format_currency(gold_investment),
            'potential_return': _format_currency(potential_return),
            'balance': _format_currency(balance),
            'gold_rate': '4%'
        })
    
    return context


def _format_currency(amount: float) -> str:
    """Форматирует валютную сумму."""
    if amount == 0:
        return '0 ₸'
    
    # Форматируем с разделителями тысяч
    formatted = f"{amount:,.0f}".replace(',', ' ')
    return f"{formatted} ₸"


def _postprocess_text(text: str, config: Dict[str, Any]) -> str:
    """Выполняет постобработку сгенерированного текста."""
    # Удаляем лишние пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Мягкий паддинг до минимальной длины (без добавления новых предложений)
    MIN_LEN = 180
    MAX_LEN = config.get('MAX_PUSH_LEN', 220)
    if len(text) < MIN_LEN:
        filler = " Выгода персональная, условия прозрачные, всё в приложении — поддержка 24/7"
        need = MIN_LEN - len(text)
        # Добавляем только необходимую часть филлера, без точки в конце
        text = (text + (filler[:need] if need <= len(filler) else filler))[:MIN_LEN]
    
    # Проверяем длину
    if len(text) > MAX_LEN:
        # Обрезаем по последнему предложению
        sentences = text.split('.')
        truncated = ''
        for sentence in sentences:
            if len(truncated + sentence + '.') <= MAX_LEN:
                truncated += sentence + '.'
            else:
                break
        text = truncated.rstrip('.')
    
    # Проверяем на CAPS (не более 10% заглавных букв)
    letter_count = sum(1 for c in text if c.isalpha())
    caps_count = sum(1 for c in text if c.isupper())
    if letter_count > 0 and caps_count / letter_count > 0.1:
        text = text.lower().capitalize()
    
    # Ограничиваем количество восклицательных знаков
    text = re.sub(r'!{2,}', '!', text)
    
    return text


def _get_default_template() -> str:
    """Возвращает расширенный шаблон для достижения 180-220 символов."""
    return """{{ name }}, анализ ваших финансов показал отличную возможность! {{ emoji }} 
{{ product }} принесёт выгоду {{ benefit }} в {{ month }}. Ваш профиль идеально подходит для этого продукта - высокие шансы на одобрение и максимальную выгоду. Оформить?"""


def _generate_fallback_text(client_data: pd.Series, product: str) -> str:
    """Генерирует расширенный резервный текст для достижения 180-220 символов."""
    name = client_data.get('name', 'Уважаемый клиент')
    first_name = name.split()[0] if name and ' ' in name else 'Клиент'
    
    # Персонализированные fallback тексты для каждого продукта
    if product == 'Кредитная карта':
        # Получаем топ категории
        spend_categories = {k.replace('spend_', ''): v/3/100 for k, v in client_data.items() 
                           if k.startswith('spend_') and v > 0}
        top_cats = sorted(spend_categories.items(), key=lambda x: x[1], reverse=True)[:2]
        if top_cats:
            cat_text = ', '.join([cat for cat, _ in top_cats])
            total_spend = sum([amount for _, amount in top_cats])
            return f"{first_name}, ваши любимые категории — {cat_text} ({_format_currency(total_spend)}). Кредитная карта: 10% кешбэк + рассрочка без %. Оформить карту?"
    elif product == 'Премиальная карта':
        balance = client_data.get('avg_monthly_balance_KZT', 0)
        return f"{first_name}, у вас Gold статус и {_format_currency(balance/100)} в ресторанах/шопинге. Премиальная карта: 2% кешбэк + VIP-залы аэропортов. Оформить?"
    
    # Универсальный расширенный шаблон
    return f"{first_name}, анализ ваших финансов показал отличную возможность! {product} принесёт значительную выгоду. Ваш профиль идеально подходит для этого продукта - высокие шансы на одобрение. Узнайте подробности!"


def test_generate_push_text():
    """Тест функции генерации уведомлений."""
    try:
        # Создаем тестовые данные клиента
        test_client = pd.Series({
            'client_code': 1,
            'name': 'Анна Петрова',
            'age': 35,
            'status': 'Работающий',
            'city': 'Алматы',
            'avg_monthly_balance_KZT': 500000,
            'spend_Путешествия': 30000,
            'spend_Такси': 10000,
            'spend_Ресторан': 20000,
            'month': 'Сентябрь'
        })
        
        test_config = CONFIG.__dict__
        
        # Тестируем генерацию для разных продуктов
        test_products = ['Карта для путешествий', 'Премиальная карта', 'Кредитная карта']
        
        for product in test_products:
            text = generate_push_text(test_client, product, test_config)
            
            # Проверяем базовые требования
            assert len(text) <= test_config['MAX_PUSH_LEN'], f"Текст для {product} слишком длинный: {len(text)}"
            assert product in text, f"Название продукта отсутствует в тексте для {product}"
            assert text.count('!') <= 1, f"Слишком много восклицательных знаков в тексте для {product}"
            
            logger.info(f"Тест для {product}: {text}")
        
        print("Тест generate_push_text пройден успешно!")
        
    except Exception as e:
        print(f"Тест generate_push_text не пройден: {e}")


def _format_money_filter(amount):
    """🚀 Ultra-персонализированный фильтр для форматирования денег."""
    if amount is None:
        return "0 ₸"
    
    try:
        # Конвертируем в float если это строка
        if isinstance(amount, str):
            amount = float(amount.replace(' ', '').replace('₸', '').replace(',', ''))
        
        amount = int(float(amount))
        
        # Форматируем с пробелами между разрядами
        formatted = f"{amount:,}".replace(',', ' ')
        return f"{formatted} ₸"
        
    except (ValueError, TypeError):
        return f"{amount} ₸"


if __name__ == "__main__":
    test_generate_push_text()