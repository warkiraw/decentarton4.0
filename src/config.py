"""
Модуль конфигурации системы персонализации банковских предложений.
Содержит все константы и параметры для избежания hardcoding в коде.
"""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class Config:
    """Конфигурация системы персонализации банковских предложений."""
    
    DATA_PATHS: Dict[str, str] = None
    CURRENCY_RATES: Dict[str, float] = None
    RFMD_THRESHOLDS: Dict[str, float] = None
    CLUSTER_PARAMS: Dict[str, Any] = None
    PROPENSITY_PARAMS: Dict[str, Any] = None
    PRODUCTS: List[str] = None
    RULES_JSON_PATH: str = 'rules.json'
    TEMPLATE_DIR: str = '../templates/'
    MAX_PUSH_LEN: int = 220  # Оптимальная длина для push quality scoring
    AB_TEST_FLAG: str = 'B'
    
    def __post_init__(self):
        """Инициализация значений по умолчанию."""
        if self.DATA_PATHS is None:
            self.DATA_PATHS = {
                'clients': '../data/clients.csv',
                'transactions': '../data/transactions.csv',
                'transfers': '../data/transfers.csv',
                'output': '../data/output.csv'
            }
            
        if self.CURRENCY_RATES is None:
            # Hardcoded rates to KZT (as of 2025-09-13)
            self.CURRENCY_RATES = {
                'USD': 480.0,
                'EUR': 530.0,
                'KZT': 1.0
            }
            
        if self.RFMD_THRESHOLDS is None:
            # Пороги для метрик
            self.RFMD_THRESHOLDS = {
                'high_balance': 1000000.0,  # Для tier в премиальной карте
                'fx_volume_threshold': 10000.0  # Для FX-активности
            }
            
        if self.CLUSTER_PARAMS is None:
            self.CLUSTER_PARAMS = {
                'n_clusters': 4,
                'random_state': 42
            }
            
        if self.PROPENSITY_PARAMS is None:
            self.PROPENSITY_PARAMS = {
                'train_split': 0.8,
                'threshold_spend': 50000.0  # Для синтетического таргета
            }
            
        if self.PRODUCTS is None:
            # Точные названия продуктов согласно ТЗ
            self.PRODUCTS = [
                'Карта для путешествий',
                'Премиальная карта',
                'Кредитная карта',
                'Обмен валют',
                'Кредит наличными',
                'Депозит мультивалютный',
                'Депозит сберегательный',
                'Депозит накопительный',
                'Инвестиции',
                'Золотые слитки'
            ]
            
        # Категории транзакций согласно ТЗ
        self.TRANSACTION_CATEGORIES = [
            'Одежда и обувь', 'Продукты питания', 'Кафе и рестораны', 'Медицина', 'Авто', 'Спорт',
            'Развлечения', 'АЗС', 'Кино', 'Питомцы', 'Книги', 'Цветы', 'Едим дома', 'Смотрим дома',
            'Играем дома', 'Косметика и Парфюмерия', 'Подарки', 'Ремонт дома', 'Мебель', 'Спа и массаж',
            'Ювелирные украшения', 'Такси', 'Отели', 'Путешествия'
        ]
        
        # Типы переводов согласно ТЗ
        self.TRANSFER_TYPES = [
            'salary_in', 'stipend_in', 'family_in', 'cashback_in', 'refund_in', 'card_in',
            'p2p_out', 'card_out', 'atm_withdrawal', 'utilities_out', 'loan_payment_out',
            'cc_repayment_out', 'installment_payment_out', 'fx_buy', 'fx_sell', 'invest_out',
            'invest_in', 'deposit_topup_out', 'deposit_fx_topup_out', 'deposit_fx_withdraw_in',
            'gold_buy_out', 'gold_sell_in'
        ]
        
        # Статусы клиентов согласно ТЗ
        self.CLIENT_STATUSES = {
            'зп': 'Зарплатный клиент',
            'студент': 'Студент', 
            'премиум': 'Премиальный клиент',
            'стандарт': 'Стандартный клиент'
        }
        
        # Фокусные категории для travel карты
        self.TRAVEL_CATEGORIES = ['Путешествия', 'Отели', 'Такси']
        
        # Фокусные категории для premium карты
        self.PREMIUM_CATEGORIES = ['Ювелирные украшения', 'Косметика и Парфюмерия', 'Кафе и рестораны']
        
        # Онлайн категории для credit карты
        self.ONLINE_CATEGORIES = ['Едим дома', 'Смотрим дома', 'Играем дома']


# Глобальная конфигурация
CONFIG = Config()