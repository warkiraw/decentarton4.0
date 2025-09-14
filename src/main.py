"""
Главный модуль системы персонализации банковских предложений.
Оркестрирует весь пайплайн от загрузки данных до генерации рекомендаций.
"""

import pandas as pd
import logging
import sys
import time
from typing import Dict, Any

# Импорты модулей системы
from data_processing import load_datasets, preprocess_and_merge
from feature_engineering import create_rfmd_features, add_cluster_labels, add_propensity_scores
from production_recommendation_engine import calculate_all_benefits, apply_rules_and_select_best
from nlg_module import generate_push_text
from config import CONFIG

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('personalization_system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Главная функция, выполняющая полный пайплайн персонализации.
    """
    start_time = time.time()
    
    try:
        logger.info("=" * 50)
        logger.info("ЗАПУСК СИСТЕМЫ ПЕРСОНАЛИЗАЦИИ БАНКОВСКИХ ПРЕДЛОЖЕНИЙ")
        logger.info("=" * 50)
        
        # 1. Загрузка данных
        logger.info("Шаг 1: Загрузка данных из CSV файлов")
        dataframes = load_datasets(CONFIG.__dict__)
        logger.info(f"Загружено наборов данных: {len(dataframes)}")
        
        # 2. Предобработка и объединение данных
        logger.info("Шаг 2: Предобработка и объединение данных")
        df = preprocess_and_merge(dataframes, CONFIG.__dict__)
        logger.info(f"Количество клиентов после предобработки: {len(df)}")
        
        if len(df) == 0:
            logger.error("Нет данных для обработки после предобработки")
            return
        
        # 3. Создание RFM-D признаков
        logger.info("Шаг 3: Создание RFM-D признаков")
        df = create_rfmd_features(df)
        logger.info("RFM-D признаки успешно созданы")
        
        # 4. Кластеризация клиентов
        logger.info("Шаг 4: Кластеризация клиентов")
        df = add_cluster_labels(df, CONFIG.__dict__)
        cluster_distribution = df['cluster'].value_counts().sort_index().to_dict()
        logger.info(f"Распределение по кластерам: {cluster_distribution}")
        
        # 5. Добавление propensity scores (если включено)
        if CONFIG.AB_TEST_FLAG == 'B':
            logger.info("Шаг 5: Создание propensity scores (A/B тест группа B)")
            df = add_propensity_scores(df, CONFIG.__dict__)
            logger.info("Propensity scores успешно созданы")
        else:
            logger.info("Шаг 5: Пропущен (A/B тест группа A)")
        
        # 6. Генерация рекомендаций для каждого клиента
        logger.info("Шаг 6: Генерация персонализированных рекомендаций")
        results = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                # Расчет выгод для всех продуктов
                benefits = calculate_all_benefits(row, CONFIG.__dict__)
                
                # Выбор лучшего продукта
                recommended_product = apply_rules_and_select_best(benefits, row, CONFIG.__dict__)
                
                # Генерация персонализированного уведомления
                push_notification = generate_push_text(row, recommended_product, CONFIG.__dict__)
                
                # Добавление результата
                results.append({
                    'client_code': row['client_code'],
                    'product': recommended_product,
                    'push_notification': push_notification,
                    # Дополнительные колонки для внутреннего анализа (не экспортируются в финальный CSV)
                    '_benefit': benefits.get(recommended_product, 0),
                    '_cluster': row.get('cluster', 0),
                    '_balance': row.get('avg_monthly_balance_KZT', 0)
                })
                
                # Логирование прогресса
                if (idx + 1) % 100 == 0:
                    logger.info(f"Обработано клиентов: {idx + 1}/{len(df)}")
                    
            except Exception as e:
                logger.error(f"Ошибка при обработке клиента {row.get('client_code', 'unknown')}: {e}")
                continue
        
        # 7. Сохранение результатов
        logger.info("Шаг 7: Сохранение результатов")
        if results:
            final_df = pd.DataFrame(results)
            
            # Создаем финальный CSV только с тремя колонками согласно ТЗ
            tz_compliant_df = final_df[['client_code', 'product', 'push_notification']].copy()
            
            # Сохранение в CSV согласно ТЗ формату
            tz_compliant_df.to_csv(
                CONFIG.DATA_PATHS['output'], 
                index=False, 
                quoting=1,  # Экранирование кавычек
                encoding='utf-8'
            )
            
            # Сохраняем расширенный файл для анализа
            extended_output_path = CONFIG.DATA_PATHS['output'].replace('.csv', '_extended.csv')
            final_df.to_csv(
                extended_output_path,
                index=False,
                quoting=1,
                encoding='utf-8'
            )
            
            logger.info(f"Результаты сохранены в {CONFIG.DATA_PATHS['output']}")
            logger.info(f"Обработано клиентов: {len(final_df)}")
            
            # Статистика по продуктам
            product_stats = final_df['product'].value_counts()
            logger.info("Статистика рекомендованных продуктов:")
            for product, count in product_stats.items():
                logger.info(f"  {product}: {count} ({count/len(final_df)*100:.1f}%)")
            
            # Статистика по кластерам
            cluster_stats = final_df['_cluster'].value_counts().sort_index()
            logger.info("Статистика по кластерам:")
            for cluster, count in cluster_stats.items():
                logger.info(f"  Кластер {cluster}: {count} клиентов")
            
            # Примеры рекомендаций
            logger.info("Примеры сгенерированных рекомендаций:")
            for i, row in final_df.head(3).iterrows():
                logger.info(f"  Клиент {row['client_code']}: {row['product']}")
                logger.info(f"    Текст: {row['push_notification']}")
                logger.info(f"    Выгода: {row['_benefit']:.0f} ₸")
        
        else:
            logger.error("Не удалось создать рекомендации ни для одного клиента")
            return
        
        # Завершение
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("=" * 50)
        logger.info(f"СИСТЕМА ПЕРСОНАЛИЗАЦИИ ЗАВЕРШЕНА УСПЕШНО")
        logger.info(f"Время выполнения: {execution_time:.2f} секунд")
        logger.info(f"Обработано клиентов: {len(final_df)}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Критическая ошибка в главном пайплайне: {e}")
        logger.exception("Детали ошибки:")
        sys.exit(1)


def validate_environment() -> bool:
    """
    Проверяет готовность окружения для запуска системы.
    
    Returns:
        True если окружение готово, False иначе
    """
    logger.info("Проверка готовности окружения...")
    
    issues = []
    
    # Проверка наличия файлов данных
    for name, path in CONFIG.DATA_PATHS.items():
        if name != 'output':  # output файл создается системой
            try:
                pd.read_csv(path, nrows=1)
                logger.info(f"✓ Файл {name} найден: {path}")
            except FileNotFoundError:
                issues.append(f"✗ Файл {name} не найден: {path}")
            except Exception as e:
                issues.append(f"✗ Ошибка при чтении {name}: {e}")
    
    # Проверка директории шаблонов
    import os
    if os.path.exists(CONFIG.TEMPLATE_DIR):
        logger.info(f"✓ Директория шаблонов найдена: {CONFIG.TEMPLATE_DIR}")
    else:
        issues.append(f"✗ Директория шаблонов не найдена: {CONFIG.TEMPLATE_DIR}")
    
    # Проверка зависимостей
    required_modules = ['pandas', 'sklearn', 'jinja2']
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ Модуль {module} доступен")
        except ImportError:
            issues.append(f"✗ Модуль {module} не установлен")
    
    if issues:
        logger.error("Обнаружены проблемы в окружении:")
        for issue in issues:
            logger.error(f"  {issue}")
        return False
    
    logger.info("✓ Окружение готово к работе")
    return True


def create_sample_data() -> None:
    """
    Создает образцы данных для тестирования системы.
    """
    logger.info("Создание образцов данных для тестирования...")
    
    # Создание образца клиентов
    clients_data = {
        'client_code': [1001, 1002, 1003, 1004, 1005],
        'name': ['Анна Петрова', 'Михаил Иванов', 'Елена Сидорова', 'Дмитрий Козлов', 'Ольга Николаева'],
        'status': ['Работающий', 'Пенсионер', 'Студент', 'Работающий', 'Работающий'],
        'age': [35, 65, 22, 45, 28],
        'city': ['Алматы', 'Нур-Султан', 'Шымкент', 'Алматы', 'Караганда'],
        'avg_monthly_balance_KZT': [500000, 1200000, 50000, 800000, 300000]
    }
    
    clients_df = pd.DataFrame(clients_data)
    clients_df.to_csv(CONFIG.DATA_PATHS['clients'], index=False, encoding='utf-8')
    
    # Создание образца транзакций
    transactions_data = {
        'client_code': [1001, 1001, 1002, 1003, 1004, 1005] * 3,
        'date': ['2025-08-15 10:30:00'] * 18,
        'category': ['Путешествия', 'Такси', 'Ресторан', 'Покупки', 'Развлечения', 'Такси'] * 3,
        'amount': [45000, 5000, 25000, 15000, 12000, 3000] * 3,
        'currency': ['KZT'] * 18
    }
    
    transactions_df = pd.DataFrame(transactions_data)
    transactions_df.to_csv(CONFIG.DATA_PATHS['transactions'], index=False, encoding='utf-8')
    
    # Создание образца переводов
    transfers_data = {
        'client_code': [1001, 1002, 1003, 1004, 1005] * 2,
        'date': ['2025-08-20 14:15:00'] * 10,
        'type': ['fx_buy', 'fx_sell', 'transfer_out', 'transfer_in', 'fx_buy'] * 2,
        'amount': [20000, 35000, 100000, 150000, 25000] * 2,
        'currency': ['USD', 'EUR', 'KZT', 'KZT', 'USD'] * 2,
        'direction': ['out', 'in', 'out', 'in', 'out'] * 2
    }
    
    transfers_df = pd.DataFrame(transfers_data)
    transfers_df.to_csv(CONFIG.DATA_PATHS['transfers'], index=False, encoding='utf-8')
    
    logger.info("Образцы данных созданы:")
    logger.info(f"  Клиенты: {CONFIG.DATA_PATHS['clients']}")
    logger.info(f"  Транзакции: {CONFIG.DATA_PATHS['transactions']}")
    logger.info(f"  Переводы: {CONFIG.DATA_PATHS['transfers']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Система персонализации банковских предложений')
    parser.add_argument('--create-sample-data', action='store_true', 
                       help='Создать образцы данных для тестирования')
    parser.add_argument('--validate-only', action='store_true',
                       help='Только проверить готовность окружения')
    
    args = parser.parse_args()
    
    if args.create_sample_data:
        create_sample_data()
        logger.info("Образцы данных созданы. Теперь можно запустить систему.")
    elif args.validate_only:
        if validate_environment():
            logger.info("Система готова к запуску")
            sys.exit(0)
        else:
            logger.error("Система не готова к запуску")
            sys.exit(1)
    else:
        if validate_environment():
            main()
        else:
            logger.error("Исправьте проблемы в окружении перед запуском")
            sys.exit(1)