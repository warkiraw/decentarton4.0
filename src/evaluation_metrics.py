"""
Скрипт для оценки системы персонализации согласно метрикам ТЗ.
Рассчитывает точность продукта (до 20 баллов) и качество пуша (до 20 баллов).
"""

import pandas as pd
import re
import logging
from typing import Dict, List, Tuple
from production_recommendation_engine import calculate_all_benefits
from config import CONFIG

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_system_performance(output_file: str, clients_data_file: str) -> Dict[str, float]:
    """
    Оценивает систему по метрикам ТЗ.
    
    Args:
        output_file: Путь к файлу с результатами системы
        clients_data_file: Путь к файлу с данными клиентов для расчета Top-4
        
    Returns:
        Словарь с метриками оценки
    """
    logger.info("Начало оценки системы персонализации")
    
    # КРИТИЧЕСКИЙ ФИКС: Воспроизводим полный feature engineering pipeline как в main.py!
    try:
        from data_processing import load_datasets, preprocess_and_merge
        from feature_engineering import create_rfmd_features, add_cluster_labels, add_propensity_scores
        
        # Воспроизводим точно тот же pipeline
        datasets = load_datasets(CONFIG.__dict__)
        enriched_df = preprocess_and_merge(datasets, CONFIG.__dict__)
        enriched_df = create_rfmd_features(enriched_df)
        enriched_df = add_cluster_labels(enriched_df, CONFIG.__dict__)
        enriched_df = add_propensity_scores(enriched_df, CONFIG.__dict__)
        logger.info(f"Feature engineering завершен: {len(enriched_df)} клиентов")
        
        # Загружаем результаты для анализа
        results_df = pd.read_csv(output_file)
        analysis_df = results_df
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return {}
    
    # 1. Метрика точности продукта (до 20 баллов)
    logger.info("Расчет метрики точности продукта")
    product_accuracy_scores = []
    
    for _, row in analysis_df.iterrows():
        client_code = row['client_code']
        recommended_product = row['product']
        
        # ФИКС: Находим обогащенные данные клиента с полными features
        client_features = enriched_df[enriched_df['client_code'] == client_code]
        if client_features.empty:
            logger.warning(f"Данные клиента {client_code} не найдены в enriched_df")
            continue
            
        client_features = client_features.iloc[0]
        
        # КРИТИЧЕСКИЙ ФИКС: Используем нормализованные benefits как в production!
        try:
            from production_recommendation_engine import ProductionRecommendationEngine
            production_engine = ProductionRecommendationEngine(CONFIG.__dict__)
            all_benefits = production_engine.calculate_all_benefits_tz_compliant(client_features)
            
            # Сортируем продукты по выгоде (Top-4)
            sorted_products = sorted(all_benefits.items(), key=lambda x: x[1], reverse=True)
            top_4_products = [product for product, _ in sorted_products[:4]]
            
            # Определяем позицию рекомендованного продукта
            if recommended_product in top_4_products:
                position = top_4_products.index(recommended_product) + 1
                score = {1: 20, 2: 15, 3: 10, 4: 5}[position]
            else:
                score = 0
                
            product_accuracy_scores.append(score)
            
        except Exception as e:
            logger.warning(f"Ошибка при расчете выгод для клиента {client_code}: {e}")
            product_accuracy_scores.append(0)
    
    avg_product_accuracy = sum(product_accuracy_scores) / len(product_accuracy_scores) if product_accuracy_scores else 0
    
    # 2. Метрика качества пуша (до 20 баллов)
    logger.info("Расчет метрики качества пуша")
    push_quality_scores = []
    
    for _, row in results_df.iterrows():
        push_text = row['push_notification']
        client_code = row['client_code']
        product = row['product']
        
        # Оцениваем по 4 критериям × 5 баллов
        personalization_score = evaluate_personalization(push_text, client_code)
        tov_score = evaluate_tov_compliance(push_text)
        clarity_score = evaluate_clarity_and_brevity(push_text)
        format_score = evaluate_format_compliance(push_text)
        
        total_push_score = personalization_score + tov_score + clarity_score + format_score
        push_quality_scores.append(total_push_score)
    
    avg_push_quality = sum(push_quality_scores) / len(push_quality_scores) if push_quality_scores else 0
    
    # Итоговые метрики
    total_score = avg_product_accuracy + avg_push_quality
    
    metrics = {
        'product_accuracy': avg_product_accuracy,
        'push_quality': avg_push_quality,
        'total_score': total_score,
        'max_possible_score': 40,
        'percentage': (total_score / 40) * 100,
        'clients_evaluated': len(results_df)
    }
    
    # Логируем результаты
    logger.info("=" * 50)
    logger.info("РЕЗУЛЬТАТЫ ОЦЕНКИ СИСТЕМЫ ПЕРСОНАЛИЗАЦИИ")
    logger.info("=" * 50)
    logger.info(f"Точность продукта: {avg_product_accuracy:.1f}/20 баллов")
    logger.info(f"Качество пуша: {avg_push_quality:.1f}/20 баллов")
    logger.info(f"Общий балл: {total_score:.1f}/40 баллов ({metrics['percentage']:.1f}%)")
    logger.info(f"Клиентов оценено: {metrics['clients_evaluated']}")
    
    # Детальная статистика по точности продуктов
    if product_accuracy_scores:
        perfect_matches = sum(1 for score in product_accuracy_scores if score == 20)
        good_matches = sum(1 for score in product_accuracy_scores if score >= 15)
        logger.info(f"Идеальных совпадений (20 баллов): {perfect_matches} ({perfect_matches/len(product_accuracy_scores)*100:.1f}%)")
        logger.info(f"Хороших совпадений (15+ баллов): {good_matches} ({good_matches/len(product_accuracy_scores)*100:.1f}%)")
    
    return metrics


def evaluate_personalization(push_text: str, client_code: int) -> float:
    """Оценивает персонализацию и уместность (5 баллов)."""
    score = 0
    
    # Проверяем наличие имени (2 балла)
    if re.search(r'[А-Я][а-я]+', push_text):
        score += 2
    
    # Проверяем наличие конкретных цифр/сумм (2 балла)
    if re.search(r'\d+\s*₸|%', push_text):
        score += 2
    
    # Проверяем контекстуальность (1 балл)
    context_words = ['в месяц', 'поездок', 'траты', 'статус', 'операций', 'категории']
    if any(word in push_text.lower() for word in context_words):
        score += 1
    
    return score


def evaluate_tov_compliance(push_text: str) -> float:
    """Оценивает соответствие TOV (5 баллов)."""
    score = 0
    
    # Проверяем "вы" с маленькой буквы (1 балл)
    if 'вы' in push_text.lower() and 'Вы' not in push_text:
        score += 1
    elif 'у вас' in push_text.lower():
        score += 1
    
    # Проверяем отсутствие КАПС (1 балл)
    caps_ratio = sum(1 for c in push_text if c.isupper()) / len(push_text) if push_text else 0
    if caps_ratio <= 0.1:  # Не более 10% заглавных
        score += 1
    
    # Проверяем доброжелательный тон (1 балл)
    positive_words = ['удобно', 'выгодно', 'привилегии', 'экономьте', 'принесёт']
    if any(word in push_text.lower() for word in positive_words):
        score += 1
    
    # Проверяем простоту языка (1 балл)
    if not re.search(r'(которые|таким образом|в связи с|вследствие)', push_text.lower()):
        score += 1
    
    # Проверяем отсутствие драматизации (1 балл)
    dramatic_words = ['срочно', 'немедленно', 'только сегодня', 'ограниченное время']
    if not any(word in push_text.lower() for word in dramatic_words):
        score += 1
    
    return score


def evaluate_clarity_and_brevity(push_text: str) -> float:
    """Оценивает ясность и краткость (5 баллов)."""
    score = 0
    
    # Проверяем длину (2 балла)
    length = len(push_text)
    if 180 <= length <= 220:
        score += 2
    elif 160 <= length <= 240:
        score += 1
    
    # Проверяем наличие четкого CTA (2 балла)
    cta_words = ['оформить', 'открыть', 'настроить', 'подключить', 'получить']
    if any(word in push_text.lower() for word in cta_words):
        score += 2
    
    # Проверяем одну основную мысль (1 балл)
    if push_text.count('.') <= 2:  # Не более 2 предложений
        score += 1
    
    return score


def evaluate_format_compliance(push_text: str) -> float:
    """Оценивает редполитику и формат (5 баллов)."""
    score = 0
    
    # Проверяем максимум 1 восклицательный знак (1 балл)
    if push_text.count('!') <= 1:
        score += 1
    
    # Проверяем правильное форматирование чисел (2 балла)
    currency_pattern = r'\d+\s*₸'
    if re.search(currency_pattern, push_text):
        score += 1
        # Проверяем разделители тысяч пробелами
        if re.search(r'\d+\s\d+\s*₸', push_text):
            score += 1
    
    # Проверяем правильное оформление процентов (1 балл)
    if re.search(r'\d+%', push_text):
        score += 1
    
    # Проверяем отсутствие проблемных символов (1 балл)
    if not re.search(r'[<>{}@#$]', push_text):
        score += 1
    
    return score


def analyze_product_distribution(output_file: str) -> Dict[str, any]:
    """Анализирует распределение рекомендованных продуктов."""
    try:
        df = pd.read_csv(output_file)
        distribution = df['product'].value_counts()
        
        analysis = {
            'total_clients': len(df),
            'unique_products': len(distribution),
            'distribution': distribution.to_dict(),
            'distribution_percentage': (distribution / len(df) * 100).to_dict(),
            'most_recommended': distribution.index[0] if len(distribution) > 0 else None,
            'diversity_score': len(distribution) / len(CONFIG.PRODUCTS)  # Разнообразие от 0 до 1
        }
        
        logger.info("Анализ распределения продуктов:")
        logger.info(f"Уникальных продуктов: {analysis['unique_products']}/10")
        logger.info(f"Коэффициент разнообразия: {analysis['diversity_score']:.2f}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Ошибка при анализе распределения: {e}")
        return {}


def generate_evaluation_report(output_file: str, clients_data_file: str) -> str:
    """Генерирует полный отчет об оценке системы."""
    report_lines = []
    
    # Основные метрики
    metrics = evaluate_system_performance(output_file, clients_data_file)
    
    # Анализ распределения
    distribution = analyze_product_distribution(output_file)
    
    # Формируем отчет
    report_lines.append("# ОТЧЕТ ОБ ОЦЕНКЕ СИСТЕМЫ ПЕРСОНАЛИЗАЦИИ")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    if metrics:
        report_lines.append("## Основные метрики (согласно ТЗ)")
        report_lines.append(f"- **Точность продукта**: {metrics['product_accuracy']:.1f}/20 баллов")
        report_lines.append(f"- **Качество пуша**: {metrics['push_quality']:.1f}/20 баллов")
        report_lines.append(f"- **Общий балл**: {metrics['total_score']:.1f}/40 баллов ({metrics['percentage']:.1f}%)")
        report_lines.append(f"- **Клиентов оценено**: {metrics['clients_evaluated']}")
        report_lines.append("")
    
    if distribution:
        report_lines.append("## Анализ рекомендаций")
        report_lines.append(f"- **Разнообразие продуктов**: {distribution['unique_products']}/10")
        report_lines.append(f"- **Коэффициент разнообразия**: {distribution['diversity_score']:.2f}")
        report_lines.append("")
        report_lines.append("### Распределение по продуктам:")
        for product, count in distribution['distribution'].items():
            percentage = distribution['distribution_percentage'][product]
            report_lines.append(f"- {product}: {count} ({percentage:.1f}%)")
    
    report_text = "\n".join(report_lines)
    
    # Сохраняем отчет
    report_file = output_file.replace('.csv', '_evaluation_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Отчет сохранен в {report_file}")
    return report_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка системы персонализации')
    parser.add_argument('--output', default='../data/output.csv',
                       help='Путь к файлу с результатами')
    parser.add_argument('--clients', default='../data/clients.csv',
                       help='Путь к файлу с данными клиентов')
    parser.add_argument('--report', action='store_true',
                       help='Создать полный отчет')
    
    args = parser.parse_args()
    
    if args.report:
        print(generate_evaluation_report(args.output, args.clients))
    else:
        evaluate_system_performance(args.output, args.clients)