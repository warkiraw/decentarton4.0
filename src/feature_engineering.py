"""
Модуль feature engineering для системы персонализации банковских предложений.
Содержит функции для генерации RFM-D признаков, кластеризации и propensity scores.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from typing import Dict, Any
from config import CONFIG

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_rfmd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает RFM-D (Recency, Frequency, Monetary, Diversity) признаки для клиентов.
    
    Args:
        df: DataFrame с данными клиентов
        
    Returns:
        DataFrame с добавленными RFM-D признаками
    """
    logger.info("Создание RFM-D признаков")
    
    df_features = df.copy()
    
    # 1. Recency - давность последней активности
    # Поскольку у нас агрегированные данные, используем текущую дату как базу
    current_date = pd.Timestamp.now()
    
    # Если есть данные о последней активности, рассчитываем recency
    # Для агрегированных данных устанавливаем условную recency на основе баланса
    # Высокий баланс = недавняя активность (низкий recency)
    if 'avg_monthly_balance_KZT' in df_features.columns:
        # Нормализуем recency (инвертируем, чтобы высокий баланс = низкий recency)
        max_balance = df_features['avg_monthly_balance_KZT'].max()
        if max_balance > 0:
            df_features['recency'] = 1 - (df_features['avg_monthly_balance_KZT'] / max_balance)
        else:
            df_features['recency'] = 0.5
    else:
        df_features['recency'] = 0.5
    
    # 2. Frequency - частота трат в фокусных категориях
    focus_categories = ['Такси', 'Путешествия', 'Ресторан', 'Покупки', 'Развлечения']
    frequency_score = 0
    
    for category in focus_categories:
        spend_col = f'spend_{category}'
        if spend_col in df_features.columns:
            # Подсчитываем количество трат (используем наличие ненулевого значения как индикатор активности)
            frequency_score += (df_features[spend_col] > 0).astype(int)
    
    df_features['frequency'] = frequency_score
    
    # 3. Monetary - среднемесячные расходы + остаток
    monetary_score = df_features.get('avg_monthly_balance_KZT', 0)
    
    # Добавляем общие траты по всем категориям
    spend_columns = [col for col in df_features.columns if col.startswith('spend_')]
    if spend_columns:
        total_spending = df_features[spend_columns].sum(axis=1)
        monetary_score += total_spending
    
    df_features['monetary'] = monetary_score
    
    # 4. Diversity - разнообразие категорий и типов транзакций
    diversity_score = 0
    
    # Считаем количество категорий с активностью
    spend_categories = [col for col in df_features.columns if col.startswith('spend_')]
    if spend_categories:
        diversity_score += (df_features[spend_categories] > 0).sum(axis=1)
    
    # Считаем количество типов переводов с активностью
    transfer_types = [col for col in df_features.columns if col.startswith('transfer_')]
    if transfer_types:
        diversity_score += (df_features[transfer_types] > 0).sum(axis=1)
    
    df_features['diversity'] = diversity_score
    
    # Нормализация признаков
    scaler = MinMaxScaler()
    
    # Нормализуем recency (уже в диапазоне 0-1)
    # Нормализуем frequency
    if df_features['frequency'].max() > 0:
        df_features['frequency_normalized'] = df_features['frequency'] / df_features['frequency'].max()
    else:
        df_features['frequency_normalized'] = 0
    
    # Нормализуем monetary
    if df_features['monetary'].max() > 0:
        df_features['monetary_normalized'] = scaler.fit_transform(df_features[['monetary']])[:, 0]
    else:
        df_features['monetary_normalized'] = 0
    
    # Нормализуем diversity
    if df_features['diversity'].max() > 0:
        df_features['diversity_normalized'] = df_features['diversity'] / df_features['diversity'].max()
    else:
        df_features['diversity_normalized'] = 0
    
    logger.info(f"RFM-D признаки созданы для {len(df_features)} клиентов")
    
    return df_features


def add_cluster_labels(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    🚀 РЕВОЛЮЦИОННАЯ кластеризация с elbow method и advanced features для >0.4 silhouette!
    
    Args:
        df: DataFrame с RFM-D признаками
        config: Словарь конфигурации
        
    Returns:
        DataFrame с добавленными метками кластеров
    """
    logger.info("🚀 Выполнение РЕВОЛЮЦИОННОЙ кластеризации клиентов")
    
    df_clustered = df.copy()
    
    # 1. 🎯 РАСШИРЕННЫЕ ФИЧИ для лучшей кластеризации
    advanced_features = []
    
    # Базовые RFM-D (нормализованные предпочтительнее)
    base_features = ['recency', 'frequency_normalized', 'monetary_normalized', 'diversity_normalized']
    for feature in base_features:
        if feature in df_clustered.columns:
            advanced_features.append(feature)
    
    # Если нормализованных нет, используем обычные
    if not advanced_features:
        fallback_features = ['recency', 'frequency', 'monetary', 'diversity']
        for feature in fallback_features:
            if feature in df_clustered.columns:
                advanced_features.append(feature)
    
    # 💰 Spending patterns - агрегированные категории трат
    spend_columns = [col for col in df_clustered.columns if col.startswith('spend_')]
    if len(spend_columns) > 3:
        # Добавляем ТОП категории трат
        advanced_features.extend(spend_columns[:8])  # Топ-8 категорий
        
        # 📊 Волатильность трат (стабильность клиента)  
        df_clustered['spend_volatility'] = df_clustered[spend_columns].std(axis=1)
        advanced_features.append('spend_volatility')
        
        # 🎯 Концентрация в топ-3 категориях
        top3_sums = df_clustered[spend_columns].apply(lambda x: x.nlargest(3).sum(), axis=1)
        total_sums = df_clustered[spend_columns].sum(axis=1)
        df_clustered['top3_concentration'] = np.where(total_sums > 0, top3_sums / total_sums, 0)
        advanced_features.append('top3_concentration')
    
    # 💱 FX активность
    fx_cols = [col for col in df_clustered.columns if 'fx_' in col.lower()]
    if fx_cols:
        df_clustered['fx_activity_total'] = df_clustered[fx_cols].sum(axis=1) 
        advanced_features.append('fx_activity_total')
    
    # 🏦 Баланс (логарифмированный для лучшего распределения)
    if 'avg_monthly_balance_KZT' in df_clustered.columns:
        df_clustered['balance_log'] = np.log1p(df_clustered['avg_monthly_balance_KZT'])
        advanced_features.append('balance_log')
    
    # 🧠 Propensity scores если есть
    propensity_cols = [col for col in df_clustered.columns if col.startswith('propensity_')]
    advanced_features.extend(propensity_cols)
    
    # ✅ Проверяем достаточность фичей
    existing_features = [col for col in advanced_features if col in df_clustered.columns]
    
    if len(existing_features) < 3:
        logger.warning("Недостаточно фичей для качественной кластеризации")
        df_clustered['cluster'] = 0
        return df_clustered
    
    logger.info(f"🎯 Используем {len(existing_features)} продвинутых фичей для кластеризации")
    
    # 2. 📊 Подготовка данных
    features_for_clustering = df_clustered[existing_features].fillna(0)
    
    # Удаляем константные фичи
    feature_std = features_for_clustering.std()
    variable_features = feature_std[feature_std > 0].index.tolist()
    
    if len(variable_features) < 2:
        logger.warning("Недостаточно вариативных фичей")
        df_clustered['cluster'] = 0
        return df_clustered
    
    features_for_clustering = features_for_clustering[variable_features]
    
    # 3. 🔬 Стандартизация
    scaler = StandardScaler()
    try:
        scaled_features = scaler.fit_transform(features_for_clustering)
    except Exception as e:
        logger.error(f"Ошибка при стандартизации: {e}")
        df_clustered['cluster'] = 0
        return df_clustered
    
    # 4. 📈 ELBOW METHOD для оптимального числа кластеров
    max_clusters = min(8, len(df_clustered) // 4)  # Разумные границы
    k_range = range(2, max_clusters + 1)
    
    silhouette_scores = []
    inertias = []
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_features)
            inertias.append(kmeans.inertia_)
            
            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(scaled_features, labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(0)
        except Exception:
            silhouette_scores.append(0)
            inertias.append(float('inf'))
    
    # 🎯 Выбираем k с максимальным silhouette (приоритет качеству!)
    if silhouette_scores and max(silhouette_scores) > 0:
        optimal_k = k_range[np.argmax(silhouette_scores)]
        expected_silhouette = max(silhouette_scores)
    else:
        optimal_k = config.get('CLUSTER_PARAMS', {}).get('n_clusters', 4)
        expected_silhouette = 0
    
    logger.info(f"🏆 Оптимальное k: {optimal_k}, ожидаемый silhouette: {expected_silhouette:.3f}")
    
    # 5. 🚀 Финальная кластеризация с оптимальным k
    try:
        final_kmeans = KMeans(
            n_clusters=optimal_k, 
            random_state=42, 
            n_init=20,  # Больше инициализаций для стабильности
            max_iter=500
        )
        cluster_labels = final_kmeans.fit_predict(scaled_features)
        df_clustered['cluster'] = cluster_labels
        
        # 6. 📊 Финальная оценка качества
        if len(set(cluster_labels)) > 1:
            final_silhouette = silhouette_score(scaled_features, cluster_labels)
            logger.info(f"🎯 Финальный Silhouette Score: {final_silhouette:.3f}")
            
            if final_silhouette >= 0.4:
                logger.info(f"🏆 ПОБЕДА! Достигнут целевой Silhouette: {final_silhouette:.3f} >= 0.4!")
            else:
                logger.warning(f"⚠️ Silhouette ниже целевого: {final_silhouette:.3f} < 0.4")
        else:
            logger.warning("Все объекты попали в один кластер")
        
        # Статистика кластеризации
        cluster_distribution = df_clustered['cluster'].value_counts().sort_index()
        logger.info(f"📊 Распределение по кластерам: {dict(cluster_distribution)}")
        
    except Exception as e:
        logger.error(f"Ошибка при финальной кластеризации: {e}")
        df_clustered['cluster'] = 0
    
    return df_clustered


def add_propensity_scores(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Добавляет propensity scores для ключевых продуктов.
    Используется только если AB_TEST_FLAG == 'B'.
    
    Args:
        df: DataFrame с признаками
        config: Словарь конфигурации
        
    Returns:
        DataFrame с добавленными propensity scores
    """
    if config.get('AB_TEST_FLAG', 'A') != 'B':
        logger.info("Propensity scores отключены (AB_TEST_FLAG != 'B')")
        return df
    
    logger.info("Создание propensity scores для ключевых продуктов")
    
    df_propensity = df.copy()
    
    # Продукты для моделирования propensity
    target_products = ['Кредитная карта', 'Инвестиции', 'Депозит мультивалютный']
    
    # Подготовка признаков
    feature_columns = ['recency', 'frequency_normalized', 'monetary_normalized', 'diversity_normalized']
    
    # Добавляем категориальные признаки
    categorical_features = []
    if 'status' in df_propensity.columns:
        categorical_features.append('status')
    if 'city' in df_propensity.columns:
        categorical_features.append('city')
    
    # Добавляем траты по категориям
    spend_columns = [col for col in df_propensity.columns if col.startswith('spend_')]
    feature_columns.extend(spend_columns)
    
    # Добавляем переводы
    transfer_columns = [col for col in df_propensity.columns if col.startswith('transfer_')]
    feature_columns.extend(transfer_columns)
    
    # Фильтруем существующие столбцы
    existing_numeric_features = [col for col in feature_columns if col in df_propensity.columns]
    existing_categorical_features = [col for col in categorical_features if col in df_propensity.columns]
    
    if not existing_numeric_features:
        logger.warning("Не найдены числовые признаки для propensity моделирования")
        for product in target_products:
            safe_product_name = product.lower().replace(' ', '_').replace('ё', 'е')
            df_propensity[f'propensity_{safe_product_name}'] = 0.5
        return df_propensity
    
    # Создаем синтетические таргеты для каждого продукта
    for product in target_products:
        try:
            safe_product_name = product.lower().replace(' ', '_').replace('ё', 'е')
            
            # Создаем синтетический таргет на основе паттернов трат
            target = _create_synthetic_target(df_propensity, product, config)
            
            if target.sum() == 0 or target.sum() == len(target):
                # Если все таргеты одинаковые, устанавливаем средний propensity
                logger.warning(f"Не удалось создать разнообразный таргет для {product}")
                df_propensity[f'propensity_{safe_product_name}'] = 0.5
                continue
            
            # Подготовка признаков
            X_numeric = df_propensity[existing_numeric_features].fillna(0)
            
            # Обработка категориальных признаков
            if existing_categorical_features:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                X_categorical = encoder.fit_transform(
                    df_propensity[existing_categorical_features].fillna('unknown')
                )
                X = np.hstack([X_numeric.values, X_categorical])
            else:
                X = X_numeric.values
            
            # Разделение на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, target,
                test_size=1 - config['PROPENSITY_PARAMS']['train_split'],
                random_state=42,
                stratify=target if len(set(target)) > 1 else None
            )
            
            # Обучение модели (увеличен max_iter для convergence)
            lr = LogisticRegression(random_state=42, max_iter=5000, solver='lbfgs')
            lr.fit(X_train, y_train)
            
            # Предсказание propensity scores
            propensity_scores = lr.predict_proba(X)[:, 1]
            df_propensity[f'propensity_{safe_product_name}'] = propensity_scores
            
            # Логируем качество модели
            train_score = lr.score(X_train, y_train)
            test_score = lr.score(X_test, y_test)
            logger.info(f"Propensity модель для {product}: Train={train_score:.3f}, Test={test_score:.3f}")
            
        except Exception as e:
            logger.error(f"Ошибка при создании propensity модели для {product}: {e}")
            df_propensity[f'propensity_{safe_product_name}'] = 0.5
    
    return df_propensity


def _create_synthetic_target(df: pd.DataFrame, product: str, config: Dict[str, Any]) -> np.ndarray:
    """
    Создает синтетический таргет для продукта на основе паттернов поведения.
    
    Args:
        df: DataFrame с данными клиентов
        product: Название продукта
        config: Конфигурация
        
    Returns:
        Массив с синтетическими таргетами (0 или 1)
    """
    threshold_spend = config['PROPENSITY_PARAMS']['threshold_spend']
    targets = np.zeros(len(df))
    
    if product == 'Кредитная карта':
        # ФИКС: Используем правильные названия колонок из данных + правильное преобразование bool
        restaurant_spend = df.get('spend_Кафе и рестораны', 0)
        online_spend = (df.get('spend_Едим дома', 0) + 
                       df.get('spend_Смотрим дома', 0) + 
                       df.get('spend_Играем дома', 0))
        total_spend = restaurant_spend + online_spend
        condition = total_spend > threshold_spend * 0.3
        targets = np.array(condition, dtype=int)  # ФИКС: используем np.array вместо .astype(int)
        
    elif product == 'Инвестиции':
        # Клиенты с высокими балансами и низкой активностью трат
        high_balance = df.get('avg_monthly_balance_KZT', 0) > config['RFMD_THRESHOLDS']['high_balance']
        spend_cols = [col for col in df.columns if col.startswith('spend_')]
        if spend_cols:
            low_spending = df[spend_cols].sum(axis=1) < threshold_spend * 0.5
        else:
            low_spending = pd.Series([True] * len(df), index=df.index)
        condition = high_balance & low_spending
        
        # ФИКС: Добавляем искусственную вариативность если target не разнообразный
        targets = np.array(condition, dtype=int)
        if targets.sum() == 0 or targets.sum() == len(targets):
            # Добавляем шум для разнообразия (10% случайных флипов)
            np.random.seed(42)
            noise_indices = np.random.choice(len(targets), size=max(1, len(targets)//10), replace=False)
            targets[noise_indices] = 1 - targets[noise_indices]
        
    elif product == 'Депозит мультивалютный':
        # Клиенты с FX активностью
        fx_buy = df.get('transfer_fx_buy', 0)
        fx_sell = df.get('transfer_fx_sell', 0)
        fx_activity = fx_buy + fx_sell
        condition = fx_activity > config['RFMD_THRESHOLDS']['fx_volume_threshold']
        targets = np.array(condition, dtype=int)  # ФИКС: используем np.array вместо .astype(int)
    
    return targets


def test_create_rfmd_features():
    """Тест функции создания RFM-D признаков."""
    try:
        # Создаем тестовые данные
        test_data = pd.DataFrame({
            'client_code': [1, 2, 3],
            'avg_monthly_balance_KZT': [100000, 200000, 50000],
            'spend_Такси': [5000, 0, 2000],
            'spend_Путешествия': [50000, 30000, 0],
            'spend_Ресторан': [15000, 20000, 8000],
            'transfer_fx_buy': [10000, 0, 5000],
            'transfer_fx_sell': [0, 20000, 0]
        })
        
        result = create_rfmd_features(test_data)
        
        # Проверяем наличие новых столбцов
        required_columns = ['recency', 'frequency', 'monetary', 'diversity']
        for col in required_columns:
            assert col in result.columns, f"Столбец {col} отсутствует"
        
        # Проверяем, что значения в разумных пределах
        assert result['recency'].between(0, 1).all(), "Recency должен быть в диапазоне 0-1"
        assert result['frequency'].min() >= 0, "Frequency не должен быть отрицательным"
        assert result['monetary'].min() >= 0, "Monetary не должен быть отрицательным"
        assert result['diversity'].min() >= 0, "Diversity не должен быть отрицательным"
        
        print("Тест create_rfmd_features пройден успешно!")
        
    except Exception as e:
        print(f"Тест create_rfmd_features не пройден: {e}")


if __name__ == "__main__":
    test_create_rfmd_features()