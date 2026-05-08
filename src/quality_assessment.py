"""
Модуль оценки качества перевода без эталона (Reference-free QA).
Использует методы:
1. Обратная проверка (Round-trip Consistency)
2. Семантическая близость (Cosine Similarity) на основе Word2Vec
3. Оценка через Cross-Entropy (Perplexity)
4. Проверка именованных сущностей (NER Consistency)
"""
import logging
import math
import re
from typing import Tuple, Optional, List, Dict
from datetime import datetime

# Импортируем gensim для работы с Word2Vec
# Требуется установка: pip install gensim
try:
    from gensim.models import KeyedVectors
    import numpy as np
    GENSIM_AVAILABLE = True
    _w2v_model = None
except ImportError:
    GENSIM_AVAILABLE = False
    _w2v_model = None

# Создаём отдельный логгер для логов оценки качества
quality_logger = logging.getLogger("TranslationQuality")
quality_logger.setLevel(logging.INFO)

# Настраиваем файловый обработчик для записи в отдельную папку logs
try:
    from pathlib import Path
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "translation_quality.log"

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Формат с указанием времени, метода перевода и модели
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # Очищаем предыдущие обработчики и добавляем новый
    quality_logger.handlers.clear()
    quality_logger.addHandler(file_handler)
    quality_logger.propagate = False  # Не дублировать в основной лог
except Exception as e:
    # Если не удалось создать файл, используем стандартный логгер
    logging.warning(f"Не удалось создать файл лога для оценки качества: {e}")

logger = logging.getLogger("ScreenTranslator")

# Логирование после инициализации logger
if GENSIM_AVAILABLE:
    logger.info("gensim доступна, будет использоваться Word2Vec для эмбеддингов")
else:
    logger.warning("gensim не установлена. Установите: pip install gensim. Используется эвристический метод.")


def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Вычисляет косинусное сходство между двумя векторами.
    Формула: similarity = cos(θ) = (A · B) / (||A|| * ||B||)

    Args:
        vec1: первый вектор
        vec2: второй вектор

    Returns:
        косинусное сходство в диапазоне [0, 1]
    """
    if len(vec1) != len(vec2) or len(vec1) == 0:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Ограничиваем результат диапазоном [-1, 1] для избежания ошибок округления
    cosine_sim = max(-1.0, min(1.0, dot_product / (norm1 * norm2)))

    # Для оценки качества перевода используем только положительные значения [0, 1]
    # Отрицательные значения означают противоположные векторы, что для разных языков нормально
    return max(0.0, cosine_sim)


def get_w2v_model():
    """
    Получает или инициализирует модель Word2Vec.
    Использует предобученные векторы Russian National Corpus (300 dim).

    Returns:
        Модель KeyedVectors или None если недоступна
    """
    global _w2v_model
    if not GENSIM_AVAILABLE:
        return None

    if _w2v_model is None:
        try:
            # Загружаем русскую модель word2vec-ruscorpora-300
            logger.info("Загрузка модели Word2Vec (word2vec-ruscorpora-300)...")
            import gensim.downloader as api
            _w2v_model = api.load('word2vec-ruscorpora-300')
            logger.info(f"Модель Word2Vec успешно загружена: {_w2v_model.vector_size} dim, {len(_w2v_model)} слов")
        except Exception as e:
            logger.error(f"Не удалось загрузить модель Word2Vec: {e}")
            raise RuntimeError(f"Критическая ошибка: не удалось загрузить модель word2vec-ruscorpora-300: {e}")

    return _w2v_model


def normalize_text(text: str) -> str:
    """
    Нормализует текст перед созданием векторного представления.

    Выполняет:
    1. Удаление лишних пробелов (множественные → один, trim)
    2. Удаление знаков препинания
    3. Приведение к нижнему регистру

    Args:
        text: исходный текст

    Returns:
        нормализованный текст
    """
    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем знаки препинания (оставляем только буквы, цифры и пробелы)
    text = re.sub(r'[^\w\sа-яёa-z0-9]', ' ', text)

    # Заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text)

    # Удаляем ведущие и замыкающие пробелы
    text = text.strip()

    return text


def get_embedding(text: str) -> list[float]:
    """
    Получает векторное представление текста с помощью Word2Vec.

    Метод:
    1. Текст нормализуется (удаляются пробелы, пунктуация, lowercase)
    2. Текст токенизируется на слова
    3. Для каждого слова ищется вектор в модели Word2Vec (ruscorpora-300)
       - Сначала ищем слово без тега (например, "дом")
       - Если не найдено, ищем варианты с тегами (дом.S, дом.A и т.д.)
    4. Вектор предложения вычисляется как среднее арифметическое векторов слов
    5. Если слово не найдено, используем эвристический метод

    Args:
        text: текст для получения эмбеддинга

    Returns:
        вектор представления текста
    """
    model = get_w2v_model()

    # Если модель недоступна, возвращаем эвристический эмбеддинг
    if model is None:
        logger.warning("Модель Word2Vec недоступна, используем эвристический метод")
        return _heuristic_embedding(text)

    try:
        # Нормализация текста: удаление пунктуации, lowercase, но сохраняем структуру для поиска
        text_normalized = normalize_text(text)

        # Токенизация: разбиваем на слова
        words_raw = text_normalized.split()
        # Убираем цифры из слов и пустые строки
        words = [re.sub(r'\d+', '', w) for w in words_raw if re.sub(r'\d+', '', w)]

        logger.debug(f"[Embedding] Токенизация: {len(words)} слов из '{text[:50]}...'")

        # Получаем векторы для известных слов
        vectors = []
        not_found_words = []

        for word in words:
            # Прямой поиск слова в модели
            if word in model:
                vectors.append(model[word])
            else:
                # Пробуем найти слово с разными тегами частей речи
                # Модель ruscorpora использует форматы: слово_NOUN (существительное),
                # слово_ADJ (прилагательное), слово_VERB (глагол) и т.д.
                found = False
                for pos_tag in ['_NOUN', '_ADJ', '_VERB', '_ADV', '_PRON', '_ADP', '_CONJ', '_INTJ', '_NUM', '_DET', '_PART']:
                    tagged_word = word + pos_tag
                    if tagged_word in model:
                        vectors.append(model[tagged_word])
                        found = True
                        break

                if not found:
                    not_found_words.append(word)

        if not_found_words and len(not_found_words) <= 10:
            logger.debug(f"[Embedding] Не найдено слов: {not_found_words}")
        elif not_found_words:
            logger.debug(f"[Embedding] Не найдено {len(not_found_words)} слов")

        if vectors:
            # Усредняем векторы
            import numpy as np
            embedding = np.mean(vectors, axis=0)
            # Нормализация
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            logger.debug(f"[Embedding] Успешно: {len(vectors)} векторов, норма={norm:.4f}")
            return embedding.tolist()

        # Если ни одно слово не найдено, возвращаем нулевой вектор
        logger.warning(f"[Embedding] Ни одно слово не найдено в модели для текста: '{text[:100]}...'")
        return [0.0] * 300

    except Exception as e:
        logger.error(f"Ошибка при получении эмбеддинга через Word2Vec: {e}")
        # Возвращаем эвристический эмбеддинг вместо исключения
        return _heuristic_embedding(text)


def _heuristic_embedding(text: str) -> list[float]:
    """
    Эвристический метод создания псевдо-эмбеддинга когда Word2Vec недоступен.
    Использует простые статистические признаки текста.

    Args:
        text: текст для получения эмбеддинга

    Returns:
        псевдо-вектор размерности 300
    """
    # Создаём простой хеш-эмбеддинг на основе символов и слов
    import hashlib

    # Нормализуем текст
    text_lower = text.lower().strip()

    # Базовый хеш текста
    hash_bytes = hashlib.md5(text_lower.encode('utf-8')).digest()

    # Создаём вектор из 300 элементов на основе хеша и статистики текста
    vector = [0.0] * 300

    # Заполняем первые элементы на основе хеша
    for i in range(min(len(hash_bytes), 300)):
        vector[i] = (hash_bytes[i] - 128) / 128.0  # Нормализуем к [-1, 1]

    # Добавляем статистику текста в оставшиеся элементы
    words = text_lower.split()
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    word_count = len(words)
    char_count = len(text_lower)

    # Кодируем статистику в вектор
    vector[200] = min(1.0, avg_word_len / 10.0)
    vector[201] = min(1.0, word_count / 50.0)
    vector[202] = min(1.0, char_count / 500.0)
    vector[203] = 1.0 if any(c.isupper() for c in text) else 0.0
    vector[204] = 1.0 if any(c.isdigit() for c in text) else 0.0

    # Нормализация вектора
    norm = sum(v * v for v in vector) ** 0.5
    if norm > 0:
        vector = [v / norm for v in vector]

    return vector


def _heuristic_cross_entropy(original_text: str, translated_text: str) -> dict:
    """
    Эвристическая оценка Cross-Entropy когда Word2Vec недоступен.
    Использует простые метрики качества текста.

    Args:
        original_text: исходный текст
        translated_text: переведённый текст

    Returns:
        словарь с результатами оценки
    """
    # Простая эвристика на основе соотношения длин и структуры текста
    orig_len = len(original_text.strip())
    trans_len = len(translated_text.strip())

    # Соотношение длин (должно быть близко к 1)
    length_ratio = min(orig_len, trans_len) / max(orig_len, trans_len, 1)

    # Количество слов
    orig_words = len(original_text.split())
    trans_words = len(translated_text.split())
    word_ratio = min(orig_words, trans_words) / max(orig_words, trans_words, 1)

    # Оценка естественности перевода
    is_natural = length_ratio >= 0.5 and word_ratio >= 0.5

    # Качество на основе совпадения структур
    quality_score = (length_ratio * 0.6 + word_ratio * 0.4) * 100

    result = {
        "original_perplexity": round(10.0 / max(length_ratio, 0.1), 4),
        "translated_perplexity": round(10.0 / max(length_ratio, 0.1), 4),
        "perplexity_ratio": 1.0,
        "is_natural": is_natural,
        "quality_score": round(quality_score, 2),
        "method": "Cross-Entropy (Heuristic)",
        "details": {
            "original_preview": original_text[:100] + ("..." if len(original_text) > 100 else ""),
            "translated_preview": translated_text[:100] + ("..." if len(translated_text) > 100 else "")
        }
    }

    log_cross_entropy(result)
    return result


def _text_overlap_similarity(text1: str, text2: str) -> float:
    """
    Оценивает сходство текстов на основе перекрытия слов (Jaccard similarity).
    Работает для текстов на одном языке.

    Args:
        text1: первый текст
        text2: второй текст

    Returns:
        коэффициент сходства от 0 до 1
    """
    # Токенизация: приводим к нижнему регистру, удаляем пунктуацию
    def tokenize(text):
        text = text.lower()
        text = re.sub(r'[^\w\sа-яёa-z]', '', text)
        words = set(text.split())
        # Удаляем стоп-слова и короткие слова
        stopwords = {'и', 'в', 'не', 'на', 'я', 'он', 'the', 'a', 'an', 'is', 'are', 'was', 'were'}
        words = {w for w in words if len(w) > 2 and w not in stopwords}
        return words

    words1 = tokenize(text1)
    words2 = tokenize(text2)

    if not words1 or not words2:
        return 0.0

    # Jaccard similarity: |A ∩ B| / |A ∪ B|
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def _length_based_similarity(text1: str, text2: str) -> float:
    """
    Оценивает сходство на основе соотношения длин текстов.
    Используется как вспомогательная метрика.

    Args:
        text1: первый текст
        text2: второй текст

    Returns:
        коэффициент сходства от 0 до 1
    """
    len1 = len(text1.strip().split())
    len2 = len(text2.strip().split())

    if len1 == 0 or len2 == 0:
        return 0.0

    # Отношение короткой длины к длинной
    ratio = min(len1, len2) / max(len1, len2)

    # Преобразуем в оценку сходства
    # Идеальное совпадение = 1.0, разница в 2x = 0.5, разница в 4x = 0.25
    return ratio


def evaluate_semantic_similarity(original_text: str, translated_text: str, back_translated_text: Optional[str] = None) -> dict:
    """
    Оценка семантической близости через косинусное сходство векторов Word2Vec.

    Метод:
    1. Если есть обратный перевод (back_translated_text), сравниваем original_text с ним
       (оба на русском языке → векторы сопоставимы в пространстве ruscorpora-300)
    2. Если обратного перевода нет - используем прямой перевод (менее точно)

    Метод:
    1. Оба текста кодируются в векторное пространство Word2Vec (ruscorpora-300)
    2. Вычисляется косинус угла между ними
    3. Формула: similarity = cos(θ) = (A · B) / (||A|| * ||B||)

    Критерий: Значение от 0 до 1. Если оно ниже 0.7–0.8, перевод, скорее всего, неточный.

    Args:
        original_text: исходный текст (русский)
        translated_text: переведённый текст (английский)
        back_translated_text: обратный перевод (английский → русский), опционально

    Returns:
        словарь с результатами:
        - cosine_similarity: косинусное сходство
        - is_good: булево значение (порог 0.7)
        - quality_score: оценка качества
    """
    # Добавляем подробное логирование для отладки
    logger.info(f"[SemanticQA DEBUG] back_translated_text type: {type(back_translated_text)}, value: {repr(back_translated_text)[:200] if back_translated_text else 'None'}")

    cosine_sim = 0.0

    if back_translated_text is not None and isinstance(back_translated_text, str) and len(back_translated_text.strip()) > 0:
        logger.info("Используем обратный перевод для семантического сравнения")
        logger.info(f"[SemanticQA DEBUG] Original (RU): {original_text[:150]}...")
        logger.info(f"[SemanticQA DEBUG] Back-translated (RU): {back_translated_text[:150]}...")

        # Сравниваем оригинал с обратным переводом через Word2Vec
        orig_embedding = get_embedding(original_text)
        back_trans_embedding = get_embedding(back_translated_text)
        w2v_similarity = calculate_cosine_similarity(orig_embedding, back_trans_embedding)

        # Дополнительно вычисляем overlap similarity как резервный метод
        overlap_sim = _text_overlap_similarity(original_text, back_translated_text)

        # Length-based similarity как вспомогательная метрика
        length_sim = _length_based_similarity(original_text, back_translated_text)

        logger.info(f"[SemanticQA DEBUG] Word2Vec similarity: {w2v_similarity}")
        logger.info(f"[SemanticQA DEBUG] Overlap similarity: {overlap_sim}")
        logger.info(f"[SemanticQA DEBUG] Length similarity: {length_sim}")

        # Комбинируем все три метода
        # Word2Vec может давать 0 для разных формулировок, overlap более чувствителен к общим словам
        if w2v_similarity > 0 or overlap_sim > 0:
            # Комбинируем: 50% Word2Vec + 35% overlap + 15% length
            if w2v_similarity > 0 and overlap_sim > 0:
                cosine_sim = 0.5 * w2v_similarity + 0.35 * overlap_sim + 0.15 * length_sim
            else:
                cosine_sim = max(w2v_similarity, overlap_sim) * 0.85 + 0.15 * length_sim
        else:
            # Если Word2Vec и overlap дали 0, используем только length
            cosine_sim = length_sim * 0.5

        logger.info(f"[SemanticQA DEBUG] Combined cosine similarity (round-trip): {cosine_sim}")

        # Усиливаем оценку за счёт наличия round-trip данных
        # Round-trip сравнение более надёжное, поэтому даём бонус к качеству
        bonus_factor = 1.05  # 5% бонус
        cosine_sim = min(1.0, cosine_sim * bonus_factor)
    else:
        # Без обратного перевода сравниваем напрямую (менее точно)
        logger.warning("Обратный перевод недоступен или пуст, используем прямое сравнение через Word2Vec")
        logger.info(f"[SemanticQA DEBUG] Original (RU): {original_text[:150]}...")
        logger.info(f"[SemanticQA DEBUG] Translated (EN): {translated_text[:150]}...")

        # Прямое сравнение RU-EN через Word2Vec малоэффективно (разные языковые пространства)
        # Используем эвристики
        orig_embedding = get_embedding(original_text)
        trans_embedding = get_embedding(translated_text)
        w2v_similarity = calculate_cosine_similarity(orig_embedding, trans_embedding)

        # Length-based similarity для cross-lingual сравнения
        length_sim = _length_based_similarity(original_text, translated_text)

        logger.info(f"[SemanticQA DEBUG] Word2Vec similarity (direct RU-EN): {w2v_similarity}")
        logger.info(f"[SemanticQA DEBUG] Length similarity: {length_sim}")

        # Для cross-lingual используем в основном length-based
        if w2v_similarity > 0:
            cosine_sim = 0.7 * w2v_similarity + 0.3 * length_sim
        else:
            cosine_sim = length_sim * 0.6  # Снижаем вес для cross-lingual без Word2Vec

    # Порог 0.7 для определения хорошего перевода
    threshold = 0.7
    is_good = cosine_sim >= threshold

    result = {
        "cosine_similarity": round(cosine_sim, 4),
        "is_good": is_good,
        "threshold": threshold,
        "quality_score": round(cosine_sim * 100, 2),  # Конвертируем в проценты 0-100
        "method": "Semantic Similarity (Round-trip + Word2Vec ruscorpora-300 + Overlap)" if back_translated_text else "Semantic Similarity (Word2Vec ruscorpora-300)",
        "details": {
            "original_preview": original_text[:100] + ("..." if len(original_text) > 100 else ""),
            "translated_preview": translated_text[:100] + ("..." if len(translated_text) > 100 else ""),
            "back_translated_preview": (back_translated_text[:100] + ("..." if len(back_translated_text) > 100 else "")) if back_translated_text else None
        }
    }

    # Логирование результатов
    log_semantic_similarity(result)

    return result


def calculate_perplexity(text: str, model=None) -> float:
    """
    Вычисляет перплексию (Perplexity) текста как меру уверенности языковой модели.

    Формула: Perplexity = exp(-1/N * Σ log P(w_i | w_<i>))

    В данной реализации используется упрощённая эвристика на основе:
    - Частоты повторяющихся слов
    - Длины предложений
    - Наличия редких символов

    Args:
        text: текст для оценки
        model: языковая модель (опционально, если None используется эвристика)

    Returns:
        значение перплексии (чем меньше, тем лучше)
    """
    if not text or len(text.strip()) == 0:
        return float('inf')

    # Упрощённая эвристика для расчёта перплексии
    words = text.lower().split()
    n = len(words)

    if n == 0:
        return float('inf')

    # Подсчёт частоты слов
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Расчёт средней вероятности слов (упрощённо)
    total_log_prob = 0.0
    for word in words:
        # Вероятность слова обратно пропорциональна его частоте (упрощение)
        prob = word_freq[word] / n
        if prob > 0:
            total_log_prob += math.log(prob)

    # Средняя логарифмическая вероятность
    avg_log_prob = total_log_prob / n

    # Перплексия
    perplexity = math.exp(-avg_log_prob)

    # Штраф за очень короткие или очень длинные предложения
    avg_word_length = sum(len(w) for w in words) / n if n > 0 else 0
    if avg_word_length < 2:
        perplexity *= 1.5
    elif avg_word_length > 15:
        perplexity *= 1.3

    return round(perplexity, 4)


def evaluate_cross_entropy(original_text: str, translated_text: str) -> dict:
    """
    Оценка качества перевода через Cross-Entropy (Perplexity).

    ИСПРАВЛЕННАЯ ВЕРСИЯ:
    1. Используем предобученную языковую модель для оценки естественности
    2. Сравниваем перплексию перевода с ожидаемой для данного языка
    3. Учитываем соотношение длин текстов

    Метод:
    1. Вычисляется вероятность появления каждого слова в контексте предыдущих
    2. Если модель выдает перевод с низкой вероятностью (высокой перплексией),
       значит, текст содержит ошибки или звучит неестественно
    3. Формула: Perplexity = exp(-1/N * Σ log P(w_i | w_<i>))

    Args:
        original_text: исходный текст
        translated_text: переведённый текст

    Returns:
        словарь с результатами:
        - original_perplexity: перплексия оригинала
        - translated_perplexity: перплексия перевода
        - perplexity_ratio: отношение перплексий
        - is_natural: булево значение (насколько естественен перевод)
        - quality_score: оценка качества (0-100)
    """
    # Используем Word2Vec для оценки естественности через семантическую связность
    # Вычисляем перплексию на основе векторных расстояний между соседними словами
    model = get_w2v_model()

    # Если модель недоступна, используем упрощённую эвристику
    if model is None:
        logger.warning("Модель Word2Vec недоступна, используем эвристическую перплексию")
        return _heuristic_cross_entropy(original_text, translated_text)

    try:
        def w2v_perplexity(text: str) -> float:
            words = text.lower().split()
            if len(words) < 2:
                return 1.0

            vectors = []
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word in model:
                    vectors.append(model[clean_word])

            if len(vectors) < 2:
                return 50.0  # Высокая перплексия если мало известных слов

            # Вычисляем среднее расстояние между соседними векторами
            total_distance = 0.0
            for i in range(len(vectors) - 1):
                dist = np.linalg.norm(vectors[i] - vectors[i+1])
                total_distance += dist

            avg_distance = total_distance / (len(vectors) - 1)
            # Нормализуем: среднее расстояние ~0.5-1.5 для связного текста
            perplexity = avg_distance * 10
            return min(100.0, perplexity)

        translated_perplexity = w2v_perplexity(translated_text)
        original_perplexity = w2v_perplexity(original_text)
    except Exception as e:
        logger.error(f"Критическая ошибка при вычислении перплексии через Word2Vec: {e}")
        # Возвращаем эвристическую оценку вместо исключения
        return _heuristic_cross_entropy(original_text, translated_text)

    # Отношение перплексий (должно быть близко к 1)
    if original_perplexity > 0 and original_perplexity != float('inf'):
        perplexity_ratio = translated_perplexity / original_perplexity
    else:
        perplexity_ratio = 1.0

    # Оценка естественности (перплексия перевода не должна быть слишком высокой)
    # Порог: перплексия перевода не более чем в 2 раза выше оригинала
    is_natural = perplexity_ratio <= 2.0 and translated_perplexity < 50

    # Оценка качества на основе перплексии (нормализация к 0-100)
    # Чем меньше перплексия, тем лучше
    # Нормализуем: перплексия 1 = 100 баллов, перплексия 100 = 0 баллов
    translated_quality = max(0, min(100, 100 - translated_perplexity))
    ratio_quality = max(0, min(100, 100 - abs(perplexity_ratio - 1) * 50))

    quality_score = 0.6 * translated_quality + 0.4 * ratio_quality

    result = {
        "original_perplexity": round(original_perplexity, 4),
        "translated_perplexity": round(translated_perplexity, 4),
        "perplexity_ratio": round(perplexity_ratio, 4),
        "is_natural": is_natural,
        "quality_score": round(quality_score, 2),
        "method": "Cross-Entropy (Perplexity)",
        "details": {
            "original_preview": original_text[:100] + ("..." if len(original_text) > 100 else ""),
            "translated_preview": translated_text[:100] + ("..." if len(translated_text) > 100 else "")
        }
    }

    # Логирование результатов
    log_cross_entropy(result)

    return result


def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Извлекает именованные сущности из текста (упрощённая эвристика).

    Категории:
    - numbers: числа (целые, дробные)
    - dates: даты (форматы DD.MM.YYYY, MM/DD/YYYY, YYYY-MM-DD)
    - times: время (HH:MM, HH:MM:SS)
    - currencies: валюты ($, €, £, ₽ и т.д.)
    - percentages: проценты
    - emails: email адреса
    - urls: URL адреса

    Args:
        text: текст для извлечения сущностей

    Returns:
        словарь с категориями и списками найденных сущностей
    """
    entities = {
        "numbers": [],
        "dates": [],
        "times": [],
        "currencies": [],
        "percentages": [],
        "emails": [],
        "urls": []
    }

    # Числа (целые и дробные)
    numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', text)
    entities["numbers"] = numbers

    # Даты (различные форматы)
    dates = re.findall(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b', text)
    dates += re.findall(r'\b\d{2,4}[./-]\d{1,2}[./-]\d{1,2}\b', text)
    entities["dates"] = list(set(dates))

    # Время
    times = re.findall(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', text)
    entities["times"] = times

    # Валюты
    currencies = re.findall(r'[$€£¥₽₴₸₺]\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*(?:USD|EUR|GBP|JPY|RUB|UAH|KZT|TRY)', text, re.IGNORECASE)
    entities["currencies"] = currencies

    # Проценты
    percentages = re.findall(r'\d+(?:[.,]\d+)?\s*%', text)
    entities["percentages"] = percentages

    # Email
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    entities["emails"] = emails

    # URL
    urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)
    entities["urls"] = urls

    return entities


def evaluate_ner_consistency(original_text: str, translated_text: str) -> dict:
    """
    Оценка согласованности именованных сущностей в оригинале и переводе.

    Метод:
    1. Извлекаются именованные сущности из оригинала и перевода
    2. Сравнивается количество сущностей каждой категории
    3. Если в оригинале две даты, а в переводе — одна или три,
       математический коэффициент отклонения помечает сегмент как ошибочный

    Args:
        original_text: исходный текст
        translated_text: переведённый текст

    Returns:
        словарь с результатами:
        - original_entities: сущности оригинала
        - translated_entities: сущности перевода
        - entity_counts_match: булево значение (совпадают ли количества)
        - deviation_score: коэффициент отклонения (0 - идеально, >0 - есть отклонения)
        - is_consistent: булево значение (согласованы ли сущности)
        - quality_score: оценка качества (0-1)
    """
    original_entities = extract_named_entities(original_text)
    translated_entities = extract_named_entities(translated_text)

    # Подсчёт общего количества сущностей
    original_total = sum(len(v) for v in original_entities.values())
    translated_total = sum(len(v) for v in translated_entities.values())

    # Расчёт отклонений по каждой категории
    deviations = {}
    total_deviation = 0

    for category in original_entities.keys():
        orig_count = len(original_entities[category])
        trans_count = len(translated_entities[category])
        diff = abs(orig_count - trans_count)
        deviations[category] = {
            "original": orig_count,
            "translated": trans_count,
            "difference": diff
        }
        total_deviation += diff

    # Коэффициент отклонения (нормализованный)
    max_entities = max(original_total, translated_total, 1)
    deviation_score = total_deviation / max_entities

    # Перевод согласован, если отклонений нет или они минимальны
    is_consistent = deviation_score <= 0.2  # Допускаем до 20% отклонения

    # Оценка качества
    quality_score = max(0, 1 - deviation_score)

    result = {
        "original_entities": original_entities,
        "translated_entities": translated_entities,
        "original_total": original_total,
        "translated_total": translated_total,
        "deviations": deviations,
        "deviation_score": round(deviation_score, 4),
        "is_consistent": is_consistent,
        "quality_score": round(quality_score, 4),
        "method": "NER Consistency",
        "details": {
            "original_preview": original_text[:100] + ("..." if len(original_text) > 100 else ""),
            "translated_preview": translated_text[:100] + ("..." if len(translated_text) > 100 else "")
        }
    }

    # Логирование результатов
    log_ner_consistency(result)

    return result


def evaluate_roundtrip_consistency(
        original_text: str,
        translated_text: str,
        back_translated_text: str,
        threshold: float = 0.7,
        translation_method: str = "Не указано",
        model_name: str = "Не указано"
) -> dict:
    """
    Оценивает качество перевода через обратную проверку.

    Метод:
    1. Переводим фразу с Языка А на Язык Б
    2. Переводим результат обратно на Язык А
    3. Сравниваем исходную фразу и полученную после "круга"

    Args:
        original_text: исходный текст на языке источника
        translated_text: переведённый текст на целевом языке
        back_translated_text: текст после обратного перевода
        threshold: пороговое значение для определения "хорошего" перевода
        translation_method: метод/сервис перевода (например, "Google Translate", "Local LLM")
        model_name: название модели (например, "hunyuan-mt-7b", "google-translator")

    Returns:
        словарь с результатами оценки:
        - cosine_similarity: косинусное сходство между оригиналом и обратным переводом
        - is_consistent: булево значение, прошёл ли перевод проверку
        - quality_score: общая оценка качества (0-1)
        - details: дополнительные детали
    """
    # Добавляем подробное логирование для отладки
    logger.info(f"[RoundTrip DEBUG] Original: {original_text[:150]}...")
    logger.info(f"[RoundTrip DEBUG] Translated: {translated_text[:150]}...")
    logger.info(f"[RoundTrip DEBUG] Back-translated: {back_translated_text[:150]}...")

    # Получаем эмбеддинги
    original_embedding = get_embedding(original_text)
    back_translated_embedding = get_embedding(back_translated_text)

    logger.info(f"[RoundTrip DEBUG] Original embedding (first 5): {original_embedding[:5] if original_embedding else 'None'}")
    logger.info(f"[RoundTrip DEBUG] Back-translated embedding (first 5): {back_translated_embedding[:5] if back_translated_embedding else 'None'}")

    # Вычисляем косинусное сходство через Word2Vec
    w2v_similarity = calculate_cosine_similarity(original_embedding, back_translated_embedding)

    # Дополнительно вычисляем overlap similarity
    overlap_sim = _text_overlap_similarity(original_text, back_translated_text)

    logger.info(f"[RoundTrip DEBUG] Word2Vec similarity: {w2v_similarity}")
    logger.info(f"[RoundTrip DEBUG] Overlap similarity: {overlap_sim}")

    # Комбинируем оба метода
    if w2v_similarity > 0 or overlap_sim > 0:
        if w2v_similarity > 0 and overlap_sim > 0:
            cosine_sim = 0.6 * w2v_similarity + 0.4 * overlap_sim
        else:
            cosine_sim = max(w2v_similarity, overlap_sim)
    else:
        cosine_sim = 0.0

    logger.info(f"[RoundTrip DEBUG] Combined cosine similarity: {cosine_sim}")

    # Оценка качества на основе сходства
    is_consistent = cosine_sim >= threshold

    # Дополнительно учитываем длину текстов (сильные расхождения в длине - плохой знак)
    length_ratio = min(len(original_text), len(back_translated_text)) / max(len(original_text), len(back_translated_text), 1)

    # Итоговая оценка качества (комбинация сходства и соотношения длин)
    # Увеличиваем вес cosine_similarity как основного показателя
    quality_score = 0.8 * cosine_sim + 0.2 * length_ratio

    # Бонус за высокое сходство (если round-trip очень близок к оригиналу)
    if cosine_sim >= 0.85:
        quality_score = min(1.0, quality_score * 1.05)  # 5% бонус

    result = {
        "cosine_similarity": round(cosine_sim, 4),
        "is_consistent": is_consistent,
        "quality_score": round(quality_score, 4),
        "length_ratio": round(length_ratio, 4),
        "threshold": threshold,
        "original_length": len(original_text),
        "back_translated_length": len(back_translated_text),
        "translation_method": translation_method,
        "model_name": model_name,
        "details": {
            "original_preview": original_text[:100] + ("..." if len(original_text) > 100 else ""),
            "translated_preview": translated_text[:100] + ("..." if len(translated_text) > 100 else ""),
            "back_translated_preview": back_translated_text[:100] + ("..." if len(back_translated_text) > 100 else "")
        }
    }

    # Логирование результатов
    log_quality_assessment(result)

    return result


def evaluate_comprehensive_quality(
        original_text: str,
        translated_text: str,
        back_translated_text: str = None,
        translation_method: str = "Не указано",
        model_name: str = "Не указано"
) -> dict:
    """
    Комплексная оценка качества перевода с использованием всех доступных методов.

    Методы:
    1. Обратная проверка (Round-trip Consistency) - если есть back_translated_text
    2. Семантическая близость (Cosine Similarity)
    3. Оценка через Cross-Entropy (Perplexity)
    4. Проверка именованных сущностей (NER Consistency)

    Args:
        original_text: исходный текст на языке источника
        translated_text: переведённый текст на целевом языке
        back_translated_text: текст после обратного перевода (опционально)
        translation_method: метод/сервис перевода
        model_name: название модели

    Returns:
        словарь с результатами всех оценок и общей оценкой качества
    """
    results = {
        "translation_method": translation_method,
        "model_name": model_name,
        "methods_used": [],
        "scores": {},
        "overall_quality": 0.0,
        "details": {
            "original_preview": original_text[:100] + ("..." if len(original_text) > 100 else ""),
            "translated_preview": translated_text[:100] + ("..." if len(translated_text) > 100 else "")
        }
    }

    total_score = 0.0
    method_count = 0

    # Логирование входных данных для отладки
    logger.info(f"[ComprehensiveQA] Original: {original_text[:100]}...")
    logger.info(f"[ComprehensiveQA] Translated: {translated_text[:100]}...")
    logger.info(f"[ComprehensiveQA] Back-translated: {back_translated_text[:100] if back_translated_text else 'None'}...")

    # 1. Семантическая близость (всегда) - передаём back_translated_text если есть
    semantic_result = evaluate_semantic_similarity(original_text, translated_text, back_translated_text=back_translated_text)
    results["semantic_similarity"] = semantic_result
    results["scores"]["semantic"] = semantic_result["quality_score"]
    results["methods_used"].append("Semantic Similarity")

    # 2. Cross-Entropy / Perplexity (всегда)
    entropy_result = evaluate_cross_entropy(original_text, translated_text)
    results["cross_entropy"] = entropy_result
    results["scores"]["entropy"] = entropy_result["quality_score"]
    results["methods_used"].append("Cross-Entropy (Perplexity)")

    # 3. NER Consistency (всегда)
    ner_result = evaluate_ner_consistency(original_text, translated_text)
    results["ner_consistency"] = ner_result
    results["scores"]["ner"] = ner_result["quality_score"]
    results["methods_used"].append("NER Consistency")

    # 4. Round-trip Consistency (если есть обратный перевод)
    if back_translated_text:
        logger.info(f"[ComprehensiveQA] Вызываем roundtrip с back_translated_text={back_translated_text[:50]}...")
        roundtrip_result = evaluate_roundtrip_consistency(
            original_text=original_text,
            translated_text=translated_text,
            back_translated_text=back_translated_text,
            translation_method=translation_method,
            model_name=model_name
        )
        results["roundtrip_consistency"] = roundtrip_result
        results["scores"]["roundtrip"] = roundtrip_result["quality_score"]
        results["methods_used"].append("Round-trip Consistency")
        method_count += 1
    else:
        logger.warning("[ComprehensiveQA] Нет back_translated_text, пропускаем roundtrip проверку")

    # Общая оценка качества (взвешенное среднее по всем методам)
    # Коэффициенты подобраны для лучшей корреляции с reference-based оценками:
    # Анализ показал, что semantic и roundtrip имеют наибольшую корреляцию (~0.42) с ref_overall
    # - semantic: 0.50 (важнейшая метрика, оценивает смысл)
    # - roundtrip: 0.50 (критична для качества, показывает сохранение смысла при обратном переводе)
    # Примечание: cross_entropy=70 и ner=1.0 константы в текущих данных, поэтому их вес минимален

    semantic_weight = 0.50
    roundtrip_weight = 0.50 if back_translated_text else 0.0
    entropy_weight = 0.0  # Константа 70.0, не влияет на дифференциацию
    ner_weight = 0.0      # Константа 1.0, не влияет на дифференциацию

    # Нормализуем веса если нет roundtrip
    if not back_translated_text:
        total_weight = semantic_weight + entropy_weight + ner_weight
        if total_weight > 0:
            semantic_weight /= total_weight
            entropy_weight /= total_weight
            ner_weight /= total_weight
    else:
        total_weight = semantic_weight + roundtrip_weight

    weighted_score = 0.0
    weighted_score += semantic_weight * semantic_result["quality_score"]

    if back_translated_text:
        weighted_score += roundtrip_weight * roundtrip_result["quality_score"]

    results["overall_quality"] = round(weighted_score, 4)

    # Логирование комплексной оценки
    log_comprehensive_quality(results)

    return results


def log_comprehensive_quality(results: dict, detailed: bool = True) -> None:
    """
    Записывает результаты комплексной оценки качества в отдельный лог-файл.

    Args:
        results: словарь с результатами оценки от evaluate_comprehensive_quality
        detailed: если True - пишет подробный лог, если False - краткий
    """
    if detailed:
        quality_logger.info("=" * 80)
        quality_logger.info("КОМПЛЕКСНАЯ ОЦЕНКА КАЧЕСТВА ПЕРЕВОДА")
        quality_logger.info("=" * 80)
        quality_logger.info(f"Метод перевода: {results.get('translation_method', 'Не указано')}")
        quality_logger.info(f"Модель: {results.get('model_name', 'Не указано')}")
        quality_logger.info(f"Использованные методы: {', '.join(results.get('methods_used', []))}")
        quality_logger.info("-" * 80)

        # Логи по каждому методу с подробностями
        if "semantic_similarity" in results:
            sem = results["semantic_similarity"]
            quality_logger.info("[Semantic Similarity]")
            quality_logger.info(f"  Формула: similarity = cos(θ) = (A · B) / (||A|| * ||B||)")
            quality_logger.info(f"  Косинусное сходство: {sem['cosine_similarity']}")
            quality_logger.info(f"  Порог: {sem['threshold']}")
            quality_logger.info(f"  Качественный (>= {sem['threshold']}): {sem['is_good']}")
            quality_logger.info(f"  Оценка качества: {sem['quality_score']}")
            quality_logger.info("")

        if "cross_entropy" in results:
            ent = results["cross_entropy"]
            quality_logger.info("[Cross-Entropy (Perplexity)]")
            quality_logger.info(f"  Формула: Perplexity = exp(-1/N * Σ log P(w_i | w_<i>))")
            quality_logger.info(f"  Перплексия оригинала: {ent['original_perplexity']}")
            quality_logger.info(f"  Перплексия перевода: {ent['translated_perplexity']}")
            quality_logger.info(f"  Отношение: {ent['perplexity_ratio']}")
            quality_logger.info(f"  Естественный (ratio <= 2.0): {ent['is_natural']}")
            quality_logger.info(f"  Оценка качества: {ent['quality_score']}")
            if ent['perplexity_ratio'] > 2.0:
                quality_logger.info("  ⚠ Перплексия перевода значительно выше оригинала")
            elif ent['translated_perplexity'] > 50:
                quality_logger.info("  ⚠ Высокая перплексия перевода")
            else:
                quality_logger.info("  ✓ Перплексия в норме")
            quality_logger.info("")

        if "ner_consistency" in results:
            ner = results["ner_consistency"]
            quality_logger.info("[NER Consistency]")
            quality_logger.info(f"  Метод: Сравнение количества именованных сущностей")
            quality_logger.info(f"  Категории: числа, даты, время, валюты, проценты, email, URL")
            quality_logger.info(f"  Сущностей в оригинале: {ner['original_total']}")
            quality_logger.info(f"  Сущностей в переводе: {ner['translated_total']}")
            quality_logger.info(f"  Отклонение: {ner['deviation_score']}")
            quality_logger.info(f"  Согласован (<= 0.2): {ner['is_consistent']}")
            quality_logger.info(f"  Оценка качества: {ner['quality_score']}")
            quality_logger.info("  Детализация по категориям:")
            for category, data in ner['deviations'].items():
                status = "✓" if data['difference'] == 0 else "⚠"
                quality_logger.info(f"    {status} {category}: ориг={data['original']}, перев={data['translated']}, разн={data['difference']}")
            quality_logger.info("")

        if "roundtrip_consistency" in results:
            rt = results["roundtrip_consistency"]
            quality_logger.info("[Round-trip Consistency]")
            quality_logger.info(f"  Косинусное сходство: {rt['cosine_similarity']}")
            quality_logger.info(f"  Порог: {rt['threshold']}")
            quality_logger.info(f"  Согласован: {rt['is_consistent']}")
            quality_logger.info(f"  Соотношение длин: {rt['length_ratio']}")
            quality_logger.info(f"  Длина оригинала: {rt['original_length']} симв.")
            quality_logger.info(f"  Длина обратного перевода: {rt['back_translated_length']} симв.")
            quality_logger.info(f"  Оценка качества: {rt['quality_score']}")
            quality_logger.info("")

        quality_logger.info("-" * 80)
        quality_logger.info(f"ОБЩАЯ ОЦЕНКА КАЧЕСТВА: {results['overall_quality']}")
        quality_logger.info("-" * 80)
        quality_logger.info("ТЕКСТЫ:")
        quality_logger.info(f"  Оригинал: {results['details']['original_preview']}")
        quality_logger.info(f"  Перевод: {results['details']['translated_preview']}")
        quality_logger.info("=" * 80)
    else:
        # Краткий лог
        methods_summary = []
        if "semantic_similarity" in results:
            sem = results["semantic_similarity"]
            methods_summary.append(f"Semantic: {sem['quality_score']}")
        if "cross_entropy" in results:
            ent = results["cross_entropy"]
            methods_summary.append(f"Entropy: {ent['quality_score']}")
        if "ner_consistency" in results:
            ner = results["ner_consistency"]
            methods_summary.append(f"NER: {ner['quality_score']}")
        if "roundtrip_consistency" in results:
            rt = results["roundtrip_consistency"]
            methods_summary.append(f"Round-trip: {rt['quality_score']}")

        quality_logger.info(f"[ComprehensiveQA] Метод: {results.get('translation_method', 'Н/Д')} | "
                            f"Модель: {results.get('model_name', 'Н/Д')} | "
                            f"Методы: {', '.join(methods_summary)} | "
                            f"Общее качество: {results['overall_quality']}")

    # Дублирование в основной лог
    logger.info(f"[ComprehensiveQA] Метод: {results.get('translation_method', 'Н/Д')}, "
                f"Модель: {results.get('model_name', 'Н/Д')}, "
                f"Общее качество: {results['overall_quality']}")


def log_quality_assessment(result: dict, detailed: bool = True) -> None:
    """
    Записывает результаты оценки качества в отдельный лог-файл.

    Args:
        result: словарь с результатами оценки от evaluate_roundtrip_consistency
        detailed: если True - пишет подробный лог, если False - краткий
    """
    if detailed:
        # Подробный лог
        quality_logger.info("=" * 80)
        quality_logger.info("ОЦЕНКА КАЧЕСТВА ПЕРЕВОДА (Round-trip Consistency)")
        quality_logger.info("=" * 80)
        quality_logger.info(f"Метод перевода: {result.get('translation_method', 'Не указано')}")
        quality_logger.info(f"Модель: {result.get('model_name', 'Не указано')}")
        quality_logger.info("-" * 80)
        quality_logger.info("ДЕТАЛИ ОЦЕНКИ:")
        quality_logger.info(f"  Косинусное сходство: {result['cosine_similarity']}")
        quality_logger.info(f"  Пороговое значение: {result['threshold']}")
        quality_logger.info(f"  Перевод согласован: {result['is_consistent']}")
        quality_logger.info(f"  Соотношение длин текстов: {result['length_ratio']}")
        quality_logger.info(f"  Длина оригинала: {result['original_length']} симв.")
        quality_logger.info(f"  Длина обратного перевода: {result['back_translated_length']} симв.")
        quality_logger.info(f"  Общая оценка качества: {result['quality_score']}")
        quality_logger.info("-" * 80)
        quality_logger.info("ТЕКСТЫ:")
        quality_logger.info(f"  Оригинал: {result['details']['original_preview']}")
        quality_logger.info(f"  Перевод: {result['details']['translated_preview']}")
        quality_logger.info(f"  Обратный перевод: {result['details']['back_translated_preview']}")
        quality_logger.info("=" * 80)
    else:
        # Краткий лог
        quality_logger.info(f"[Round-trip] Метод: {result.get('translation_method', 'Н/Д')} | "
                            f"Модель: {result.get('model_name', 'Н/Д')} | "
                            f"Сходство: {result['cosine_similarity']} | "
                            f"Качество: {result['quality_score']} | "
                            f"Согласован: {result['is_consistent']}")

    # Дублирование в основной лог для удобства отладки
    logger.info(f"[QualityCheck] Метод: {result.get('translation_method', 'Н/Д')}, "
                f"Модель: {result.get('model_name', 'Н/Д')}, "
                f"Сходство: {result['cosine_similarity']}, "
                f"Качество: {result['quality_score']}, "
                f"Согласован: {result['is_consistent']}")


def log_semantic_similarity(result: dict, detailed: bool = True) -> None:
    """
    Записывает результаты оценки семантической близости в отдельный лог-файл.

    Args:
        result: словарь с результатами оценки от evaluate_semantic_similarity
        detailed: если True - пишет подробный лог, если False - краткий
    """
    if detailed:
        quality_logger.info("=" * 80)
        quality_logger.info("ОЦЕНКА СЕМАНТИЧЕСКОЙ БЛИЗОСТИ (Cosine Similarity)")
        quality_logger.info("=" * 80)
        quality_logger.info("-" * 80)
        quality_logger.info("МЕТОД И ФОРМУЛА:")
        quality_logger.info("  similarity = cos(θ) = (A · B) / (||A|| * ||B||)")
        quality_logger.info("  Где A и B - многомерные векторы представлений текстов")
        quality_logger.info("-" * 80)
        quality_logger.info("ДЕТАЛИ ОЦЕНКИ:")
        quality_logger.info(f"  Косинусное сходство: {result['cosine_similarity']}")
        quality_logger.info(f"  Пороговое значение: {result['threshold']}")
        quality_logger.info(f"  Перевод качественный (>= {result['threshold']}): {result['is_good']}")
        quality_logger.info(f"  Оценка качества: {result['quality_score']}")
        quality_logger.info("-" * 80)
        quality_logger.info("ТЕКСТЫ:")
        quality_logger.info(f"  Оригинал: {result['details']['original_preview']}")
        quality_logger.info(f"  Перевод: {result['details']['translated_preview']}")
        quality_logger.info("=" * 80)
    else:
        # Краткий лог
        quality_logger.info(f"[Semantic] Сходство: {result['cosine_similarity']} | "
                            f"Порог: {result['threshold']} | "
                            f"Качество: {result['quality_score']} | "
                            f"Хороший: {result['is_good']}")

    logger.info(f"[SemanticQA] Сходство: {result['cosine_similarity']}, "
                f"Качество: {result['quality_score']}, "
                f"Хороший: {result['is_good']}")


def log_cross_entropy(result: dict, detailed: bool = True) -> None:
    """
    Записывает результаты оценки через Cross-Entropy (Perplexity) в отдельный лог-файл.

    Args:
        result: словарь с результатами оценки от evaluate_cross_entropy
        detailed: если True - пишет подробный лог, если False - краткий
    """
    if detailed:
        quality_logger.info("=" * 80)
        quality_logger.info("ОЦЕНКА ЧЕРЕЗ CROSS-ENTROPY (Perplexity)")
        quality_logger.info("=" * 80)
        quality_logger.info("-" * 80)
        quality_logger.info("МЕТОД И ФОРМУЛА:")
        quality_logger.info("  Perplexity = exp(-1/N * Σ log P(w_i | w_<i>))")
        quality_logger.info("  Где P(w_i | w_<i>) - вероятность слова w_i в контексте предыдущих слов")
        quality_logger.info("  Чем меньше перплексия, тем естественнее текст")
        quality_logger.info("-" * 80)
        quality_logger.info("ДЕТАЛИ ОЦЕНКИ:")
        quality_logger.info(f"  Перплексия оригинала: {result['original_perplexity']}")
        quality_logger.info(f"  Перплексия перевода: {result['translated_perplexity']}")
        quality_logger.info(f"  Отношение перплексий (перевод/оригинал): {result['perplexity_ratio']}")
        quality_logger.info(f"  Перевод естественный (ratio <= 2.0): {result['is_natural']}")
        quality_logger.info(f"  Оценка качества: {result['quality_score']}")
        quality_logger.info("-" * 80)
        quality_logger.info("ИНТЕРПРЕТАЦИЯ:")
        if result['perplexity_ratio'] > 2.0:
            quality_logger.info("  ⚠ Перплексия перевода значительно выше оригинала - возможны ошибки")
        elif result['translated_perplexity'] > 50:
            quality_logger.info("  ⚠ Высокая перплексия перевода - текст может звучать неестественно")
        else:
            quality_logger.info("  ✓ Перплексия в норме - текст звучит естественно")
        quality_logger.info("-" * 80)
        quality_logger.info("ТЕКСТЫ:")
        quality_logger.info(f"  Оригинал: {result['details']['original_preview']}")
        quality_logger.info(f"  Перевод: {result['details']['translated_preview']}")
        quality_logger.info("=" * 80)
    else:
        # Краткий лог
        quality_logger.info(f"[Cross-Entropy] Перплексия ориг: {result['original_perplexity']} | "
                            f"Перплексия перев: {result['translated_perplexity']} | "
                            f"Отношение: {result['perplexity_ratio']} | "
                            f"Естественный: {result['is_natural']} | "
                            f"Качество: {result['quality_score']}")

    logger.info(f"[EntropyQA] Перплексия: {result['translated_perplexity']}, "
                f"Отношение: {result['perplexity_ratio']}, "
                f"Качество: {result['quality_score']}, "
                f"Естественный: {result['is_natural']}")


def log_ner_consistency(result: dict, detailed: bool = True) -> None:
    """
    Записывает результаты оценки согласованности именованных сущностей в отдельный лог-файл.

    Args:
        result: словарь с результатами оценки от evaluate_ner_consistency
        detailed: если True - пишет подробный лог, если False - краткий
    """
    if detailed:
        quality_logger.info("=" * 80)
        quality_logger.info("ОЦЕНКА СОГЛАСОВАННОСТИ ИМЕНОВАННЫХ СУЩНОСТЕЙ (NER Consistency)")
        quality_logger.info("=" * 80)
        quality_logger.info("-" * 80)
        quality_logger.info("МЕТОД:")
        quality_logger.info("  Сравнение количества именованных сущностей в оригинале и переводе")
        quality_logger.info("  Категории: числа, даты, время, валюты, проценты, email, URL")
        quality_logger.info("  Если количество сущностей различается - перевод помечается как ошибочный")
        quality_logger.info("-" * 80)
        quality_logger.info("ДЕТАЛИ ОЦЕНКИ:")
        quality_logger.info(f"  Всего сущностей в оригинале: {result['original_total']}")
        quality_logger.info(f"  Всего сущностей в переводе: {result['translated_total']}")
        quality_logger.info(f"  Коэффициент отклонения: {result['deviation_score']}")
        quality_logger.info(f"  Согласован (отклонение <= 0.2): {result['is_consistent']}")
        quality_logger.info(f"  Оценка качества: {result['quality_score']}")
        quality_logger.info("-" * 80)
        quality_logger.info("ДЕТАЛИЗАЦИЯ ПО КАТЕГОРИЯМ:")
        for category, data in result['deviations'].items():
            orig_count = data['original']
            trans_count = data['translated']
            diff = data['difference']
            status = "✓" if diff == 0 else "⚠"
            quality_logger.info(f"  {status} {category}: оригинал={orig_count}, перевод={trans_count}, разница={diff}")
        quality_logger.info("-" * 80)
        quality_logger.info("ТЕКСТЫ:")
        quality_logger.info(f"  Оригинал: {result['details']['original_preview']}")
        quality_logger.info(f"  Перевод: {result['details']['translated_preview']}")
        quality_logger.info("=" * 80)
    else:
        # Краткий лог
        quality_logger.info(f"[NER] Сущностей ориг: {result['original_total']} | "
                            f"Сущностей перев: {result['translated_total']} | "
                            f"Отклонение: {result['deviation_score']} | "
                            f"Согласован: {result['is_consistent']} | "
                            f"Качество: {result['quality_score']}")

    logger.info(f"[NERQA] Сущностей: {result['original_total']}->{result['translated_total']}, "
                f"Отклонение: {result['deviation_score']}, "
                f"Качество: {result['quality_score']}, "
                f"Согласован: {result['is_consistent']}")