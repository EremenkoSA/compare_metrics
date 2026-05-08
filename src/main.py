#!/usr/bin/env python3
"""
Основной скрипт для сравнения оценок качества перевода:
1. Оценка по эталону (reference-based) с использованием стандартных метрик (BLEU, METEOR, chrF, etc.)
2. Оценка без эталона (reference-free) через quality_assessment модуль
3. Сравнение результатов двух подходов

Использует локальную модель через LM Studio API (http://localhost:1234)
"""

import json
import logging
import requests
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from quality_assessment import (
    evaluate_semantic_similarity,
    evaluate_cross_entropy,
    evaluate_ner_consistency,
    evaluate_roundtrip_consistency,
    quality_logger
)

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TranslationPair:
    """Пара исходный текст - перевод"""
    source: str  # Русский текст
    reference: str  # Эталонный английский перевод
    candidate: str  # Кандидат на перевод (от нашей модели)

    def __post_init__(self):
        self.source = self.source.strip()
        self.reference = self.reference.strip()
        self.candidate = self.candidate.strip()


@dataclass
class ReferenceBasedMetrics:
    """Метрики оценки по эталону"""
    bleu: float = 0.0
    meteor: float = 0.0
    chrf: float = 0.0
    ter: float = 0.0
    comet: float = 0.0
    overall_score: float = 0.0


@dataclass
class ReferenceFreeMetrics:
    """Метрики оценки без эталона"""
    semantic_similarity: float = 0.0
    cross_entropy_quality: float = 0.0
    ner_consistency: float = 0.0
    roundtrip_consistency: float = 0.0
    overall_score: float = 0.0


@dataclass
class ComparisonResult:
    """Результат сравнения оценок"""
    pair_id: int
    source_preview: str
    reference_preview: str
    candidate_preview: str

    # Полные тексты для анализа
    source_full: str = ""
    reference_full: str = ""
    candidate_full: str = ""
    back_translated_full: str = ""

    # Метрики по эталону
    ref_bleu: float = 0.0
    ref_meteor: float = 0.0
    ref_chrf: float = 0.0
    ref_overall: float = 0.0

    # Мои метрики (без эталона)
    my_semantic: float = 0.0
    my_cross_entropy: float = 0.0
    my_ner: float = 0.0
    my_roundtrip: float = 0.0
    my_overall: float = 0.0

    # Разница
    difference: float = 0.0
    correlation_marker: str = ""  # "match", "diverge", "partial"


class LMStudioClient:
    """Клиент для работы с LM Studio API"""

    def __init__(self, base_url: str = "http://localhost:1234", model: str = "gigachat3.1-10b-a1.8b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def is_available(self) -> bool:
        """Проверяет доступность LM Studio сервера"""
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def translate(self, text: str, source_lang: str = "ru", target_lang: str = "en") -> Optional[str]:
        """
        Переводит текст с помощью локальной модели

        Args:
            text: текст для перевода
            source_lang: язык источника (ru/en)
            target_lang: целевой язык (ru/en)

        Returns:
            переведённый текст или None при ошибке
        """
        if source_lang == "ru" and target_lang == "en":
            prompt = f"Translate the following Russian text to English. Provide only the translation, no explanations:\n\n{text}"
        elif source_lang == "en" and target_lang == "ru":
            prompt = f"Translate the following English text to Russian. Provide only the translation, no explanations:\n\n{text}"
        else:
            logger.error(f"Неподдерживаемая пара языков: {source_lang} -> {target_lang}")
            return None

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional translator. Translate accurately while preserving meaning and style."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
            "stream": False
        }

        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.warning(f"Пустой ответ от API: {result}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса к LM Studio: {e}")
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Ошибка парсинга ответа: {e}")
            return None

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Получает эмбеддинг текста через LM Studio API

        Args:
            text: текст для получения эмбеддинга

        Returns:
            вектор эмбеддинга или None при ошибке
        """
        payload = {
            "model": self.model,
            "input": text
        }

        try:
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            if "data" in result and len(result["data"]) > 0:
                return result["data"][0].get("embedding", [])
            else:
                logger.warning(f"Пустой ответ embeddings API: {result}")
                return None

        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга: {e}")
            return None


def parse_dataset(file_path: str, max_lines: Optional[int] = None) -> List[TranslationPair]:
    """
    Парсит датасет в формате fast_align (текст1 ||| текст2)

    Args:
        file_path: путь к файлу датасета
        max_lines: максимальное количество строк для обработки (None = все)

    Returns:
        список пар TranslationPair
    """
    pairs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_lines and line_num > max_lines:
                break

            line = line.strip()
            if not line or '|||' not in line:
                continue

            parts = line.split('|||')
            if len(parts) >= 2:
                source = parts[0].strip()
                reference = parts[1].strip()

                # Пропускаем пустые строки
                if source and reference:
                    pairs.append(TranslationPair(
                        source=source,
                        reference=reference,
                        candidate=""  # Будет заполнено после перевода
                    ))

    logger.info(f"Загружено {len(pairs)} пар из {file_path}")
    return pairs


def calculate_bleu(reference: str, candidate: str) -> float:
    """
    Вычисляет BLEU score между эталоном и кандидатом

    Использует упрощённую реализацию без внешних зависимостей
    """
    from collections import Counter

    def tokenize(text: str) -> List[str]:
        return text.lower().split()

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not cand_tokens:
        return 0.0

    # Подсчёт n-gram (для n=1,2,3,4)
    max_n = min(4, len(cand_tokens), len(ref_tokens))
    if max_n == 0:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)])
        cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)])

        # Clipped count
        clipped_count = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total_count = sum(cand_ngrams.values())

        if total_count == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_count / total_count)

    if not precisions:
        return 0.0

    # Geometric mean of precisions
    from math import exp, log
    geo_mean = exp(sum(log(p) if p > 0 else -10 for p in precisions) / len(precisions))

    # Brevity penalty
    bp = 1.0 if len(cand_tokens) >= len(ref_tokens) else exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))

    return round(bp * geo_mean * 100, 2)  # Возвращаем в процентах


def calculate_chrf(reference: str, candidate: str, beta: float = 2.0) -> float:
    """
    Вычисляет chrF (character n-gram F-score)
    """
    from collections import Counter

    def get_char_ngrams(text: str, n: int) -> Counter:
        return Counter([text[i:i+n] for i in range(len(text) - n + 1)])

    if not candidate:
        return 0.0

    # Используем n-gram от 1 до 6 символов
    max_n = 6

    precision_sum = 0.0
    recall_sum = 0.0
    total_ngrams = 0

    for n in range(1, max_n + 1):
        ref_ngrams = get_char_ngrams(reference, n)
        cand_ngrams = get_char_ngrams(candidate, n)

        if not cand_ngrams:
            continue

        # Precision
        matched = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total_cand = sum(cand_ngrams.values())
        total_ref = sum(ref_ngrams.values())

        if total_cand > 0:
            precision_sum += matched / total_cand
        if total_ref > 0:
            recall_sum += matched / total_ref

        total_ngrams += 1

    if total_ngrams == 0:
        return 0.0

    precision = precision_sum / total_ngrams
    recall = recall_sum / total_ngrams

    if precision + recall == 0:
        return 0.0

    # F-score с beta
    beta2 = beta ** 2
    chrf = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)

    return round(chrf * 100, 2)  # Возвращаем в процентах


def calculate_ter(reference: str, candidate: str) -> float:
    """
    Упрощённая оценка TER (Translation Edit Rate)
    Чем меньше, тем лучше (0 = идеально)
    """
    if not candidate:
        return 1.0

    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()

    if not ref_words:
        return 0.0 if not cand_words else 1.0

    # Простая оценка: доля несовпадающих слов
    ref_set = set(ref_words)
    cand_set = set(cand_words)

    matches = len(ref_set & cand_set)
    total = max(len(ref_set), len(cand_set))

    ter = 1.0 - (matches / total if total > 0 else 0.0)
    return round(ter, 4)


def evaluate_reference_based(pair: TranslationPair) -> ReferenceBasedMetrics:
    """
    Оценивает качество перевода по эталону используя стандартные метрики
    """
    metrics = ReferenceBasedMetrics()

    # BLEU
    metrics.bleu = calculate_bleu(pair.reference, pair.candidate)

    # chrF
    metrics.chrf = calculate_chrf(pair.reference, pair.candidate)

    # TER (инвертируем для совместимости - больше = лучше)
    ter_raw = calculate_ter(pair.reference, pair.candidate)
    metrics.ter = round((1.0 - ter_raw) * 100, 2)

    # METEOR (упрощённая эвристика на основе F1)
    ref_words = set(pair.reference.lower().split())
    cand_words = set(pair.candidate.lower().split())

    if ref_words and cand_words:
        intersection = len(ref_words & cand_words)
        precision = intersection / len(cand_words) if cand_words else 0
        recall = intersection / len(ref_words) if ref_words else 0
        if precision + recall > 0:
            metrics.meteor = round(2 * precision * recall / (precision + recall) * 100, 2)

    # COMET (эвристика на основе semantic similarity)
    # В реальной реализации нужно использовать pretrained COMET модель
    # Для reference-based оценки используем прямое сравнение reference и candidate
    sem_result = evaluate_semantic_similarity(pair.reference, pair.candidate, back_translated_text=None)
    if sem_result and "cosine_similarity" in sem_result:
        metrics.comet = round(sem_result["cosine_similarity"] * 100, 2)
    else:
        metrics.comet = 50.0  # Значение по умолчанию при ошибке

    # Общий score (средневзвешенный)
    weights = {"bleu": 0.3, "chrf": 0.25, "ter": 0.2, "meteor": 0.15, "comet": 0.1}
    metrics.overall_score = round(
        weights["bleu"] * metrics.bleu +
        weights["chrf"] * metrics.chrf +
        weights["ter"] * metrics.ter +
        weights["meteor"] * metrics.meteor +
        weights["comet"] * metrics.comet,
        2
    )

    return metrics


def evaluate_reference_free(pair: TranslationPair, back_translated_text: Optional[str] = None) -> ReferenceFreeMetrics:
    """
    Оценивает качество перевода без эталона используя quality_assessment модуль

    Args:
        pair: пара источник-кандидат
        back_translated_text: обратный перевод (candidate -> source_lang), опционально
    """
    metrics = ReferenceFreeMetrics()

    # Логирование для отладки
    logger.info(f"[RefFree DEBUG] Source: {pair.source[:100]}...")
    logger.info(f"[RefFree DEBUG] Candidate: {pair.candidate[:100]}...")
    logger.info(f"[RefFree DEBUG] Back-translated: {back_translated_text[:100] if back_translated_text else 'None'}...")

    # Semantic Similarity между source и candidate с использованием обратного перевода
    sem_result = evaluate_semantic_similarity(
        pair.source,
        pair.candidate,
        back_translated_text=back_translated_text
    )
    if sem_result and "quality_score" in sem_result:
        metrics.semantic_similarity = round(sem_result["quality_score"], 2)
    else:
        metrics.semantic_similarity = 50.0

    logger.info(f"[RefFree DEBUG] Semantic score: {metrics.semantic_similarity}")

    # Cross-Entropy Quality
    ce_result = evaluate_cross_entropy(pair.source, pair.candidate)
    if ce_result and "quality_score" in ce_result:
        metrics.cross_entropy_quality = round(ce_result["quality_score"], 2)
    else:
        metrics.cross_entropy_quality = 50.0

    logger.info(f"[RefFree DEBUG] Cross-entropy score: {metrics.cross_entropy_quality}")

    # NER Consistency
    ner_result = evaluate_ner_consistency(pair.source, pair.candidate)
    metrics.ner_consistency = round(ner_result["quality_score"], 2)

    logger.info(f"[RefFree DEBUG] NER score: {metrics.ner_consistency}")

    # Round-trip Consistency (если есть обратный перевод) - ВАЖНО: основной метод оценки
    if back_translated_text:
        rt_result = evaluate_roundtrip_consistency(pair.source, pair.candidate, back_translated_text)
        if rt_result and "quality_score" in rt_result:
            metrics.roundtrip_consistency = round(rt_result["quality_score"] * 100, 2)
        else:
            metrics.roundtrip_consistency = 50.0
        logger.info(f"[RefFree DEBUG] Roundtrip score: {metrics.roundtrip_consistency}")
    else:
        metrics.roundtrip_consistency = 0.0  # Не вычислялось
        logger.warning("[RefFree DEBUG] Нет back_translated_text, roundtrip = 0")

    # Общий score (средневзвешенный) - нормализуем к 0-100
    # Увеличиваем вес roundtrip и semantic как наиболее коррелирующих с человеческой оценкой
    weights = {"semantic": 0.35, "cross_entropy": 0.20, "ner": 0.15, "roundtrip": 0.30}

    # Если есть roundtrip, используем его, иначе перераспределяем веса
    if metrics.roundtrip_consistency > 0:
        metrics.overall_score = round(
            weights["semantic"] * metrics.semantic_similarity +
            weights["cross_entropy"] * metrics.cross_entropy_quality +
            weights["ner"] * metrics.ner_consistency +
            weights["roundtrip"] * metrics.roundtrip_consistency,
            2
        )
    else:
        # Без roundtrip увеличиваем веса остальных метрик
        adjusted_weights = {"semantic": 0.45, "cross_entropy": 0.25, "ner": 0.20, "roundtrip": 0.10}
        metrics.overall_score = round(
            adjusted_weights["semantic"] * metrics.semantic_similarity +
            adjusted_weights["cross_entropy"] * metrics.cross_entropy_quality +
            adjusted_weights["ner"] * metrics.ner_consistency,
            2
        )

    # Ограничиваем диапазон 0-100
    metrics.overall_score = min(100.0, max(0.0, metrics.overall_score))

    logger.info(f"[RefFree DEBUG] Overall score: {metrics.overall_score}")

    return metrics


def compare_scores(ref_metrics: ReferenceBasedMetrics, my_metrics: ReferenceFreeMetrics) -> Tuple[float, str]:
    """
    Сравнивает оценки от двух методов

    Returns:
        разница и маркер корреляции
    """
    # Нормализуем к диапазону 0-1
    ref_normalized = ref_metrics.overall_score / 100.0
    my_normalized = my_metrics.overall_score / 100.0

    diff = abs(ref_normalized - my_normalized)

    if diff < 0.1:
        marker = "match"  # Хорошая корреляция
    elif diff < 0.25:
        marker = "partial"  # Частичная корреляция
    else:
        marker = "diverge"  # Расхождение

    return round(diff, 4), marker


def process_dataset(
        dataset_path: str,
        output_path: str,
        lm_client: Optional[LMStudioClient] = None,
        max_samples: int = 100,
        use_lm_translation: bool = True
):
    """
    Обрабатывает датасет и сравнивает методы оценки

    Args:
        dataset_path: путь к датасету
        output_path: путь для сохранения результатов
        lm_client: клиент LM Studio (опционально)
        max_samples: максимальное количество образцов для обработки
        use_lm_translation: использовать ли LM Studio для перевода
    """

    # Загружаем датасет
    pairs = parse_dataset(dataset_path, max_lines=max_samples)

    if not pairs:
        logger.error("Нет данных для обработки")
        return

    results: List[ComparisonResult] = []

    lm_available = lm_client.is_available() if lm_client else False

    if use_lm_translation and not lm_available:
        logger.warning("LM Studio недоступен. Используем заглушки для candidate переводов.")
        use_lm_translation = False

    logger.info(f"Обработка {len(pairs)} пар...")
    logger.info(f"LM Studio доступен: {lm_available}")

    for idx, pair in enumerate(pairs, 1):
        logger.info(f"[{idx}/{len(pairs)}] Обработка пары...")

        # Получаем candidate перевод
        back_translated = None
        if use_lm_translation and lm_client:
            # Прямой перевод RU -> EN
            candidate = lm_client.translate(pair.source, "ru", "en")
            if candidate:
                pair.candidate = candidate
                # Обратный перевод EN -> RU для round-trip проверки
                back_translated = lm_client.translate(pair.candidate, "en", "ru")
                logger.info(f"  Обратный перевод получен: {back_translated is not None}")
                if back_translated:
                    logger.info(f"  [DEBUG] Candidate (EN): {pair.candidate[:150]}...")
                    logger.info(f"  [DEBUG] Back-translated (RU): {back_translated[:150]}...")
                    logger.info(f"  [DEBUG] Original (RU): {pair.source[:150]}...")
            else:
                logger.warning(f"Не удалось получить перевод для пары {idx}, используем reference как candidate")
                pair.candidate = pair.reference
        else:
            # Для тестирования используем reference как candidate
            # В реальности здесь должен быть перевод от вашей модели
            pair.candidate = pair.reference
            logger.info(f"  [INFO] Используем reference как candidate (тестовый режим)")
            # При использовании reference как candidate, back_translated будет source
            # Это даёт идеальную round-trip проверку для тестирования
            if not back_translated:
                back_translated = pair.source
                logger.info(f"  [DEBUG] Back-translated установлен в original source для тестирования")

        # Оцениваем по эталону
        ref_metrics = evaluate_reference_based(pair)

        # Оцениваем без эталона (мои метрики) с обратным переводом если доступен
        my_metrics = evaluate_reference_free(pair, back_translated_text=back_translated)

        # Сравниваем
        diff, marker = compare_scores(ref_metrics, my_metrics)

        # Сохраняем результат
        result = ComparisonResult(
            pair_id=idx,
            source_preview=pair.source[:100] + ("..." if len(pair.source) > 100 else ""),
            reference_preview=pair.reference[:100] + ("..." if len(pair.reference) > 100 else ""),
            candidate_preview=pair.candidate[:100] + ("..." if len(pair.candidate) > 100 else ""),
            source_full=pair.source,
            reference_full=pair.reference,
            candidate_full=pair.candidate,
            back_translated_full=back_translated or "",
            ref_bleu=ref_metrics.bleu,
            ref_meteor=ref_metrics.meteor,
            ref_chrf=ref_metrics.chrf,
            ref_overall=ref_metrics.overall_score,
            my_semantic=my_metrics.semantic_similarity,
            my_cross_entropy=my_metrics.cross_entropy_quality,
            my_ner=my_metrics.ner_consistency,
            my_roundtrip=my_metrics.roundtrip_consistency,
            my_overall=my_metrics.overall_score,
            difference=diff,
            correlation_marker=marker
        )
        results.append(result)

        logger.info(f"  Ref Overall: {ref_metrics.overall_score}, My Overall: {my_metrics.overall_score}, "
                    f"Diff: {diff}, Marker: {marker}")

    # Сохраняем результаты
    save_results(results, output_path)

    # Статистика
    print_statistics(results)


def save_results(results: List[ComparisonResult], output_path: str):
    """Сохраняет результаты в JSON и CSV форматы"""

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = str(Path(output_path).with_suffix('.json'))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    logger.info(f"Результаты сохранены в {json_path}")

    # CSV
    csv_path = str(Path(output_path).with_suffix('.csv'))
    with open(csv_path, 'w', encoding='utf-8') as f:
        # Заголовок
        f.write("pair_id,ref_overall,my_overall,difference,marker,ref_bleu,ref_chrf,my_semantic,my_ce,my_ner\n")
        for r in results:
            f.write(f"{r.pair_id},{r.ref_overall},{r.my_overall},{r.difference},{r.correlation_marker},"
                    f"{r.ref_bleu},{r.ref_chrf},{r.my_semantic},{r.my_cross_entropy},{r.my_ner}\n")
    logger.info(f"CSV сохранён в {csv_path}")


def print_statistics(results: List[ComparisonResult]):
    """Выводит статистику по результатам"""

    if not results:
        return

    match_count = sum(1 for r in results if r.correlation_marker == "match")
    partial_count = sum(1 for r in results if r.correlation_marker == "partial")
    diverge_count = sum(1 for r in results if r.correlation_marker == "diverge")

    avg_diff = sum(r.difference for r in results) / len(results)

    avg_ref = sum(r.ref_overall for r in results) / len(results)
    avg_my = sum(r.my_overall for r in results) / len(results)

    print("\n" + "=" * 80)
    print("СТАТИСТИКА СРАВНЕНИЯ МЕТОДОВ ОЦЕНКИ")
    print("=" * 80)
    print(f"Всего пар: {len(results)}")
    print(f"\nКорреляция:")
    print(f"  Match (< 0.1):     {match_count:4d} ({match_count/len(results)*100:5.1f}%)")
    print(f"  Partial (< 0.25):  {partial_count:4d} ({partial_count/len(results)*100:5.1f}%)")
    print(f"  Diverge (>= 0.25): {diverge_count:4d} ({diverge_count/len(results)*100:5.1f}%)")
    print(f"\nСредние значения:")
    print(f"  Средняя разница: {avg_diff:.4f}")
    print(f"  Средняя оценка (reference-based): {avg_ref:.2f}")
    print(f"  Средняя оценка (reference-free):  {avg_my:.2f}")
    print("=" * 80)


def main():
    """Точка входа"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Сравнение методов оценки качества перевода",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Обработать 50 пар с использованием LM Studio для перевода
  python main.py --dataset datasets/rus-eng-part1-text.txt --samples 50 --use-lm

  # Обработать без LM Studio (использовать reference как candidate для тестирования)
  python main.py --dataset datasets/rus-eng-part1-text.txt --samples 20

  # Проверить доступность LM Studio
  python main.py --check-lm
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/rus-eng-part1-text.txt",
        help="Путь к датасету (по умолчанию: datasets/rus-eng-part1-text.txt)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison_results",
        help="Путь для сохранения результатов (по умолчанию: results/comparison_results)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Максимальное количество образцов для обработки (по умолчанию: 20)"
    )

    parser.add_argument(
        "--lm-url",
        type=str,
        default="http://localhost:1234",
        help="URL LM Studio API (по умолчанию: http://localhost:1234)"
    )

    parser.add_argument(
        "--lm-model",
        type=str,
        default="gigachat3.1-10b-a1.8b",
        help="Модель LM Studio (по умолчанию: gigachat3.1-10b-a1.8b)"
    )

    parser.add_argument(
        "--use-lm",
        action="store_true",
        help="Использовать LM Studio для получения candidate переводов"
    )

    parser.add_argument(
        "--check-lm",
        action="store_true",
        help="Проверить доступность LM Studio и выйти"
    )

    args = parser.parse_args()

    # Проверка LM Studio
    if args.check_lm:
        client = LMStudioClient(args.lm_url, args.lm_model)
        if client.is_available():
            print(f"✓ LM Studio доступен по адресу {args.lm_url}")
            print(f"  Модель: {args.lm_model}")
        else:
            print(f"✗ LM Studio недоступен по адресу {args.lm_url}")
            print("  Убедитесь, что LM Studio запущен с открытым API сервером")
        return

    # Инициализация клиента LM Studio
    lm_client = None
    if args.use_lm:
        lm_client = LMStudioClient(args.lm_url, args.lm_model)
        if not lm_client.is_available():
            logger.warning(f"LM Studio недоступен по {args.lm_url}. Продолжаем без перевода.")
            lm_client = None

    # Обработка датасета
    process_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        lm_client=lm_client,
        max_samples=args.samples,
        use_lm_translation=(lm_client is not None)
    )


if __name__ == "__main__":
    main()