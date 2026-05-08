# my_evaluator.py
"""
Обёртка для вызова комплексной оценки из quality_assessment.py
Точная интеграция вашей реальной метрики.
"""
import sys
import os

# Добавляем путь к папке src, где лежит quality_assessment.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quality_assessment import evaluate_comprehensive_quality

def on_the_fly_score(source: str, candidate: str, back_translation: str) -> float:
    """
    Ваша оригинальная метрика «на лету» из Screen_translator1.

    Возвращает общую оценку quality (от 0 до 1), среднее арифметическое
    четырёх методов: семантическая близость, перплексия, NER и round-trip.
    """
    result = evaluate_comprehensive_quality(
        original_text=source,
        translated_text=candidate,
        back_translated_text=back_translation if back_translation else None,
        translation_method="LM Studio",
        model_name="gigachat3.1-10b-a1.8b"
    )
    # В комплексной оценке всегда есть поле overall_quality
    return result.get("overall_quality", 0.0)