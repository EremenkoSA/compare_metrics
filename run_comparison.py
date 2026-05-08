import json
import time
import requests
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sacrebleu import sentence_chrf

# Импорт вашей функции
from my_evaluator import on_the_fly_score

# ================= НАСТРОЙКИ =================
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"   # обновлённый API LM Studio
MODEL_NAME = "gigachat3.1-10b-a1.8b"
DATASET_PATH = "data/dataset.jsonl"
OUTPUT_CSV = "results/evaluated.csv"
PLOT_PATH = "results/correlation.png"
SLEEP_BETWEEN_CALLS = 1.0
# ==============================================

def load_dataset(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def translate_text(text: str, direction: str = "en->ru") -> str:
    """
    Отправляет запрос в LM Studio.
    direction: 'en->ru' или 'ru->en' – определяет system_prompt.
    """
    if direction == "en->ru":
        system_msg = "Translate the following English text to Russian. Return only the translation."
    else:
        system_msg = "Translate the following Russian text to English. Return only the translation."

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text.strip()}
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }
    try:
        resp = requests.post(LMSTUDIO_URL, json=payload, timeout=90)
        resp.raise_for_status()
        # LM Studio OpenAI-совместимый ответ
        result = resp.json()
        translation = result["choices"][0]["message"]["content"].strip()
        return translation
    except Exception as e:
        print(f"Ошибка перевода: {e}")
        return ""

def reference_metric(reference: str, candidate: str) -> float:
    """chrF как эталонная метрика (0..1)."""
    return sentence_chrf(candidate, [reference]).score / 100.0

def main():
    # 1. Загрузка данных
    dataset = load_dataset(DATASET_PATH)
    print(f"Загружено {len(dataset)} примеров")

    results = []
    for i, item in enumerate(dataset):
        source = item["source"]
        reference = item["reference"]
        print(f"\n[{i+1}/{len(dataset)}] src: {source[:60]}...")

        # 2. Прямой перевод (en->ru)
        candidate = translate_text(source, "en->ru")
        time.sleep(SLEEP_BETWEEN_CALLS)

        # 3. Обратный перевод (ru->en) для вашей оценки
        back_translation = translate_text(candidate, "ru->en") if candidate else ""
        time.sleep(SLEEP_BETWEEN_CALLS)

        # 4. Эталонная метрика
        ref_score = reference_metric(reference, candidate)

        # 5. Ваша оценка "на лету" (использует source, candidate и back_translation)
        your_score = on_the_fly_score(source, candidate, back_translation)

        results.append({
            "source": source,
            "reference": reference,
            "candidate": candidate,
            "back_translation": back_translation,
            "ref_metric_chrF": ref_score,
            "your_score": your_score
        })

        # частичное сохранение
        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    # 6. Сохранение и анализ
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

    mask = df["ref_metric_chrF"].notna() & df["your_score"].notna()
    filtered = df[mask]
    if len(filtered) < 5:
        print("⚠️ Недостаточно данных для корреляции")
        return

    pearson_r, p_pearson = pearsonr(filtered["ref_metric_chrF"], filtered["your_score"])
    spearman_r, p_spearman = spearmanr(filtered["ref_metric_chrF"], filtered["your_score"])

    print("\n===== Сравнение оценок =====")
    print(f"Pearson r  = {pearson_r:.3f} (p={p_pearson:.4f})")
    print(f"Spearman ρ = {spearman_r:.3f} (p={p_spearman:.4f})")

    # График
    plt.figure(figsize=(7,6))
    plt.scatter(filtered["ref_metric_chrF"], filtered["your_score"], alpha=0.6)
    plt.xlabel("Эталонная chrF")
    plt.ylabel("Ваша оценка (на лету)")
    plt.title(f"Корреляция (Spearman ρ = {spearman_r:.2f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()
    print(f"График сохранён: {PLOT_PATH}")

if __name__ == "__main__":
    main()