# prepare_dataset.py
import json
from datasets import load_dataset

print("Загрузка датасета Tatoeba (eng-rus)...")
# Загружаем сплит "train" — он самый полный
dataset = load_dataset("tatoeba", "eng-rus", split="train")

print(f"Загружено {len(dataset)} пар предложений.")

with open("data/dataset.jsonl", "w", encoding="utf-8") as f:
    for item in dataset:
        # В этом датасете поля называются 'source' и 'target'
        source = item['source']['String'] if isinstance(item['source'], dict) else item['source']
        target = item['target']['String'] if isinstance(item['target'], dict) else item['target']

        f.write(json.dumps({
            "source": source.strip(),
            "reference": target.strip()
        }, ensure_ascii=False) + "\n")

print("Готово! Данные сохранены в data/dataset.jsonl")