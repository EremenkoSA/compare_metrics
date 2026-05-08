from PySide6.QtCore import QObject
from src.core.event_bus import get_event_bus
from src.ui.overlay_display import TranslationOverlay
from src.quality_assessment import evaluate_comprehensive_quality
import requests
import json
import logging

# Настройка логгера для этого модуля, если нужно отдельное оформление
logger = logging.getLogger(__name__)

class LocalTranslator(QObject):
    """Универсальный локальный / OpenAI-совместимый переводчик."""

    def __init__(self, endpoint="http://localhost:1234/v1/chat/completions",
                 model="hunyuan-mt-7b", target_lang="ru",
                 api_key=None, api_format="openai",
                 provider_name="Local", parent=None):
        super().__init__(parent)
        self.endpoint = endpoint
        self.model = model
        self.target_lang = target_lang
        self.api_key = api_key
        self.api_format = api_format
        self.provider_name = provider_name       # отображаемое имя
        self.override_prompt_name = None         # временное переопределение промта
        self.event_bus = get_event_bus()
        self.settings = None  # Будет установлено извне при необходимости

        # Флаг включения проверки качества
        self.enable_quality_check = True

        print(f"[LocalTranslator] Инициализирован: endpoint={endpoint}, model={model}, format={api_format}")

    def translate_sync(self, text: str, target_lang: str = None, source_lang: str = "auto"):
        """Синхронная версия перевода для использования в Worker потоках."""
        lang = target_lang or self.target_lang
        quality_score = None

        # Получаем выбранный пользовательский промт из настроек
        from src.core.settings_manager import SettingsManager
        settings = SettingsManager()

        # Приоритет: override_prompt_name > selected_prompt
        selected_prompt_name = self.override_prompt_name if self.override_prompt_name else settings.get("selected_prompt")
        custom_prompts = settings.get("custom_prompts", [])

        # Базовый системный промт (всегда используется)
        base_system_instruction = (
            f"You are a professional translator. Translate the following text from {source_lang} to {lang}. "
            "Return ONLY the translated text, without any explanations."
        )

        # Пользовательское уточнение (если выбрано)
        user_instruction = ""
        if selected_prompt_name:
            for p in custom_prompts:
                if p["name"] == selected_prompt_name:
                    # Используем сгенерированный промт если есть
                    user_instruction = p.get("generated_prompt", "")
                    if not user_instruction:
                        # Если промт ещё не сгенерирован, формируем короткую инструкцию из данных
                        parts = []
                        sphere = p.get("sphere", "")
                        use_slang = p.get("use_slang", False)
                        comment = p.get("comment", "")

                        if sphere:
                            parts.append(f"Context: {sphere}.")
                        if use_slang:
                            parts.append("Style: Use informal language and slang.")
                        if comment:
                            clean_comment = comment[:100].replace('"', "'")
                            parts.append(f"Requirement: {clean_comment}")

                        if parts:
                            user_instruction = " ".join(parts)
                    break

        # Объединяем базовый и пользовательский промты
        if user_instruction:
            system_instruction = base_system_instruction + "\n\n" + user_instruction
        else:
            system_instruction = base_system_instruction

        # Формирование payload
        if self.api_format == "native":
            payload = {
                "model": self.model,
                "system_prompt": system_instruction,
                "input": text,
                "temperature": 0.0
            }
        else:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": text}
            ]
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 1024
            }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # === ЛОГИРОВАНИЕ ЗАПРОСА ===
        print("\n" + "=" * 80)
        print(f"[LM STUDIO REQUEST] Endpoint: {self.endpoint}")
        print(f"[LM STUDIO REQUEST] Model: {self.model}")
        print("-" * 40 + " PAYLOAD " + "-" * 40)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("=" * 80 + "\n")

        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=15)
            print(f"[LM STUDIO RESPONSE] HTTP статус: {resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()

                # === ЛОГИРОВАНИЕ ОТВЕТА ===
                print("-" * 40 + " RAW RESPONSE " + "-" * 40)
                print(json.dumps(data, indent=2, ensure_ascii=False))

                translated = self._extract_translation(data)

                print("-" * 40 + " EXTRACTED TRANSLATION " + "-" * 40)
                print(f"Original:   {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"Translated: {translated}")
                print("=" * 80 + "\n")

                # Обратная проверка качества (если включена)
                if self.enable_quality_check:
                    print("[QualityCheck] Выполняем обратный перевод для оценки...")
                    back_translated = self._translate_back(translated, source_lang, lang)

                    print("-" * 40 + " BACK TRANSLATION DATA " + "-" * 40)
                    print(f"Original (Source):      {text[:100]}{'...' if len(text) > 100 else ''}")
                    print(f"Translated (Target):    {translated[:100]}{'...' if len(translated) > 100 else ''}")
                    print(f"Back-Translated (Src):  {back_translated[:100]}{'...' if len(back_translated) > 100 else ''}")
                    print("=" * 80 + "\n")

                    # Вызов функции оценки с явным логированием входных данных
                    print("[QualityCheck] Вызов evaluate_comprehensive_quality...")
                    quality_result = evaluate_comprehensive_quality(
                        original_text=text,
                        translated_text=translated,
                        back_translated_text=back_translated,
                        translation_method=self.provider_name,
                        model_name=self.model
                    )

                    print("-" * 40 + " QUALITY RESULT " + "-" * 40)
                    print(json.dumps(quality_result, indent=2, ensure_ascii=False))
                    print("=" * 80 + "\n")

                    quality_score = quality_result.get("overall_quality")

                return translated, quality_score
            else:
                err = f"Ошибка API (HTTP {resp.status_code}): {resp.text}"
                print(f"[LM STUDIO ERROR] {err}")
                return err, None

        except requests.exceptions.ConnectionError:
            msg = "Ошибка: сервер не запущен. Проверьте endpoint " + self.endpoint
            print(f"[LM STUDIO CONNECTION ERROR] {msg}")
            return msg, None
        except Exception as e:
            msg = f"Ошибка локального перевода: {e}"
            print(f"[LM STUDIO EXCEPTION] {msg}")
            import traceback
            traceback.print_exc()
            return msg, None

    def translate(self, text: str, target_lang: str = None, source_lang: str = "auto"):
        """Асинхронная версия перевода через signal/slot."""
        result, quality_score = self.translate_sync(text, target_lang, source_lang)

        # Если есть settings, используем уникальное окно для этого провайдера
        if self.settings:
            window_id = f"local_{self.provider_name}".replace(" ", "_").replace("/", "_")
            TranslationOverlay.show_translation_for(self.settings, window_id, result, text, self.provider_name, quality_score)
        else:
            self.event_bus.translation_ready.emit(result, text, self.provider_name)

    def _extract_translation(self, data):
        if "response" in data:
            return data["response"].strip()
        if "output" in data and isinstance(data["output"], list):
            for item in data["output"]:
                if isinstance(item, dict) and "content" in item:
                    return item["content"].strip()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        return str(data).strip()

    def _translate_back(self, translated_text: str, source_lang: str, target_lang: str) -> str:
        """
        Выполняет обратный перевод для проверки качества.
        Логирование уже добавлено в вызывающий метод, здесь только чистая логика.
        """
        # Определяем язык источника для обратного перевода
        src_lang_clean = source_lang if source_lang != 'auto' else 'en'
        tgt_lang_clean = target_lang

        back_system_instruction = (
            f"You are a professional translator. Translate the following text from {tgt_lang_clean} to {src_lang_clean}. "
            "Return ONLY the translated text, without any explanations."
        )

        if self.api_format == "native":
            payload = {
                "model": self.model,
                "system_prompt": back_system_instruction,
                "input": translated_text,
                "temperature": 0.0
            }
        else:
            messages = [
                {"role": "system", "content": back_system_instruction},
                {"role": "user", "content": translated_text}
            ]
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 1024
            }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            # Тихий запрос (логирование делается в translate_sync перед вызовом этой функции)
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return self._extract_translation(data)
            else:
                return f"Ошибка обратного перевода (HTTP {resp.status_code})"
        except Exception as e:
            return f"Ошибка обратного перевода: {e}"