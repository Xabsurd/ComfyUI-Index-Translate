"""ComfyUI node for local Index-Translate text translation.

The model is downloaded lazily. Hugging Face is attempted first and ModelScope
is used as a fallback, which is useful on networks where huggingface.co is not
reachable.
"""

import os
import re
import threading
import gc
from pathlib import Path


MODEL_IDS = {
    "2B (默认)": "IndexTeam/Index-Translate-2B",
    "9B": "IndexTeam/Index-Translate-9B",
    "35B-A3B (preview)": "IndexTeam/Index-Translate-35B-A3B-preview",
}

LANG_NAMES = {
    "en": "英语", "zh": "中文", "de": "德语", "fr": "法语",
    "es": "西班牙语", "ja": "日语", "ko": "韩语", "pt": "葡萄牙语",
    "ru": "俄语", "ar": "阿拉伯语", "it": "意大利语", "nl": "荷兰语",
    "pl": "波兰语", "ro": "罗马尼亚语", "sv": "瑞典语", "tr": "土耳其语",
    "hi": "印地语", "vi": "越南语", "th": "泰语", "id": "印尼语",
    "ms": "马来语", "fil": "菲律宾语",
    "ukr_cyrl": "乌克兰语", "fas_arab": "波斯语", "ces_latn": "捷克语",
    "ell_grek": "希腊语", "dan_latn": "丹麦语", "hun_latn": "匈牙利语",
    "fin_latn": "芬兰语", "nob_latn": "书面挪威语", "slk_latn": "斯洛伐克语",
    "bul_cyrl": "保加利亚语",
}

# Native names keep the language menu readable in every UI locale.
LANG_OPTIONS = [
    "en · English", "zh · 中文", "de · Deutsch", "fr · Français",
    "es · Español", "ja · 日本語", "ko · 한국어", "pt · Português",
    "ru · Русский", "ar · العربية", "it · Italiano", "nl · Nederlands",
    "pl · Polski", "ro · Română", "sv · Svenska", "tr · Türkçe",
    "hi · हिन्दी", "vi · Tiếng Việt", "th · ไทย", "id · Bahasa Indonesia",
    "ms · Bahasa Melayu", "fil · Filipino", "ukr_cyrl · Українська",
    "fas_arab · فارسی", "ces_latn · Čeština", "ell_grek · Ελληνικά",
    "dan_latn · Dansk", "hun_latn · Magyar", "fin_latn · Suomi",
    "nob_latn · Norsk bokmål", "slk_latn · Slovenčina", "bul_cyrl · Български",
]

_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _model_dir(model_id: str) -> Path:
    override = os.environ.get("INDEX_TRANSLATE_MODEL_DIR")
    if override:
        return Path(override).expanduser() / model_id.rsplit("/", 1)[-1]
    try:
        import folder_paths
        root = Path(folder_paths.models_dir)
    except Exception:
        root = Path(__file__).resolve().parents[2] / "models"
    return root / "index_translate" / model_id.rsplit("/", 1)[-1]


def _download_model(model_id: str) -> str:
    target = _model_dir(model_id)
    if (target / "config.json").is_file():
        return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    hf_error = None
    try:
        from huggingface_hub import snapshot_download
        print(f"[Index-Translate] Downloading {model_id} from Hugging Face...")
        snapshot_download(repo_id=model_id, local_dir=str(target))
        return str(target)
    except Exception as exc:
        hf_error = exc
        print(f"[Index-Translate] Hugging Face download failed: {exc}")

    try:
        from modelscope import snapshot_download
        print(f"[Index-Translate] Downloading {model_id} from ModelScope...")
        snapshot_download(model_id, local_dir=str(target))
        if not (target / "config.json").is_file():
            raise RuntimeError(f"ModelScope download did not produce {target / 'config.json'}")
        return str(target)
    except Exception as ms_error:
        raise RuntimeError(
            f"无法下载 {model_id}。Hugging Face 错误: {hf_error}; "
            f"ModelScope 错误: {ms_error}"
        ) from ms_error


def _load_model(model_id: str):
    with _CACHE_LOCK:
        if model_id in _CACHE:
            tokenizer, model = _CACHE[model_id]
            try:
                import torch
                if torch.cuda.is_available() and next(model.parameters()).device.type == "cpu":
                    model.to("cuda")
            except StopIteration:
                pass
            return tokenizer, model
        model_path = _download_model(model_id)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 Index-Translate 依赖，请在 ComfyUI 使用的 conda 环境中执行 "
                "pip install -r custom_nodes/ComfyUI-Index-Translate/requirements.txt"
            ) from exc

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
        else:
            load_kwargs = {"torch_dtype": torch.float32}
        print(f"[Index-Translate] Loading {model_id} from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **load_kwargs
        )
        if not torch.cuda.is_available():
            model.to("cpu")
        model.eval()
        _CACHE[model_id] = (tokenizer, model)
        return _CACHE[model_id]


def _offload_model(model):
    """Keep the model in host RAM between executions and release VRAM."""
    try:
        import torch
        model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[Index-Translate] Model offloaded to CPU RAM; CUDA cache released.")
    except Exception as exc:
        print(f"[Index-Translate] Warning: model offload failed: {exc}")


def _glossary_items(glossary: str):
    items = []
    for item in glossary.replace("，", ",").split(","):
        item = item.strip()
        if not item:
            continue
        for separator in ("->", "→", ":", "："):
            if separator in item:
                left, right = item.split(separator, 1)
                items.append(f"{left.strip()}→{right.strip()}")
                break
        else:
            items.append(item)
    return items


def _make_prompt(text, source, target, glossary, instruction, genre):
    source = source.split(" · ", 1)[0]
    target = target.split(" · ", 1)[0]
    source_name = LANG_NAMES.get(source.lower(), source) if source.lower() not in ("auto", "") else ""
    target_name = LANG_NAMES.get(target.lower(), target)
    constraints = []
    terms = _glossary_items(glossary)
    if terms:
        constraints.append("〖硬性要求〗专名/术语对照: " + "、".join(terms))
    if instruction.strip():
        constraints.extend(
            line.strip() if line.strip().startswith(("〖硬性要求〗", "〖注意〗", "〖软性要求〗"))
            else f"〖注意〗{line.strip()}" for line in instruction.splitlines() if line.strip()
        )
    header_source = f"{source_name}{genre}" if source_name else genre
    if constraints:
        numbered = "\n".join(f"{i}. {item}" for i, item in enumerate(constraints, 1))
        return (
            f"请将以下{header_source}翻译成{target_name}，并且严格遵循所有约束要求。\n\n"
            f"〖源文〗\n{text.strip()}\n\n〖约束要求〗\n{numbered}\n\n"
            "只输出译文，不要有任何额外说明。"
        )
    if source_name:
        return f"请将以下{source_name}文本翻译为{target_name}，直接输出翻译结果，不要进行任何解释。\n\n{text.strip()}"
    return f"请将以下文本翻译为{target_name}，直接输出翻译结果，不要进行任何解释。\n\n{text.strip()}"


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return re.sub(r"^<think>\s*", "", text.strip()).strip()


class IndexTranslate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": "你好，世界。"}),
            "target_language": (LANG_OPTIONS, {"default": LANG_OPTIONS[0]}),
            "source_language": (["auto"] + LANG_OPTIONS, {"default": "auto"}),
            "model": (list(MODEL_IDS), {"default": "2B (默认)"}),
            "glossary": ("STRING", {"multiline": True, "default": "", "tooltip": "term:translation, term2:translation2"}),
            "instruction": ("STRING", {"multiline": True, "default": ""}),
            "genre": ("STRING", {"default": "文本"}),
            "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        }, "optional": {
            "offload": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translation",)
    FUNCTION = "translate"
    CATEGORY = "text/translation"

    def translate(self, text, target_language, source_language, model, glossary, instruction,
                  genre, max_new_tokens, temperature, offload=True):
        if not text.strip():
            return ("",)
        import torch

        model_id = MODEL_IDS[model]
        tokenizer, translator = _load_model(model_id)
        prompt = _make_prompt(text, source_language, target_language, glossary, instruction, genre)
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = tokenizer(formatted, return_tensors="pt")
        device = getattr(translator, "device", next(translator.parameters()).device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
        try:
            with torch.inference_mode():
                output = translator.generate(**inputs, **generation_kwargs)
            generated = output[0, inputs["input_ids"].shape[-1]:]
            result = _strip_thinking(tokenizer.decode(generated, skip_special_tokens=True))
        finally:
            if offload:
                _offload_model(translator)
        return (result,)


class IndexTranslateSimple(IndexTranslate):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": ""}),
            "target_language": (LANG_OPTIONS, {"default": LANG_OPTIONS[0]}),
            "model": (list(MODEL_IDS), {"default": "2B (默认)"}),
        }, "optional": {
            "offload": ("BOOLEAN", {"default": True}),
        }}

    def translate(self, text, target_language, model, offload=True):
        return super().translate(
            text, target_language, "auto", model, "", "", "文本", 1024, 0.0,
            offload=offload,
        )


NODE_CLASS_MAPPINGS = {
    "IndexTranslateSimple": IndexTranslateSimple,
    "IndexTranslate": IndexTranslate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTranslateSimple": "Index-Translate",
    "IndexTranslate": "Index-Translate Pro",
}
