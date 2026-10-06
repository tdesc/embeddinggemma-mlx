# embeddinggemma-mlx

**EmbeddingGemma 2** (Google, 740M, Apache-2.0) — нативно мультимодальный
эмбеддер (текст, код, изображения, аудио, видео, PDF → единое пространство,
100+ языков, Matryoshka 768/512/256/128). Этот пакет — **текстовая башня на
Apple Silicon через MLX**: конвертация весов, last-token pooling, L2-норма,
Matryoshka-усечение и OpenAI-совместимый сервер `/v1/embeddings`.

## Статус (честно)

- ✅ Текст+код: pooling, Matryoshka, сервер, тесты (мок-режим — CI без весов).
- ⛔ Конвертация весов: `mlx_lm 0.31.3` не поддерживает `model_type: embedding_gemma2`
  (`ValueError: Model type embedding_gemma2 not supported`). Модель — sentence-transformers-пакет
  (modules.json/1_Pooling/2_Normalize) с единым model.safetensors на все башни; текстовая башня —
  наследница gemma3_text, но с новыми полями конфига (`per_layer_config`,
  `hidden_size_per_layer_input`, `layer_types`) — нужен реальный порт архитектуры, не переименование.
  Анализ и план — в issue #1.
- 🌉 Мост на сегодня: сама модель работает через `sentence-transformers` на MPS
  (`pip install sentence-transformers; SentenceTransformer('google/embeddinggemma-2')`),
  dim 768 — наш сервер и Colony-провайдер примут её, когда порт весов появится.
- 🚧 Мультимодальные башни (vision/audio) — вне этого пакета (второй волной).

## Установка

```bash
pip install -e ".[mlx,dev]"
```

## Конвертация весов (fp16 — без квантования!)

```bash
python -m embeddinggemma_mlx.convert --hf-path google/embeddinggemma-2 \
    --mlx-path ./embeddinggemma-2-mlx
```

Эмбеддинги чувствительны к сжатию весов: `--quantize` доступен, но не
рекомендуется — сначала канарей качества.

## Сервер

```bash
python -m embeddinggemma_mlx.server --model ./embeddinggemma-2-mlx --port 7341
EMBEDDINGGEMMA_MOCK=1 python -m embeddinggemma_mlx.server   # CI/демо без весов

curl localhost:7341/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"embeddinggemma-2-mlx","input":["привет мир","hello world"],"dims":256}'
```

Ответ — OpenAI-формы: `data[].embedding`, `usage.total_tokens`, плюс
`served_by: mlx_lm|mock` и `latency_ms`.

## Контракт

- **Pooling**: last-token (EOS добавляется к входу) — рецепт Gemma-эмбеддеров.
- **Норма**: L2 всегда.
- **Matryoshka**: `dims ∈ {768,512,256,128}` — срез первых компонент + перенорма;
  вложенное усечение консистентно (256-префикс 512 == 256 напрямую — тестом).
- **Потоки**: все MLX-форварды на одном диспетчер-потоке (Metal thread-local).
- **Фейлы громкие**: пустой input → 400; неизвестная модель → 404; dims не из
  списка → 400.

## Библиотека

```python
from embeddinggemma_mlx import MlxEmbeddingGemma

emb = MlxEmbeddingGemma("./embeddinggemma-2-mlx", dims=256)
vec = emb.embed("какая рука ближе к петле?").embedding  # np.float32[256], ||v||=1
```

## Зачем это нужно (наш кейс)

Локальная память на qdrant сегодня сидит на англо-центричном MiniLM-384;
мультиязычный эмбеддер делает RU/EN/ZH-recall семантическим на той же машине
(740M ≈ 1.5 ГБ fp16 — рядом с 7B-чатом незаметно). Канарей-скрипт сравнения —
`scripts/canary_ru_en.py` (опциональные зависимости).

## Лицензии

Код — MIT. Конвертированные веса производны от `google/embeddinggemma-2`
и остаются Apache-2.0 от Google (см. LICENSE).
