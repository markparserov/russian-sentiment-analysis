# Анализ тональности русскоязычных текстов

Трёхклассовая классификация тональности (`negative` / `neutral` / `positive`) для
русского текста. Сравниваются классические модели и дообученные (fine-tuned)
энкодеры предложений, с честной оценкой на полностью независимом наборе из
соцсетей (out-of-bag, OOB). Учебный проект (СУиАДоСП).

**Главный результат:** дообученный энкодер `ai-forever/ru-en-RoSBERTa` даёт
macro-F1 **0.81–0.90** in-domain и **0.77** на независимом OOB-наборе, обходя все
12 готовых русскоязычных моделей тональности.

---

## Данные

Корпус собран из ~15 открытых русскоязычных датасетов (отзывы, новости, твиты,
посты) — полный список источников со ссылками: [`datasets_sources.md`](datasets_sources.md).

- **~1.37 млн** примеров, **24 признака** (после дедупликации).
- `label ∈ {negative, neutral, positive}` — целевая переменная.
- `label_source_type` — происхождение метки: `human` (ручная разметка),
  `rating` (бинаризация числовых звёзд), `distant` (по эмодзи/смайлам).
- `domain ∈ {geo, grocery, clothing, social, news, blogs, twitter}`.
- Признаки: текст + категориальные (`source`, `domain`, `label_source_type`,
  `category`) + числовые (`original_rating`, `price` и инженерные:
  `n_chars`, `n_words`, `n_exclaim`, `upper_ratio`, `mean_word_len`, …).

**Два варианта меток** (для проверки устойчивости к шуму):

- **A** — все данные;
- **B** — без меток, выведенных из звёздного рейтинга (`label_source_type ≠ rating`).

**OOB-тест** — 1067 вручную размеченных постов из соцсетей
(`data/combined_platform_proportional_sample.xlsx`), полностью независимых от
обучения; используется только ручная колонка `human_sentiment`.

> Тяжёлые артефакты (`master.parquet`, обученные модели, предсказания) хранятся
> через **Git LFS** — после клонирования выполните `git lfs pull`.

## Предобработка и признаки

- Дедупликация по нормализованному ключу; конфликты меток разрешаются по
  приоритету источника **human > rating > distant**, противоречивые — отбрасываются.
- Удаление слишком коротких текстов.
- Инженерия текстовых признаков: пунктуация, капс, удлинение букв, компактный
  лексикон полярности и отрицание ([`src/text_features.py`](src/text_features.py)).
- Стратифицированные `train/val/test` сплиты, независимо для вариантов A и B.

## EDA и кластеризация

В ноутбуке: описательные статистики, баланс классов, распределения признаков,
выбросы, кросс-доменный анализ, проверка дедупликации (фигуры `01`–`07`).
Кластеризация эмбеддингов `USER-bge-m3`: подбор KMeans, UMAP-проекция, HDBSCAN
(фигуры `08`–`10`).

## Модели

**Классические** ([`src/oob_classical.py`](src/oob_classical.py), [`src/catboost_models.py`](src/catboost_models.py)):

- `TF-IDF + LinearSVC` — линейный референс;
- `Embeddings (USER-bge-m3) + LogReg` — референс (+ вариант с подбором `C` через Optuna);
- `CatBoost(text+feats)` — GPU-CatBoost по сырому тексту (встроенный токенизатор) + ручные признаки;
- `CatBoost(emb)` — GPU-CatBoost по замороженным эмбеддингам (1024-d).

**Дообучение энкодеров** ([`src/finetune.py`](src/finetune.py)). Кандидаты отобраны
по бенчмарку [MTEB](https://huggingface.co/spaces/mteb/leaderboard), русскоязычная
часть **RuMTEB**, по критериям:

- модели **до 500 млн параметров**;
- строго **≥90% обучающих данных НЕ из RuMTEB** (чтобы исключить «подсматривание» в бенчмарк).

Кандидаты: `deepvk/USER-bge-m3`, `deepvk/USER2-base`, `ai-forever/ru-en-RoSBERTa`,
`sergeyzh/rubert-mini-frida`. Модель на базе **Gemma** (`google/embeddinggemma-300m`),
подходящая по критериям, **исключена** из-за несовместимости с архитектурой
доступных при файн-тюнинге GPU (Tesla V100, Volta). Случайный поиск (128 триалов)
по энкодеру и гиперпараметрам выбрал `ai-forever/ru-en-RoSBERTa`
(lr 2e-5, 3 эпохи, batch 16, max_len 512); финальное обучение — на вариантах A и B.

**Внешние модели** для честного сравнения на OOB ([`src/oob_external.py`](src/oob_external.py)):
12 готовых русских/мультиязычных моделей тональности (cardiff-xlmr, deepvk GeRaCl,
seara-rubert, blanchefort, cointegrated, mDeBERTa-NLI, clapAI, tabularisai и др.).

## Результаты (последний прогон)

### In-domain (отложенный тест), macro-F1

| Модель | A→A | B→B | A→B (перенос) |
|---|---|---|---|
| **ru-en-RoSBERTa (fine-tuned)** | **0.809** | **0.897** | **0.889** |
| CatBoost(text+feats) | 0.736 | 0.859 | 0.843 |
| Emb+LogReg (tuned) | 0.752 | 0.823 | 0.782 |
| CatBoost(emb) | 0.737 | 0.796 | 0.722 |
| TF-IDF + LinearSVC | 0.689 | 0.661 | 0.678 |

![Сравнение моделей](reports/figures/12_model_comparison.png)

### OOB — независимая разметка соцсетей (n=1067), macro-F1

| Модель | macro-F1 | |
|---|---|---|
| **ru-en-RoSBERTa (fine-tuned, B)** | **0.769** | наша |
| cardiff-xlmr | 0.753 | внешняя |
| GeRaCl (deepvk USER2) | 0.721 | внешняя |
| ru-en-RoSBERTa (fine-tuned, A) | 0.713 | наша |
| CatBoost(emb, A) | 0.692 | наша |
| seara-rubert-base | 0.690 | внешняя |
| … | … | … |

Полная таблица из 20 моделей — в ноутбуке (фигуры `14`, `16`).

![Сравнение на OOB](reports/figures/14_oob_comparison.png)

**Выводы:**

- Дообучение энкодера выигрывает и in-domain, и на полностью независимом OOB,
  обходя 12 готовых решений.
- Вариант **B** (без меток из рейтинга) обучает устойчивее (выше B→B и OOB) —
  метки, выведенные из звёзд, вносят шум.
- `CatBoost(text+feats)` — сильный дешёвый baseline (in-domain B→B 0.86), но на
  OOB заметно проседает (доменный сдвиг).

## Воспроизведение

Окружение: Python 3.10+, `pip install -r requirements.txt` (выберите колесо `torch`
под свою CUDA; `cuML` опционален — даёт GPU-ускорение, иначе откат на scikit-learn).
Разработка велась на 1× GPU **Tesla V100 32 ГБ**.

- **Быстрый путь.** `git lfs pull` подтягивает готовые `master.parquet`, модели и
  предсказания. Матрица эмбеддингов корпуса (`embeddings_full_f16.npy`, ~2.8 ГБ) в
  git не хранится — создайте её один раз через `src/embed.py`, после чего можно
  открыть [`sentiment_analysis.ipynb`](sentiment_analysis.ipynb).
- **С нуля.** [`bash run_all.sh`](run_all.sh) — весь пайплайн по стадиям:
  данные → гармонизация → `master.parquet` → эмбеддинги → файн-тюнинг → OOB → ноутбук.
  Часть сырья (Kaggle, searayeah) скачивается вручную — ссылки в `datasets_sources.md`.

| Скрипт | Назначение |
|---|---|
| `src/download_hf.py` | скачать публичные HF-источники |
| `src/harmonize.py` | привести источники к единой схеме → `concat_raw.parquet` |
| `src/build_master.py` | дедуп, признаки, сплиты A/B → `master.parquet` |
| `src/embed.py` | эмбеддинги корпуса (USER-bge-m3, fp16) |
| `src/finetune.py` | поиск энкодера + финальное дообучение (A/B) |
| `src/prep_oob.py`, `src/embed_oob.py` | сборка и эмбеддинги OOB-набора |
| `src/oob_classical.py` | классические модели: обучение + предсказания на OOB |
| `src/oob_finetune_predict.py` | дообученный энкодер на OOB |
| `src/oob_external.py` | 12 внешних моделей на OOB |
| `sentiment_analysis.ipynb` | EDA, кластеризация, сравнение, анализ OOB → фигуры |

## Структура проекта

```
modelling/
├── README.md
├── run_all.sh                      # полный пайплайн (стадии 0–6)
├── requirements.txt
├── datasets_sources.md             # ссылки на все источники данных
├── sentiment_analysis.ipynb        # EDA, кластеризация, сравнение моделей, OOB
├── src/
│   ├── download_hf.py              # 0. скачивание публичных HF-источников
│   ├── inspect_searayeah.py        #    (опц.) осмотр сырых схем searayeah
│   ├── harmonize.py                # 1. единая схема → concat_raw.parquet
│   ├── build_master.py             # 2. дедуп, признаки, сплиты A/B → master.parquet
│   ├── text_features.py            #    инженерия текстовых признаков (модуль)
│   ├── embed.py                    # 3. эмбеддинги корпуса (USER-bge-m3, fp16)
│   ├── finetune.py                 # 4. поиск энкодера + дообучение (A/B)
│   ├── catboost_models.py          #    GPU-CatBoost головы (модуль)
│   ├── prep_oob.py                 # 5. сборка OOB-набора из ручной разметки
│   ├── embed_oob.py                # 5. эмбеддинги OOB
│   ├── oob_classical.py            # 6. классические модели → OOB-предсказания
│   ├── oob_finetune_predict.py     # 6. дообученный энкодер → OOB
│   └── oob_external.py             # 6. 12 внешних моделей → OOB
├── data/                           # Git LFS (interim/, processed/); raw/ — локально
│   ├── combined_platform_proportional_sample.xlsx   # ручная разметка (OOB)
│   ├── raw/{hf,searayeah}/         # сырьё (в git не хранится)
│   ├── interim/concat_raw.parquet
│   └── processed/
│       ├── master.parquet          # ~1.37 млн строк, 24 признака
│       ├── embeddings_full_f16.npy # ~2.8 ГБ, в git не хранится (создаёт embed.py)
│       ├── oob_test.parquet · oob_emb_f16.npy
│       └── preds/                  # предсказания + *_search.json, finetune_meta.json
├── models/                         # Git LFS
│   ├── finetune_A/ · finetune_B/   # дообученный ru-en-RoSBERTa (A и B)
│   ├── catboost_{text,emb}_{A,B}.cbm
│   └── emb_logreg_A.joblib · tfidf_svc_A.joblib
└── reports/figures/                # 01–17 + матрицы ошибок (cm_*.png)
```
