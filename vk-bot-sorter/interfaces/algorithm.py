import re

import emoji
import pandas as pd
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
)

from interfaces.utils.algorithm_functions import (
    plot_simple_test_metrics,
    save_model,
)


def load_insult_word():
    words_set = set()
    with open("interfaces/data/extend_insults.txt", encoding="UTF-8") as f:
        words = f.readlines()

        for word in words:
            word_lower = word.lower()
            words_set.add(word_lower.strip())

    return words_set


def load_toxic_emojis():
    words_set = set()
    with open("interfaces/data/toxic_emoji.txt", encoding="UTF-8") as f:
        words = f.readlines()

        for word in words:
            word_lower = word.lower()
            words_set.add(word_lower.strip())
    return words_set


def load_abuse_word():
    words_set = set()
    with open("interfaces/data/abuse_words.txt", encoding="UTF-8") as f:
        words = f.readlines()

        for word in words:
            word_lower = word.lower()
            words_set.add(word_lower.strip())
    return words_set


toxic_emojis = load_toxic_emojis()
insult_words = load_insult_word()
abuse_words = load_abuse_word()


def extract_insults(text):
    """Считает количество матерных слов в тексте"""
    if not isinstance(text, str) or not text.strip():
        return 0

    clean_text = re.sub(r"[^\w\s]", " ", text.lower())  # Убираем пунктуацию
    clean_text = re.sub(
        r"\s+", " ", clean_text.strip()
    )  # Убираем лишние пробелы
    words = clean_text.split()

    count = 0
    for word in words:
        if word in insult_words:
            count += 1

    return count


def extract_abuse_words(text):
    """Считает количество матерных слов в тексте"""
    if not isinstance(text, str) or not text.strip():
        return 0

    clean_text = re.sub(r"[^\w\s]", " ", text.lower())  # Убираем пунктуацию
    clean_text = re.sub(
        r"\s+", " ", clean_text.strip()
    )  # Убираем лишние пробелы
    words = clean_text.split()

    count = 0
    for word in words:
        if word in abuse_words:
            count += 1

    return count


def extract_toxic_emojis(text):
    """Считает количество матерных слов в тексте"""
    if not isinstance(text, str):
        return 0

    found_emojis = [e["emoji"] for e in emoji.emoji_list(text)]

    count = 0
    for emoji_char in found_emojis:
        if emoji_char in toxic_emojis:
            count += 1

    return count


def load_and_prepare_dataset(csv_path):
    """
    Загружает и подготавливает датасет из CSV файла
    """
    df = pd.read_csv(csv_path, nrows=28567)

    # Создаем единый датасет из обеих колонок
    all_data = []

    # Положительные сообщения (метка 0)
    for text in df["Положительное_сообщение"].dropna():
        clean_text = str(text).strip()
        if clean_text:
            all_data.append(
                {
                    "text": clean_text,
                    "label": 0,
                    "has_insult": extract_insults(clean_text),
                    "has_abuse": extract_abuse_words(clean_text),
                    "has_toxic_emoji": extract_toxic_emojis(clean_text),
                }
            )

    # Отрицательные сообщения (метка 1)
    for text in df["Отрицательное_сообщение"].dropna():
        clean_text = str(text).strip()
        if clean_text:
            all_data.append(
                {
                    "text": clean_text,
                    "label": 1,
                    "has_insult": extract_insults(clean_text),
                    "has_abuse": extract_abuse_words(clean_text),
                    "has_toxic_emoji": extract_toxic_emojis(clean_text),
                }
            )

    df_processed = pd.DataFrame(all_data)
    return df_processed


def test_different_models():
    """Тестируем разные модели на тех же данных"""
    stats = []

    df_processed = load_and_prepare_dataset(
        "interfaces/data/toxic_comments_with_emojis.csv"
    )
    X = df_processed[["has_insult", "has_abuse", "has_toxic_emoji"]]
    y = df_processed["label"]

    from sklearn.linear_model import LogisticRegression

    models = {
        "KNN": neighbors.KNeighborsClassifier(n_neighbors=12),
        "LogisticRegression": LogisticRegression(),
    }

    for name, model in models.items():
        cv_score = cross_val_score(
            model, X, y, cv=5, scoring="accuracy"
        ).mean()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2025, stratify=y
        )
        model.fit(X_train, y_train)
        single_score = model.score(X_test, y_test)

        stats.append(
            f"{name}: CV={cv_score:.4f}, Single={single_score:.4f}, Diff={single_score - cv_score:.4f}"
        )
    return stats


# Обновленная основная функция
def process_model():
    # Протестируем для KNN классификатора и логистической регрессии, используя перекрестную проверку
    models_stats = test_different_models()
    for stat in models_stats:
        print(f"{stat}")

    # Загрузка датасета
    df_processed = load_and_prepare_dataset(
        "interfaces/data/toxic_comments_with_emojis.csv"
    )

    # Создание классификатора
    clf = LogisticRegression()

    # Подготовка данных
    X = df_processed[["has_insult", "has_abuse", "has_toxic_emoji"]]
    y = df_processed["label"]

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2025
    )

    print(f"Размер тестовой выборки: {len(X_test)}")

    # Простая визуализация
    clf.fit(X_train, y_train)  # Переобучаем модель
    plot_simple_test_metrics(clf, X_test, y_test)

    # 5. Сохраняем модель
    save_model(clf)

