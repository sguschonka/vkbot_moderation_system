import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score


def save_model(model, vectorizer=None):
    """Сохраняет модель для последующего использования"""
    # Сохраняем модель
    joblib.dump(model, "interfaces/data/toxicity_model.pkl")

    print("Модель сохранена!")


def load_model():
    """Загружает сохраненную модель"""
    model = joblib.load("interfaces/data/toxicity_model.pkl")

    print("Модель загружена!")
    return model


def plot_simple_test_metrics(model, X_test, y_test):
    """Простая визуализация основных метрик на тестовой выборке"""

    y_pred = model.predict(X_test)

    plt.figure(figsize=(8, 6))

    ax = plt.gca()

    # Матрица ошибок
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax,
        display_labels=["Нормальные", "Токсичные"],
        cmap="Purples",
        colorbar=True,
    )
    ax.set_title(
        "Матрица ошибок на тестовой выборке", fontsize=14, fontweight="bold"
    )

    accuracy = accuracy_score(y_test, y_pred)
    plt.suptitle(
        f"Точность на тестовой выборке: {accuracy:.4f}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()

    return accuracy
