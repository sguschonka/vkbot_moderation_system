import logging
import random
import re

import emoji
import vk_api
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll

from interfaces.algorithm import (
    load_abuse_word,
    load_insult_word,
    load_toxic_emojis,
)
from interfaces.utils.algorithm_functions import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
insult_words_path = "data/extend_insults.txt"
abuse_words_path = "data/abuse_words.txt"
toxic_emojis_path = "data/toxic_emoji.txt"


class VKBase:
    def __init__(self, token, group_id):
        self.token = token
        self.group_id = int(group_id)
        self.vk_session = vk_api.VkApi(token=token)
        self.longpoll = VkBotLongPoll(self.vk_session, group_id)
        self.vk = self.vk_session.get_api()


class VKBot(VKBase):
    def __init__(self, token, group_id):
        super().__init__(token, group_id)
        self.group_id = int(group_id)
        self.model = None
        # Загружаем матерные слова один раз при инициализации
        self.insult_words = load_insult_word()
        self.abuse_words = load_abuse_word()
        self.toxic_emojis = load_toxic_emojis()

    def run(self):
        """Запуск бота"""
        logger.info("Бот запущен...")
        self.load_model()

        for event in self.longpoll.listen():
            if event.type == VkBotEventType.MESSAGE_NEW:
                self.handle_message(event)

    def handle_message(self, event):
        """Обработка входящих сообщений"""
        try:
            user_id = event.message.from_id
            conversation_message_id = event.message.conversation_message_id
            peer_id = event.message.peer_id
            text = event.message.text.lower()

            logger.info(f"Получено сообщение от {user_id}: {text}")

            # Простая логика ответов
            if text == "привет":
                self.send_message(peer_id, "Привет! Как дела?")
            elif text == "пока":
                self.delete_message(
                    peer_id, conversation_message_id=conversation_message_id
                )
                self.send_message(
                    peer_id, "Извини, нам пришлось удалить твое сообщение :("
                )
            elif "как дела" in text:
                self.send_message(peer_id, "У меня всё отлично! А у вас?")

            # Проверка токсичности
            result = self.predict_toxicity(text)
            logger.info(f"Результат проверки: {result}")
            if result == "is_toxic":
                self.delete_message(
                    peer_id, conversation_message_id=conversation_message_id
                )
                self.send_message(peer_id, self.get_warning_message())
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")

    def get_warning_message(self):
        return """
🎯 *Ой-ой-ой!* 🎯
Ваш словарный запас немного "обогатился" не теми словами! 💎
                
🎪 Переключаемся на:
🎭 Цензурный режим
🎨 Красивый русский язык
😊 Вежливый тон общения
                
Спасибо за понимание! 🌈
"""

    def send_message(self, peer_id, message, keyboard=None):
        """Отправка сообщения пользователю"""
        try:
            params = {
                "peer_id": peer_id,
                "message": message,
                "random_id": random.randint(1, 1000000),
            }

            if keyboard:
                params["keyboard"] = keyboard

            self.vk.messages.send(**params)
        except Exception as e:
            logger.warning(
                f"Не удалось отправить сообщение {message}.\nОшибка: {e}"
            )

    def delete_message(
        self, peer_id, delete_for_all=True, conversation_message_id=None
    ):
        """Удаляем сообщение пользователя"""
        try:
            params = {
                "cmids": conversation_message_id,
                "delete_for_all": delete_for_all,
                "peer_id": peer_id,
                "group_id": self.group_id,
            }

            self.vk.messages.delete(**params)
            logger.info(f"Сообщение {conversation_message_id} удалено")
        except Exception as e:
            logger.warning(
                f"Не удалось удалить сообщение {conversation_message_id}. Ошибка: {e}"
            )

    def load_model(self):
        """Загружаем модель"""
        try:
            self.model = load_model()
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            self.model = None

    def predict_toxicity(self, text):
        """Предсказание токсичности текста"""
        if not self.model:
            return "is_normal"

        try:
            clean_text = re.sub(r"[^\w\s]", " ", text.lower())
            words = clean_text.split()

            # Используем уже загруженные слова, а не загружаем каждый раз
            insult_count = sum(
                1 for word in words if word in self.insult_words
            )

            abuse_count = sum(1 for word in words if word in self.abuse_words)

            found_emojis = [char for char in text if char in emoji.EMOJI_DATA]

            emoji_count = sum(
                1 for char in found_emojis if char in self.toxic_emojis
            )

            features = [[insult_count, emoji_count, abuse_count]]

            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            result = "is_toxic" if prediction else "is_normal"

            # Уверенность - вероятность предсказанного класса
            confidence = probabilities[prediction]

            # Детальное логирование
            logger.info(
                f"Признаки: нецензурная лексика={insult_count}, эмодзи={emoji_count}, оскорбления={abuse_count}"
            )
            logger.info(f"Результат: {result}, Уверенность: {confidence:.3f}")
            logger.info(
                f"Все вероятности: нормальное={probabilities[0]:.3f}, токсичное={probabilities[1]:.3f}"
            )
            return result
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return "is_normal"
