А. И. Клепиков
Омский государственный университет путей сообщения (ОмГУПС), г. Омск, Российская Федерация

СИСТЕМА АВТОМАТИЧЕСКОЙ МОДЕРАЦИИ СООБЩЕНИЙ ДЛЯ СОЦИАЛЬНОЙ СЕТИ ВКОНТАКТЕ НА ОСНОВЕ МАШИННОГО ОБУЧЕНИЯ

Целью машинного обучения является – научить машину (точнее будет сказать – программу) решать задачу, предъявив ей несколько примеров с правильным и неправильным решением. Алгоритмы машинного обучения позволяют автоматизировать рутинные процессы. В данной статье будет рассмотрен процесс создания полезного чат-бота для ВКонтакте, автоматически детектирующего «токсичные» сообщения, на основе алгоритма логистической регрессии с помощью Python.
Ключевые слова: машинное обучение, Python, классификация, логистическая регрессия, данные.

Alexander I. Klepikov
Omsk State Transport University (OSTU), Omsk, the Russian Federation

AUTOMATIC MESSAGE MODERATION SYSTEM FOR THE VKONTAKTE SOCIAL NETWORK BASED ON MACHINE LEARNING

The goal of machine learning is to teach a machine (or more accurately, a program) to solve a task by presenting it with several examples of correct and incorrect solutions. Machine learning algorithms make it possible to automate routine processes. This article will review the process of creating a useful chatbot for VKontakte that automatically detects «toxic» messages, using a logistic regression algorithm in Python.
Keywords: machine learning, Python, classification, logistic regression, data.

### Перед началом необходимо создать файл ".env", содержащий переменные окружения, а именно:
```
BOT_TOKEN = YOUR_VKBOT_TOKEN_HERE
GROUP_ID = YOUR_TARGET_VKCHAT_ID_HERE
```
**Где:**
*  **BOT_TOKEN** - ключ доступа от API VKontakte для вашего сообщества
*  **GROUP_ID** - числовой идентификатор чата, который будет модерировать бот
### Before starting, you need to create a ".env" file containing the following environment variables:
```
BOT_TOKEN = YOUR_VKBOT_TOKEN_HERE
GROUP_ID = YOUR_TARGET_VKCHAT_ID_HERE
```
**Where:**
*   **BOT_TOKEN** is the access token from the VKontakte API for your community.
*   **GROUP_ID** is the numeric identifier of the chat the bot will moderate.
