А. И. Клепиков
Омский государственный университет путей сообщения (ОмГУПС), г. Омск, Российская Федерация

Научный руководитель – Т. В. Васеева, старший преподаватель кафедры «Автоматика и системы управления», ОмГУПС.

СИСТЕМА АВТОМАТИЧЕСКОЙ МОДЕРАЦИИ СООБЩЕНИЙ ДЛЯ СОЦИАЛЬНОЙ СЕТИ ВКОНТАКТЕ НА ОСНОВЕ МАШИННОГО ОБУЧЕНИЯ

Статья посвящена созданию чат-бота для автоматического обнаружения токсичных сообщений в социальной сети ВКонтакте. Для классификации применяется алгоритм логистической регрессии, реализованный на языке программирования Python. Приводится описание подготовки данных, выбора признаков, обучения модели и интеграции бота с VK API. Точность классификации составляет 88%, что подтверждает эффективность предложенного подхода.
Ключевые слова: машинное обучение, Python, классификация, логистическая регрессия, данные.

Alexander I. Klepikov
Omsk State Transport University (OSTU), Omsk, the Russian Federation

AUTOMATIC MESSAGE MODERATION SYSTEM FOR THE VKONTAKTE SOCIAL NETWORK BASED ON MACHINE LEARNING

This article is devoted to the creation of a chatbot for automatically detecting toxic messages in the VKontakte social network. A logistic regression algorithm implemented in the Python programming language is used for classification. The article describes the data preparation process, feature selection, model training, and bot integration with the VK API. The classification accuracy is 88%, which confirms the effectiveness of the proposed approach.
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
