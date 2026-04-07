from config import BOT_TOKEN, GROUP_ID
from interfaces.algorithm import process_model
from interfaces.bot_interface import VKBot


def main():
    # Обучаем модель (если нужно)
    process_model()

    # Создаем и запускаем бота
    bot = VKBot(token=BOT_TOKEN, group_id=GROUP_ID)
    bot.run()


if __name__ == "__main__":
    main()
