import asyncio
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import ollama
import nest_asyncio
import sqlite3
from datetime import datetime
from typing import List, Dict
import nltk

nltk.download('punkt')

nest_asyncio.apply()

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = ''


# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('old_russian_texts.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT,
            modern_translation TEXT,
            summary TEXT,
            keywords TEXT,
            timestamp DATETIME,
            user_id INTEGER
        )
    ''')
    conn.commit()
    conn.close()


class TextProcessor:
    def __init__(self):
        self.ollama_model = 'llama3:8b'

    async def translate_to_modern(self, text: str) -> str:
        """Перевод старорусского текста на современный русский"""
        system_prompt = "Ты - русскоязычный эксперт по старорусским текстам. Отвечай только на русском языке."
        prompt = f"""
        Задача: переведи старорусский текст на современный русский язык.
        Важно: 
        1. Сохрани смысл и стиль оригинала
        2. Используй современные слова и обороты речи
        3. Ответ должен быть только на русском языке

        Текст для перевода: {text}
        """

        response = ollama.chat(model=self.ollama_model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']

    async def summarize(self, text: str) -> str:
        """Создание краткого содержания текста"""
        system_prompt = "Ты - русскоязычный эксперт по анализу текстов. Отвечай ТОЛЬКО на русском языке. Никакого английского в ответах."
        prompt = f"""
        Задача: создай краткое содержание текста на русском языке.
        Требования:
        1. Напиши 3-4 содержательных предложения
        2. Выдели главные мысли и ключевые события
        3. Используй только русский язык
        4. Сохрани главный смысл текста

        Текст для анализа: {text}
        """

        response = ollama.chat(model=self.ollama_model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content']

    async def extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов из текста"""
        system_prompt = "Ты - русскоязычный эксперт по анализу текстов. Отвечай только на русском языке."
        prompt = f"""
        Задача: выдели ключевые слова из текста.
        Требования:
        1. Выдели 5-7 важных слов или словосочетаний
        2. Используй только русские слова
        3. Перечисли слова через запятую
        4. Обрати особое внимание на исторические термины

        Текст для анализа: {text}
        """

        response = ollama.chat(model=self.ollama_model, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content'].split(', ')


class DatabaseManager:
    def __init__(self, db_name='old_russian_texts.db'):
        self.db_name = db_name

    def save_text(self, original: str, translation: str, summary: str, keywords: str, user_id: int):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('''
            INSERT INTO texts (original_text, modern_translation, summary, keywords, timestamp, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (original, translation, summary, keywords, datetime.now(), user_id))
        conn.commit()
        conn.close()

    def get_user_texts(self, user_id: int) -> List[Dict]:
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute('SELECT * FROM texts WHERE user_id = ?', (user_id,))
        rows = c.fetchall()
        conn.close()
        return rows


class OldRussianBot:
    def __init__(self, token: str):
        self.token = token
        self.text_processor = TextProcessor()
        self.db_manager = DatabaseManager()
        init_db()

    async def start(self, update: Update, context) -> None:
        welcome_text = """
                        Приветствую! Я бот для обработки старорусских текстов.
                        Я могу:
                        - Переводить тексты на современный русский язык
                        - Создавать краткое содержание
                        - Сохранять обработанные тексты в базу данных
                        
                        Просто отправьте мне текст на старорусском языке.
                        Используйте /help для получения списка команд.
                        """
        await update.message.reply_text(welcome_text)

    async def help(self, update: Update, context) -> None:
        help_text = """
                    Доступные команды:
                    /start - Начать работу с ботом
                    /help - Показать это сообщение
                    /history - Показать историю обработанных текстов
                    """
        await update.message.reply_text(help_text)

    async def handle_message(self, update: Update, context) -> None:
        user_id = update.effective_user.id
        text = update.message.text

        try:
            await update.message.reply_text("Обрабатываю ваш текст...")

            # Перевод текста
            translation = await self.text_processor.translate_to_modern(text)
            await update.message.reply_text(f"Современный перевод:\n\n{translation}")

            # Создание краткого содержания
            summary = await self.text_processor.summarize(translation)
            await update.message.reply_text(f"Краткое содержание:\n\n{summary}")

            # Извлечение ключевых слов
            keywords = await self.text_processor.extract_keywords(translation)
            keywords_str = ", ".join(keywords)
            await update.message.reply_text(f"Ключевые слова:\n\n{keywords_str}")

            # Сохранение в базу данных
            self.db_manager.save_text(text, translation, summary, keywords_str, user_id)
            await update.message.reply_text("✅ Текст успешно обработан и сохранен в базу данных")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await update.message.reply_text("Произошла ошибка при обработке текста. Попробуйте позже.")

    async def show_history(self, update: Update, context) -> None:
        user_id = update.effective_user.id
        texts = self.db_manager.get_user_texts(user_id)

        if not texts:
            await update.message.reply_text("У вас пока нет обработанных текстов.")
            return

        for text in texts[-3:]:  # Показываем последние 3 записи
            response = f"""
                        Дата обработки: {text[5]}
                        Оригинальный текст: {text[1][:100]}...
                        Перевод: {text[2][:100]}...
                        Краткое содержание: {text[3]}
                        Ключевые слова: {text[4]}
                        -------------------
                        """
            await update.message.reply_text(response)

    async def run(self):
        application = ApplicationBuilder().token(self.token).build()

        # Добавляем обработчики команд
        application.add_handler(CommandHandler('start', self.start))
        application.add_handler(CommandHandler('help', self.help))
        application.add_handler(CommandHandler('history', self.show_history))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        await application.run_polling()


if __name__ == '__main__':
    bot = OldRussianBot(TOKEN)
    asyncio.run(bot.run())
