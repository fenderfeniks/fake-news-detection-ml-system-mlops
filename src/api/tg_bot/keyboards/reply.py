# src/api/tg_bot/keyboards/reply.py
"""
Reply-клавиатуры (нижние кнопки).
"""

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup


def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Главная клавиатура бота."""
    keyboard = [
        [KeyboardButton(text="/start")],
    ]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        input_field_placeholder="Отправьте новость для проверки...",
    )
