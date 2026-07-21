# src/api/tg_bot/handlers/chat.py
"""
Главный обработчик сообщений (Бизнес-логика бота).
Умеет ходить по HTTP (для bot_local) и напрямую в память (для bot_webhook).
"""

import asyncio
import logging
import os

import aiohttp
from aiogram import F, Router, types
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext

from src.api.metrics import CLASSIFICATION_INFERENCE_TIME, CLASSIFICATION_REQUESTS_TOTAL
from src.api.tg_bot.keyboards.reply import get_main_keyboard
from src.api.tg_bot.states import ChatProcess


logger = logging.getLogger(__name__)
router = Router()


def format_prediction(label_id: int, confidence: float) -> str:
    """Вспомогательная функция для красивого форматирования ответа."""
    # TODO: В будущем можно вынести маппинг классов в конфиг
    label_map = {0: "🔴 Фейковая новость", 1: "🟢 Подтвержденная информация"}
    label_text = label_map.get(label_id, "⚪ Неизвестный класс")
    return f"**Анализ завершен**\n\nВердикт: {label_text}\nУверенность модели: {confidence:.2%}"


@router.message(CommandStart())
async def cmd_start(message: types.Message, state: FSMContext, cfg):
    """Обработка команды /start."""
    logger.info(f"Получено сообщение: {message.text}")
    await state.set_state(ChatProcess.chatting)
    await message.answer(
        text=cfg.api.telegram.messages.welcome,
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown",
    )


@router.message(ChatProcess.chatting, F.text)
async def process_chat_message(
    message: types.Message,
    cfg,
    classifier=None,
    api_url=None,
):
    logger.info(f"Получено сообщение: {message.text}")
    processing_msg = await message.answer("Анализирую текст...")

    try:
        # --- СЦЕНАРИЙ А: ПРОДАКШЕН (Webhooks) ---
        if classifier:
            CLASSIFICATION_REQUESTS_TOTAL.labels(source="tg").inc()
            with CLASSIFICATION_INFERENCE_TIME.labels(source="tg").time():
                results = await asyncio.to_thread(classifier, message.text)

            res = results[0]
            answer = format_prediction(res["label_id"], res["confidence"])

        # --- СЦЕНАРИЙ Б: ЛОКАЛЬНАЯ РАЗРАБОТКА (Polling) ---
        elif api_url:
            payload = {"text": message.text}
            headers = {"X-API-Key": os.getenv("API_KEY", "")}

            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        answer = format_prediction(data.get("label_id"), data.get("confidence"))
                    else:
                        answer = f"Ошибка API: HTTP {resp.status}"
        else:
            raise ValueError("Не передан ни classifier, ни api_url!")

        await processing_msg.edit_text(answer, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Ошибка в хэндлере ТГ: {str(e)}")
        await processing_msg.edit_text(cfg.api.telegram.messages.error)
