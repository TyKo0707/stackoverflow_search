from aiogram import executor
from loader import dp, bot
import handlers
from utils.set_bot_commands import set_default_commands
from utils.misc.logging import get_logger
from data.config import ADMINS
import asyncio

logger = get_logger()


async def on_startup(dispatcher):
    try:
        await set_default_commands(dispatcher)
    except Exception:
        logger.exception('An unexpected error occurred')


if __name__ == '__main__':
    executor.start_polling(dp, on_startup=on_startup)
