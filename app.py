from aiogram import executor
from bot_tg.loader import dp
import bot_tg.handlers
from bot_tg.utils.set_bot_commands import set_default_commands
from logger import get_logger

logger = get_logger()


async def on_startup(dispatcher):
    try:
        await set_default_commands(dispatcher)
    except Exception:
        logger.exception('An unexpected error occurred')


if __name__ == '__main__':
    executor.start_polling(dp, on_startup=on_startup, skip_updates=True)
