from aiogram import executor
from bot_tg.loader import dp, bot
from bot_tg.utils.set_bot_commands import set_default_commands
from bot_tg.utils.misc.logging import get_logger
from bot_tg.data.config import WEBHOOK_URL, WEBHOOK_PATH, WEBAPP_HOST, WEBAPP_PORT

logger = get_logger()


async def on_startup(dispatcher):
    try:
        await set_default_commands(dispatcher)
        await bot.set_webhook(WEBHOOK_URL, drop_pending_updates=True)
    except Exception:
        logger.exception('An unexpected error occurred')


async def on_shutdown(dispatcher):
    await bot.delete_webhook()


if __name__ == '__main__':
    executor.start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        skip_updates=True,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT
    )
