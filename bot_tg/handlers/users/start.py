from .handle_replies import reply_to_message
from bot_tg.loader import dp
from aiogram.types import Message, ChatType
from logger import get_logger
from bot_tg.keyboards import main_keyboard
import aiogram.utils.markdown as fmt

logger = get_logger()


@dp.message_handler(commands='start')
async def start(message: Message):
    text = fmt.text(f"Hi, I am stackoverflow search bot.\n"
                    f"{fmt.hbold('Enter')} the request, and I {fmt.hbold('will find')} "
                    f"corresponding articles."
                    f"\nUse /search command or click the button (not for groups) to start searching for the articles")
    if message.chat.type in [ChatType.SUPERGROUP, ChatType.GROUP]:
        await reply_to_message(message, text)
    else:
        await reply_to_message(message, text, main_keyboard)

