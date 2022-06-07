from aiogram.types import Message, ChatType
from .handle_replies import reply_to_message
from bot_tg.loader import dp
from aiogram.dispatcher.filters import Text
import aiogram.utils.markdown as fmt


@dp.message_handler(Text(equals="Help"), chat_type=ChatType.PRIVATE)
@dp.message_handler(commands='help')
async def bot_help(message: Message):
    text = fmt.text(
        fmt.text(f"{fmt.hbold('Enter')} the request, and I {fmt.hbold('will find')} corresponding articles.\n"),
        fmt.text(f"{fmt.hbold('My commands:')}"),
        fmt.text(f"/start - Start the bot"),
        fmt.text(f"/help - Help"),
        fmt.text(f"/search - Search🔎"),
        sep='\n'
    )
    await reply_to_message(message, text)
