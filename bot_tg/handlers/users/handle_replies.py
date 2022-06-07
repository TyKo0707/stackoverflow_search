from aiogram.utils.exceptions import BadRequest
from aiogram.types import Message


async def reply_to_message(message: Message, text, reply_markup=None):
    try:
        if reply_markup:
            await message.reply(text, reply_markup=reply_markup, disable_web_page_preview=True)
        else:
            await message.reply(text, disable_web_page_preview=True)
    except BadRequest:
        if reply_markup:
            await message.answer(text, reply_markup=reply_markup, disable_web_page_preview=True)
        else:
            await message.answer(text, disable_web_page_preview=True)

