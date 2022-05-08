from aiogram.types import Message, ChatType
from bot_tg.states.state_storage import States
from bot_tg.loader import dp
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
import aiogram.utils.markdown as fmt


@dp.message_handler(Text(equals="Help"), state=[States.start, None], chat_type=ChatType.PRIVATE)
@dp.message_handler(commands='help', state=[States.start, None])
async def bot_help(message: Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state != States.start.state:
        await States.start.set()
    text = fmt.text(
        fmt.text(f"{fmt.hbold('Enter')} the request, and I {fmt.hbold('will find')} corresponding articles.\n"),
        fmt.text(f"{fmt.hbold('My commands:')}"),
        fmt.text(f"/start - Start the bot"),
        fmt.text(f"/help - Help"),
        fmt.text(f"/search - Search🔎"),
        sep='\n'
    )
    await message.reply(text)
