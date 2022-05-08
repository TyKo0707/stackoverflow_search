from aiogram.types import Message
from states.state_storage import States
from loader import dp
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Text
import aiogram.utils.markdown as fmt


@dp.message_handler(Text(equals="Help"), state=[States.start, None])
async def bot_help(message: Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state != States.start.state:
        await States.start.set()
    await message.reply(fmt.text(f"{fmt.hbold('Enter')} the request, and I {fmt.hbold('will find')} "
                        f"corresponding articles."
                        f"\nClick the {fmt.hbold('search button')} to start searching for the articles"))
