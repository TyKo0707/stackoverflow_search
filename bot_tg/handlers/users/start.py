from aiogram.dispatcher import FSMContext
from states.state_storage import States
from loader import dp
from aiogram.types import Message
from utils.misc.logging import get_logger
from keyboards import main_keyboard
import aiogram.utils.markdown as fmt

logger = get_logger()


@dp.message_handler(state=[States.start, None], commands='start')
async def start(message: Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state != States.start.state:
        await States.start.set()
    await message.reply(fmt.text(f"Hi, I am stackoverflow search bot.\n"
                                 f"{fmt.hbold('Enter')} the request, and I {fmt.hbold('will find')} "
                                 f"corresponding articles."
                                 f"\nClick the {fmt.hbold('search button')} to start searching for the articles"),
                        reply_markup=main_keyboard)
