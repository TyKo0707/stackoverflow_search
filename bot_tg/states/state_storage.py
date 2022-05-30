from aiogram.dispatcher.filters.state import StatesGroup, State


class States(StatesGroup):
    input_text = State()
    input_limit = State()

