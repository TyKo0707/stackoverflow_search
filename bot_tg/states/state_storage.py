from aiogram.dispatcher.filters.state import StatesGroup, State


class States(StatesGroup):
    start = State()
    input_text = State()
    input_limit = State()

