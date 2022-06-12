from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

main_keyboard = ReplyKeyboardMarkup([
    [
        KeyboardButton("Help"),
        KeyboardButton("Search🔎")
    ]
], resize_keyboard=True)
