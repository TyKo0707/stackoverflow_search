from bot_tg.loader import dp
from search_engine.prediction_model.search_pipeline import search_results
from bot_tg.states.state_storage import States
from aiogram.types import Message, ChatType
from bot_tg.data.config import MAX_LIMIT
from aiogram.utils.markdown import hlink
import aiogram.utils.markdown as fmt
from aiogram.dispatcher.filters import Text
import random


@dp.message_handler(Text(equals='Search🔎'), state=[States.start, None], chat_type=ChatType.PRIVATE)
@dp.message_handler(commands="search", state=[States.start, None])
async def enter_search_mode(message: Message):
    await States.input_text.set()
    await message.reply(fmt.text(f"Enter the {fmt.hbold('request')} to be searched for:"))


@dp.message_handler(state=States.input_text)
async def input_request(message: Message):
    global TEXT
    TEXT = message.text
    await States.input_limit.set()
    await message.reply(fmt.text(f"Saving text..."
                                 f"\nEnter the {fmt.hbold('limit')} of articles to be shown (\u2264 {MAX_LIMIT}):"))


@dp.message_handler(state=States.input_limit)
async def input_limit(message: Message):
    text = message.text
    global TEXT
    if text.isdecimal() and 0 < int(text) <= MAX_LIMIT:
        articles = search_results(TEXT, int(text))

        result = "Articles:\n—————————"
        for i in range(5):
            result += fmt.text(
                fmt.text(f'{fmt.hbold("Title:")} {hlink(articles[i]["title"].capitalize(), articles[i]["url"])}'),
                fmt.text(f'{fmt.hbold("Similarity score:")} {articles[i]["similarity_score"]}'),
                fmt.text(f'{fmt.hbold("Tags:")} {articles[i]["tags"]}'),
                fmt.text(f'{fmt.hbold("Body:")} {articles[i]["body"][:75]}...\n—————————\n'),
                sep='\n'
            )
        await message.reply(result, disable_web_page_preview=True)
        await States.start.set()
    else:
        await message.reply(fmt.text(f"{fmt.hbold('Incorrect input')}. "
                                     f"Only non-negative integers are allowed which are \u2264 {MAX_LIMIT}."
                                     f"\nTry again:"))
