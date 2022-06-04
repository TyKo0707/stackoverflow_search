from aiogram.dispatcher import FSMContext
from bot_tg.loader import dp
from search_engine.prediction_model.search_pipeline import search_results
from bot_tg.states.state_storage import States
from aiogram.types import Message, ChatType
from bot_tg.data.config import MAX_LIMIT
from aiogram.utils.markdown import hlink
import aiogram.utils.markdown as fmt
from aiogram.dispatcher.filters import Text
import time
from aiogram.utils.exceptions import BadRequest


@dp.message_handler(Text(equals='Search🔎'), chat_type=ChatType.PRIVATE)
@dp.message_handler(commands="search")
async def enter_search_mode(message: Message):
    await States.input_text.set()
    await message.reply(fmt.text(f"Enter the {fmt.hbold('request')} to be searched for:"))


@dp.message_handler(state=States.input_text)
async def input_request(message: Message, state: FSMContext):
    await state.update_data(search_text=message.text)
    await States.input_limit.set()
    await message.reply(fmt.text(f"Saving text..."
                                 f"\nEnter the {fmt.hbold('limit')} of articles to be shown (\u2264 {MAX_LIMIT}):"))


@dp.message_handler(state=States.input_limit)
async def input_limit(message: Message, state: FSMContext):
    text = message.text
    if text.isdecimal() and 0 < int(text) <= MAX_LIMIT:
        await message.reply("Searching...")
        num = int(text)
        user_data = await state.get_data()
        await state.finish()
        search_text = user_data['search_text']
        t_0 = time.time()
        articles = search_results(search_text, num)
        if articles:
            text = "Articles:\n—————————\n"
            for i in range(num):
                text += fmt.text(
                    fmt.text(f'{fmt.hbold("Title:")} {hlink(articles[i]["title"].capitalize(), articles[i]["url"])}'),
                    fmt.text(f'{fmt.hbold("Similarity score:")} {articles[i]["similarity_score"]}'),
                    fmt.text(f'{fmt.hbold("Tags:")} {articles[i]["tags"]}'),
                    fmt.text(f'{fmt.hbold("Body:")} {articles[i]["body"][:75]}...\n—————————\n'),
                    sep='\n'
                )
            text += fmt.text(f"The search has been done for {fmt.hbold(round(time.time() - t_0, 1))} seconds\n\n")
        else:
            text = fmt.text(f'No corresponding articles were found for such request: "{fmt.hbold(search_text)}"')
    else:
        text = fmt.text(f"{fmt.hbold('Incorrect input')}. "
                        f"Only non-negative integers are allowed which are \u2264 {MAX_LIMIT}."
                        f"\nTry again:")
    try:
        await message.reply(text, disable_web_page_preview=True)
    except BadRequest:
        await message.answer(text, disable_web_page_preview=True)
