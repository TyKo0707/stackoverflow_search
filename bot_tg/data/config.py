from environs import Env
env = Env()
env.read_env()
BOT_TOKEN = env.str("BOT_TOKEN")
HEROKU_APP_NAME = env.str("HEROKU_APP_NAME")
ADMINS = env.list("ADMINS")
MAX_LIMIT = env.int("MAX_LIMIT")
