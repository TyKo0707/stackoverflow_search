from google.cloud import bigquery
import os
from environs import Env
import gdown
from logger import logger

env = Env()
env.read_env()
URL = env.str("URL")
OUT_FILE = env.str("OUT_FILE")
DATE_SIZE = env.str("DATE_SIZE")

gdown.download(URL, OUT_FILE, quiet=False)
out_file = os.path.abspath(OUT_FILE)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = out_file
client = bigquery.Client()

query = """
    SELECT q.id, q.title, q.body, q.tags, a.body as answers, a.score
    FROM `bigquery-public-data.stackoverflow.posts_questions`
    AS q INNER JOIN `bigquery-public-data.stackoverflow.posts_answers`
    AS a ON q.id = a.parent_id LIMIT """ + DATE_SIZE

dataframe = (
    client.query(query)
        .result()
        .to_dataframe()
)
dataframe.to_csv("out.csv", index=False)
logger.info(f"Data ({DATE_SIZE} items) was successfully downloaded and converted to CSV-file")
