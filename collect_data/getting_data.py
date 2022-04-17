from google.cloud import bigquery
import os
from environs import Env
import gdown

env = Env()
env.read_env()
URL = env.str("URL")
OUT_FILE = env.str("OUT_FILE")

gdown.download(URL, OUT_FILE, quiet=False)
out_file = os.path.abspath(OUT_FILE)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = out_file
client = bigquery.Client()

query = """
    SELECT q.id, q.title, q.body, q.tags, a.body as answers, a.score
    FROM `bigquery-public-data.stackoverflow.posts_questions`
    AS q INNER JOIN `bigquery-public-data.stackoverflow.posts_answers`
    AS a ON q.id = a.parent_id LIMIT 1000000
    """

dataframe = (
    client.query(query)
        .result()
        .to_dataframe()
)
dataframe.to_csv("out.csv", index=False)
