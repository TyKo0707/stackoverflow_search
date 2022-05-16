from google.cloud import bigquery
import os
from environs import Env
import gdown
from logger import get_logger

logger = get_logger()

env = Env()
env.read_env()
URL = env.str("URL")
OUT_FILE = env.str("OUT_FILE")
DATE_SIZE = env.str("DATE_SIZE")

gdown.download(URL, OUT_FILE, quiet=False)
out_file = os.path.abspath(OUT_FILE)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = out_file
client = bigquery.Client()


def build_query(min_id, max_id):
    query = f"""
        SELECT
          q.id, q.title, q.body, q.tags, a.body as answers, a.score
        FROM
          `bigquery-public-data.stackoverflow.posts_questions` AS q
        INNER JOIN
          `bigquery-public-data.stackoverflow.posts_answers` AS a
        ON
          q.id = a.parent_id
        WHERE
          q.id BETWEEN {int(min_id)} AND {int(max_id)}
          AND q.view_count > 250
          AND q.accepted_answer_id IS NOT NULL
        """
    return query


for i in range(6):
    query = build_query(35e6 + 5e6 * i, 35e6 + 5e6 * (i + 1))
    try:
        dataframe = (
            client.query(query).result().to_dataframe()
        )
        dataframe.to_parquet(f"out{i + 1}.gzip", compression='gzip', index=False)
        logger.info(f"Data (5m items (from {35 + 5 * i}m to {35 + 5 * (i + 1)}m)) "
                    f"was successfully downloaded and converted to CSV-file")
    except Exception as ex:
        logger.exception("An error occurred while pulling data from the database")
