from google.cloud import bigquery
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\38097\Downloads\stackoverflow-347020-453c5a423922.json"
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
print(dataframe.head())
dataframe.to_csv("out.csv", index=False)
