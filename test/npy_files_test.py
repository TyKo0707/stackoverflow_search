import numpy as np
from environs import Env
from timer import timer
import pyarrow.parquet as pq

env = Env()
env.read_env()
TRAIN_TEST_PATH = env.str("TRAIN_TEST_PATH")

with timer('open dataset with 1 method(.npy)', 3):
    title_embeddings1 = np.load(TRAIN_TEST_PATH + 'embedding_matrix.npy')

with timer('open dataset with 2 method(.parquet)', 3):
    title_embeddings2 = pq.read_table(TRAIN_TEST_PATH + 'embedding_matrix.parquet')

# [open dataset with 1 method(.npy)] done in 22.697 s
# [open dataset with 2 method(.parquet)] done in 25.948 s
