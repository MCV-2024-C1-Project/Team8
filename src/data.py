import pickle

from src.paths import GT_QSD1_W1_PATH
from src.paths import GT_QSD1_W2_PATH

with GT_QSD1_W1_PATH.open('rb') as f:
    GT_QSD1_W1_LIST = pickle.load(f)


with GT_QSD1_W2_PATH.open('rb') as f:
    GT_QSD1_W2_LIST = pickle.load(f)
