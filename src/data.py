import pickle

from src.paths import GT_QSD1_W1_PATH, GT_QSD1_W2_PATH, GT_QSD2_W2_PATH, GT_QSD1_W3_PATH, GT_QSD2_W3_PATH, \
    AUGMENTATIONS_QSD1_W4_PATH, FRAMES_QSD1_W4_PATH, GT_QSD1_W4_PATH

with GT_QSD1_W1_PATH.open('rb') as f:
    GT_QSD1_W1_LIST = pickle.load(f)

with GT_QSD1_W2_PATH.open('rb') as f:
    GT_QSD1_W2_LIST = pickle.load(f)
with GT_QSD2_W2_PATH.open('rb') as f:
    GT_QSD2_W2_LIST = pickle.load(f)

with GT_QSD1_W3_PATH.open('rb') as f:
    GT_QSD1_W3_LIST = pickle.load(f)
with GT_QSD2_W3_PATH.open('rb') as f:
    GT_QSD2_W3_LIST = pickle.load(f)

with AUGMENTATIONS_QSD1_W4_PATH.open('rb') as f:
    AUGMENTATIONS_QSD1_W4_LIST = pickle.load(f)
with FRAMES_QSD1_W4_PATH.open('rb') as f:
    FRAMES_QSD1_W4_LIST = pickle.load(f)
with GT_QSD1_W4_PATH.open('rb') as f:
    GT_QSD1_W4_LIST = pickle.load(f)
