from pathlib import Path

# CORE
PROJECT_ROOT_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_ROOT_PATH / "data"
assert PROJECT_ROOT_PATH.exists()
assert DATA_PATH.exists()

# BBDD
BBDD_PATH = DATA_PATH / "BBDD"
assert BBDD_PATH.exists()

# W1
WEEK_1_PATH = PROJECT_ROOT_PATH / "WEEK_1"
QSD1_W1_PATH = DATA_PATH / "qsd1_w1"
QST1_W1_PATH = DATA_PATH / "qst1_w1"
GT_QSD1_W1_PATH = QSD1_W1_PATH / "gt_corresps.pkl"
WEEK_1_RESULTS_PATH = WEEK_1_PATH / "results"

assert WEEK_1_PATH.exists()
assert QSD1_W1_PATH.exists()
assert QST1_W1_PATH.exists()
assert GT_QSD1_W1_PATH.exists()
assert WEEK_1_RESULTS_PATH.exists()

# W2
WEEK_2_PATH = PROJECT_ROOT_PATH / "WEEK_2"
WEEK_2_RESULTS_PATH = WEEK_2_PATH / "results"
QSD1_W2_PATH = DATA_PATH / "qsd1_w2"
QSD2_W2_PATH = DATA_PATH / "qsd2_w2"
QST1_W2_PATH = DATA_PATH / "qst1_w2"
QST2_W2_PATH = DATA_PATH / "qst2_w2"

GT_QSD1_W2_PATH = QSD1_W2_PATH / "gt_corresps.pkl"
GT_QSD2_W2_PATH = QSD2_W2_PATH / "gt_corresps.pkl"
GT_FRAMES_QSD2_W2_PATH = QSD2_W2_PATH / "gt_corresps.pkl"
WEEK_2_RESULTS_PATH = WEEK_2_PATH / "results"

assert WEEK_2_PATH.exists()
assert WEEK_2_RESULTS_PATH.exists()
assert QSD1_W2_PATH.exists()
assert QSD2_W2_PATH.exists()
assert GT_QSD1_W2_PATH.exists()
assert GT_QSD2_W2_PATH.exists()
assert GT_FRAMES_QSD2_W2_PATH.exists()
# assert WEEK_2_RESULTS_PATH.exists()
