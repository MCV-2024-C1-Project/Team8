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
QSD1_W1_PATH = DATA_PATH / "qsd1_w1"
GT_QSD1_W1_PATH = QSD1_W1_PATH / "gt_corresps.pkl"
assert QSD1_W1_PATH.exists()
assert GT_QSD1_W1_PATH.exists()
