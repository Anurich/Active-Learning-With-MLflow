import os 

class CONFIG:
    DIRECTORY_PATH = os.getcwd()
    TRAIN_FILE = os.path.join(DIRECTORY_PATH, "datasets/IMDB Dataset.csv")
    TEST_FILE  = ""
    DEV_FILE   = ""
    MULTIFOLD = os.path.join(DIRECTORY_PATH, "datasets")
    WORDINDEX = os.path.join(DIRECTORY_PATH, "datasets", "wordIndex")

