from configs import CONFIG
import pandas as pd
import re 
import os 
import mlflow
from tqdm import tqdm 
import pickle
import sys
import string 
from sklearn.model_selection import StratifiedKFold

class prepDataset:
    def __init__(self) -> None:
        self.df = pd.read_csv(CONFIG.TRAIN_FILE)
        self.index2word = {idx: word for idx, word in enumerate(set(self.df["sentiment"].values))}
        self.word2index = {word: idx for idx, word in self.index2word.items()}
        self.df["label"] = self.df["sentiment"].apply(lambda x: self.word2index[x])
        arg1, arg2 = sys.argv[1], sys.argv[2]
        self.dumpData(self.index2word,arg1.split("src/")[-1])
        self.dumpData(self.word2index, arg2.split("src/")[-1])
        self.threshold = 256
        self.df["preprocessedREVIEW"]  = self.df["review"].apply(self.preprocessing)
        # self.df["preprocessingSplit"] = self.df["preprocessedREVIEW"].apply(self.get_split)
        # self.df = self.createDataset()
        
        self.stratkFold = StratifiedKFold(n_splits=4)
        self.splitData()

    def createDataset(self):
        texts = []
        labels = []
        for idx, row in self.df.iterrows():
            for text in row["preprocessingSplit"]:
                texts.append(text)
                labels.append(row["label"])
        assert len(texts) == len(labels)

        df = pd.DataFrame({"preprocessedREVIEW": texts, "label": labels})
        return df 

    def get_split(self,text1):
        l_total = []
        l_parcial = []
        if len(text1.split())//150 >0:
            n = len(text1.split())//150
        else: 
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = text1.split()[:self.threshold]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = text1.split()[w*150:w*150 + self.threshold]
                l_total.append(" ".join(l_parcial))
        return l_total

    def splitData(self):
        X = self.df["preprocessedREVIEW"]
        Y = self.df["label"]
        for idx, (train_idx, test_idx) in enumerate(self.stratkFold.split(X,Y)):
            # let's create folder
            print(f"--------------------------------------- FOLD_{idx} ---------------------------------------")
            path = os.path.join(CONFIG.MULTIFOLD, f"fold_{idx}")
            if not os.path.isdir(path):
                os.mkdir(path)
            # let's save the train and test data inside the corresponding fold 
            train = self.df.iloc[train_idx]
            test  = self.df.iloc[test_idx]
            train.to_csv(os.path.join(path, "train.csv"), index=False)
            test.to_csv(os.path.join(path, "test.csv"), index=False)

    @staticmethod
    def dumpData(dictionary, filename):
        if not os.path.isdir(CONFIG.WORDINDEX):
            os.mkdir(CONFIG.WORDINDEX)
        path = os.path.join(CONFIG.WORDINDEX, filename)
        mlflow.log_artifacts(CONFIG.WORDINDEX)
        pickle.dump(dictionary, open(path, "wb"))
    
    @staticmethod
    def loadData(filename):
        path = os.path.join(CONFIG.WORDINDEX, filename)
        return pickle.load(open(path, "rb"))


    def preprocessing(self,sentence):
        sentence = sentence.translate(str.maketrans("","", string.punctuation)).lower()
        #let's remove if their any links 
        sentence = re.sub(r"https?://\s+", "", sentence)
        sentence = re.sub(r"\b\d+\b",  "", sentence)
        sentence = re.sub(r" +"," ",sentence)
        return sentence


prepDataset()