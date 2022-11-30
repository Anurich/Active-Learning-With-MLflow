import random
import torch
import pandas as pd
class ACLearning:
    def __init__(self, dataset) -> None:
        self.unlabelledDataset = dataset
        self.labelledDataset = None

    def randomSample(self, k = 200):
        labelledIndexes= random.sample(range(len(self.unlabelledDataset)), k=200)
        unlabelledIndexes = range(len(self.unlabelledDataset))
        # let's remove the labelled Indexes from unlabelled Indexes
        unlabelledIndexes = list(filter(lambda x: x not in labelledIndexes, unlabelledIndexes))
        
        self.labelledDataset = self.unlabelledDataset.iloc[labelledIndexes]
        self.unlabelledDataset= self.unlabelledDataset.iloc[unlabelledIndexes]

        self.labelledDataset.reset_index(inplace=True, drop=True)
        self.unlabelledDataset.reset_index(inplace=True, drop=True)

        # self.labelledDataset = torch.utils.data.Subset(self.unlabelledDataset, labelledIndexes)
        # self.unlabelledDataset= torch.utils.data.Subset(self.unlabelledDataset, unlabelledIndexes)
        # now we have the labelled dataset we can substract the labelled datset from unlabelled one
    
    def getlabelledDataset(self, labelledIndexes):
        unlabelledIndexes = range(len(self.unlabelledDataset))
        unlabelledIndexes = list(filter(lambda x: x not in labelledIndexes, unlabelledIndexes))

        self.labelledDataset = pd.concat([self.labelledDataset, self.unlabelledDataset.iloc[labelledIndexes]])
        self.unlabelledDataset = self.unlabelledDataset.iloc[unlabelledIndexes]

        self.labelledDataset.reset_index(inplace=True, drop=True)
        self.unlabelledDataset.reset_index(inplace=True, drop=True)
        
        # self.labelledDataset = torch.utils.data.ConcatDataset([self.labelledDataset, torch.utils.data.Subset(self.unlabelledDataset, labelledIndexes)])
        # self.unlabelledDataset= torch.utils.data.Subset(self.unlabelledDataset, unlabelledIndexes)

    @property
    def get_unlabelled_dataset(self):
        return len(self.unlabelledDataset)
    
    @property
    def get_labelled_dataset(self):
        return 0 if self.labelledDataset is None else len(self.labelledDataset) 



