import torch
import itertools
from tqdm import tqdm
from customDataset import dataset
class strategies:
    def __init__(self) -> None:
        pass

    def predict_proba(self, dataloader, rows):
        self.model.eval()
        batchIDs = []
        data = torch.ones([rows, 2])
        start = 0
        counter =0 
        with torch.no_grad():
          for  batch_idx, element in tqdm(dataloader,position=0, leave=True):
              out = self.model(**element)
              pred = torch.softmax(out.logits, dim=-1)
              end = start + element["input_ids"].shape[0]
              data[start:end] = pred
              start = end
              batchIDs.append(batch_idx.tolist())
              counter +=1
        return data, batchIDs

    def entropySampling(self, ac_learning, model, accelerator, tokenizer):
        self.model =model
        unlabelledDataset = dataset(ac_learning.unlabelledDataset["preprocessedREVIEW"], ac_learning.unlabelledDataset["label"], tokenizer)
        dataloader = torch.utils.data.DataLoader(unlabelledDataset, batch_size=32,shuffle=False)
        self.model, dataloader = accelerator.prepare(self.model, dataloader)

        probs, batchIDs =  self.predict_proba(dataloader, len(unlabelledDataset))
        log_prob = torch.log(probs)
        batchIDs = list(itertools.chain(*batchIDs))
        return (-probs * log_prob).sum(1), batchIDs


    def entropySamplignDropout(self):
        pass