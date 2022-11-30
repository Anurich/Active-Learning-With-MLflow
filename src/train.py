from configs import CONFIG
import os 
import pandas as pd
from tqdm import tqdm
from customDataset import dataset
from ActiveLearning import ACLearning
import torch
import pprint
import mlflow 
import numpy as np
import math
from strategies import strategies
from accelerate import Accelerator
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler, AutoTokenizer
import argparse
os.environ["MLFLOW_NESTED_RUN"]="1"
def metric(pred, gt):
    return f1_score(pred, gt, average="weighted", zero_division=1), precision_score(pred, gt,  average="weighted", zero_division=1), recall_score(pred, gt,  average="weighted", zero_division=1)

def main():
  with mlflow.start_run():
    parser = argparse.ArgumentParser(
        description="Make multi-label text classification dataset from Altilia Skill Studio Annotations"
    )
    parser.add_argument("--grad_accumulation", help="output_dir")
    parser.add_argument("--budget", help="Enter budget")
    parser.add_argument("--trainFile", help="Train path")
    parser.add_argument("--testFile", help="evaluation path")
    parser.add_argument("--batch", help="Enter Batch Size.")
    parser.add_argument("--lr", help="learning rate")
    
    args = parser.parse_args()

    mlflow.log_params(vars(args))
    # training on all folds
    strat = strategies()
    fold= args.trainFile.split("datasets/")[-1].split("/")[0]
    mlflow.log_artifact(args.trainFile)
    mlflow.log_artifact(args.testFile)
    gradient_accumulation_steps = int(args.grad_accumulation)
    accelerator =  Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    # folder_path = os.path.join(CONFIG.MULTIFOLD, f"fold_{i}")
    # trainFile = os.path.join(folder_path, "train.csv")
    # testFile  = os.path.join(folder_path, "test.csv")

    trainData = pd.read_csv(args.trainFile)[["preprocessedREVIEW","label"]]
    testData  = pd.read_csv(args.testFile)[["preprocessedREVIEW", "label"]]
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels =len(trainData["label"].unique().tolist()) )
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    ac_learning = ACLearning(trainData)
    ac_learning.randomSample()

    # train and test dataset 
    traindataset = dataset(ac_learning.labelledDataset["preprocessedREVIEW"], ac_learning.labelledDataset["label"],tokenizer)
    testdataset  = dataset(testData["preprocessedREVIEW"], testData["label"], tokenizer)

    trainloader = DataLoader(traindataset, batch_size=int(args.batch), shuffle=True)
    testloader = DataLoader(testdataset, batch_size=int(args.batch), shuffle=False)

    #optimizer 

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    num_train_epochs = 100
    #num_update_steps_per_epoch = math.ceil(len(trainloader) / gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=float(args.lr))
    max_train_steps = num_train_epochs * len(trainloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=max_train_steps,
    )
    trainloader, testloader, model, optimizer, lr_scheduler = accelerator.prepare(trainloader, testloader, model, optimizer, lr_scheduler)
    
    for epc in range(num_train_epochs):
        print(f"############## Labelled Dataset: {ac_learning.get_labelled_dataset} && Unlabelled Dataset: {ac_learning.get_unlabelled_dataset} ##############")
        model.train()
        for data in tqdm(trainloader,position=0, leave=True):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                batch_idx, inputs = data
                output = model(**inputs)
                loss = output.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

        # we can perform testing here
        
        model.eval()
        f1_scores = []
        precisions = []
        recalls = []
        for data in tqdm(testloader,position=0, leave=True):
            batch_idx, inputs = data
            output = model(**inputs)
            logits = output.logits
            probab = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probab, -1).tolist()
            gt   = inputs["labels"].tolist()
            f1Score, precision, recall = metric(pred, gt)
            f1_scores.append(f1Score)
            precisions.append(precision)
            recalls.append(recall)
        scores = {
            "f1_score": np.mean(np.array(f1Score)),
            "precision": np.mean(np.array(precisions)),
            "recall": np.mean(np.array(recalls))
        }
        print("\n\n")
        pprint.pprint(scores)
        print("\n\n")
        mlflow.log_metrics(scores) # logging the metrics 
        # perform the ActiveLearning pipeline 

        probabilities, batchIds = strat.entropySampling(ac_learning, model, accelerator, tokenizer)
        assert len(probabilities.tolist()) == len(batchIds)
        probIndex = list(zip( probabilities.tolist(),batchIds))
        ot = sorted(probIndex, key = lambda x: x[0], reverse=True)
        indexes = [ind for score, ind in ot]
        # now use these indxes to extract next sample from unlabelled dataset
        ac_learning.getlabelledDataset(indexes[:int(args.budget)])
        traindataset = dataset(ac_learning.labelledDataset["preprocessedREVIEW"], ac_learning.labelledDataset["label"], tokenizer)
        trainloader = DataLoader(traindataset, batch_size=int(args.batch), shuffle=True) # create train loader again
        trainloader = accelerator.prepare(trainloader)
          # logging the model 
    mlflow.pytorch.log_model(model, f"model-{fold}")
  mlflow.end_run()
if __name__ == "__main__":
    main()