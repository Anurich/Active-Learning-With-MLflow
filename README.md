# Active Learning With MLflow

### Task 
This repository shows how we can use Active Learning with MLflow. In this experimentation I utilise Active Learning to train <b> Sentiment Analysis task </b>, and use MLFlow to do the experiment tracking, model, code versioning. I also show how to do the code reproducibility using MLFlow by creating different files like  <b> MLporject, conda.yaml </b> which can be found inside src directory. 

### Active Learning.
* Active Learning provide a way to train the model using only the useful dataset and yet manage to achieve if not the best but comparetively similar result 
to the model trained with all the dataset.
* There exist many different strategies in Active Learning for sampling of dataset that can <b> uncertainity sampling, certainity sampling </b> or combination of both, etc. But in this case I used simple entropy sampling to sample the dataset. The idea is to simply show how we can combine Active Learning with real world example.  

### MLFlow.
* MLflow is a framework of MLOps, It give us many functionalities like:
      <ul>
     <li> Experiment Tracking.</li>
     <li> Data, Model, Code Versioning.</li>
     <li>Code Reproducibility.</li>
     </ul>

* In this experimentation I show:
      <ul>
      <li> How to do the experimentation tracking. </li>
      <li> Perform the Model, code versioning. </li>
      <li> Most important how to perform Code Reproducibility. </li>
      </ul>

### Dataset Information.
The dataset used is <b> Sentiment Analysis </b> taken from kaggle which can be downloaded from link <a href="https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews"> Download </a>

### Library Used.
  - python=3.7
  - pip:
    - mlflow==1.30.0
    - scipy==1.7.3
    - scikit-learn==1.0.2
    - torch==1.12.1+cu113
    - sentencepiece!=0.1.92
    - transformers==4.16.2
    - datasets>=1.8.0
    - seqeval==1.2.2
    - accelerate

### Note. 
Experiment in this repository is to show how we can combined different Methodology to create powerfull AI, this experiment can be improved further and its not the final version. Feel free to clone and experiment with it and improve it further.
