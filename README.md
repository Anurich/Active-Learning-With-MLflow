# Active Learning With MLflow

### Task 
This repository shows how we can use Active Learning with MLflow. In this experimentation I utilise Active Learning to train <b> Sentiment Analysis task </b>, and use MLFlow to do the experiment tracking, model, code versioning. I also show how to do the code reproducibility using MLFlow by creating different files like  <b> MLporject, conda.yaml </b> which can be found inside src directory. 

### Active Learning.
* Active Learning provide a way to train the model using only the useful dataset and yet manage to achieve if not the best but comparetively similar result 
to the model trained with all the dataset.
* There exist many different strategies in Active Learning for sampling of dataset that can <b> uncertainity sampling, certainity sampling </b> or combination of both, etc. But in this case I used simple entropy sampling to sample the dataset. The idea is to simply show how we can combine Active Learning with real world example.  

### MLFlow.
* MLflow is one of the frameworks of MLOps, It give us many functionality for example:
      * Experiment Tracking.
      * Data, Model, Code Versioning.
      * Code Reproducibility.


