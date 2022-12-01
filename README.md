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
  All these showned in this experimentation.

