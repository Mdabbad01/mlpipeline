import pandas as pd

import pickle
import yaml
import mlflow
from sklearn.metrics import accuracy_score

import os

from urllib.parse import urlparse

import mlflow


os.environ['MLFLOW_TRACKING_USERNAME'] = 'Mdabbad01'
os.environ['MLFLOW_TRACKING_PASSWORD'] = "c47e22c12d8703485809535598522b6e25b6f42f"
mlflow.set_tracking_uri("https://dagshub.com/Mdabbad01/demodagshub.mlflow")

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
   data= pd.read_csv(data_path)
   X=data.drop(columns=["Outcomes"])
   y=data["Outcome"]
   
   mlflow.set_tracking_uri("https://dagshub.com/Mdabbad01/demodagshub.mlflow")
   
   ##loading the model from the disk
   model=pickle.load(open(model_path,'rb'))
   predictions=model.predict(X)
   accuarcy=accuracy_score(y,predictions)
   
   ##log metics to mlflow
   mlflow.log_metric("accuracy",accuarcy)
   print("Model accuracy:{accuarcy}")
   
   if __name__=="__main__":
       evaluate(params["data"],params["model"])
   