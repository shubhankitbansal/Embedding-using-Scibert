# Embedding-using-Scibert

# Generate Embedding using scibert 
`SciBERT` is a `BERT` model trained on scientific text. This is work on every operating system. There is no requirement of GPU to run this project but GPU help to train a model faster than CPU. There is so many requests in google colaboratory so we working on localhost.
We train our model using Support Vector Machine.
## Prerequisites

 - Python
 - Tensorflow-gpu (version=1.15)
 - Scibert-scivocab-uncased
 - bert-as-service
 - bert-serving-server
## Download Tensorflow Model
- [scibert-scivocab-uncased](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz)  (**Recomended**)
## Installing all library
Now for working on embedding we use jupyter-notebook so we have to install library with required versions. Tensorflow is requred of version 1.15
Install the server using this command to enable http:

    pip install -U bert-serving-server[http]


Start the server in localhost:

    'bert-serving-start -model_dir=./scibert-scivocab-uncased -http_port 3333 &'
Install tensorflow to work in this project

    !pip install tensorflow-gpu==1.15
After all this stuff we start writing our code to embed the text in this code we request the server for embedding so open jupyter notebook in the environment in which all the library with required versions are installed:
``` 
import json
import requests
def get_embeddings(texts):
    headers = {
        'content-type':'application/json'
    }
    data = {
        "id":123,
        "texts":texts,
        "is_tokenized": False
    }
    data = json.dumps(data)
    r = requests.post("http://localhost:3333/encode", data=data, headers=headers).json()
    return r['result']
```
You get the embedding like so
```
list_item = ['cat','rat','dog']
embedded_list = get_embeddings(list_item)
```
We train our model in any **SVM** (Support Vector Machine) so read csv file using pandas and convert it into list of array using numpy as follows:
```
import numpy as np
import pandas as pd
data = pd.read_csv('file.csv')
```
Suppose in data we have two columns in which one is text in which we have to train and other is class that is our label of the text. So we this in different list as follows:
```
text = data['text']
class = data['class']
```
So start embedding our text this will take some time:
```
embedded_list = get_embeddings(text)
```
Convert embedded_list and class array into numpy array to divide into train and test model then train our model using svm so after dividing the data into train and test we have **x_train,x_test,y_train,y_test** to know how to divide into train and test data refer to link [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) . 
Start Training our model in Support Vector machine we have to import libraries
```
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
```
After importing library we train model as follows:
```
clf = SVC(gamma='scale',random_state=42,kernel='linear')
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
print("random")
print("Accuracy Score :",accuracy_score(y_test,y_predict))
print(classification_report(y_test,y_predict))
```
