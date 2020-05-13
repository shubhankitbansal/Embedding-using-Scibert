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

import numpy as np
import pandas as pd
data=pd.read_csv('C00-2123_parsed_data.csv')

text=data['Word Re-ordering and DP-based Search in Statistical Machine Translation']
classes=data['paper_title']

text_list=list(text)
class_list=list(classes)
text_embeddings_1=get_embeddings(text_list)

text_embeddings_1 = np.array(text_embeddings_1)
class_embeddings=np.array(class_list)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(class_embeddings)
class_embeddings = le.transform(class_embeddings)

x_train,x_test,y_train,y_test=train_test_split(text_embeddings_1,class_embeddings,test_size=0.3,random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
clf = SVC(gamma='scale',random_state=42,kernel='linear',C=10.0)
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
print("random")
print("Accuracy Score :",accuracy_score(y_test,y_predict))
print(classification_report(y_test,y_predict))
