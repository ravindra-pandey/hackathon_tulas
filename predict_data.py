"""
importing the necessary libraries
"""
import os
import re
import pickle
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow import keras
from pdfminer.high_level import extract_text


encoder=pickle.load(open("encoder.pkl","rb"))
model1=keras.models.load_model("models/model.h5")
model2=pickle.load(open("models/XGBClassifier.pkl","rb"))
tfidf=pickle.load(open("models/tfidf.pkl","rb"))

def clean_data(text):
    """
    we are using this function to clean the data .
    """
    text=str(text)
    text=re.sub('[^A-Za-z0-9]'," ",text)
    text=text.lower()
    text=word_tokenize(text)
    wlm=WordNetLemmatizer()
    text=[wlm.lemmatize(i) for i in text if not i in set(stopwords.words("english"))]
    text=" ".join(text)
    return text
def create_dataset(path=None):
    """
    we are using this function to create a dataset from a given path by reading pdf available there.
    """
    dataset={"resume_data":[],"user_id":[]}
    for p in os.listdir(path)[:10]:
        if ".pdf" in p:
            text=extract_text(os.path.join(path, p))
            text=clean_data(text)
            dataset["resume_data"].append(text)
            dataset["user_id"].append(p.split(".")[0])
    dataset=pd.DataFrame(dataset)
    return dataset

def test_data(path):
    """
    This function tests the dataset.
    """
    dataset=create_dataset(path)        
    X=tfidf.transform(dataset["resume_data"]).toarray()
    X=X.reshape(X.shape[0], 1,X.shape[1])
    prediction=model1.predict(X)
    
    return prediction,dataset[["user_id"]]

def filter_data(prediction,uid,category=None):
    outputs=np.argmax(prediction,axis=1)
    out_data={"index":[],"out1":[],"perc1":[],"out2":[],"perc2":[],"out3":[],"perc3":[]}
    
    for out in prediction:
        arr=[]
        for idx,i in enumerate(sorted(out,reverse=True)[:3],1):
            arr.append(list(out).index(i))
            out_data[f"perc{idx}"].append(np.round(i*100,3))
        
            
        out_data["index"].append(arr)

    
    for out in out_data["index"]:
        for idx,outp in enumerate(encoder.inverse_transform(out),1):
            out_data[f"out{idx}"].append(outp)
    del out_data["index"]
    outputs=pd.concat([uid,pd.DataFrame(out_data)],axis=1)
    if category is None:
        return outputs 
    else:
        return outputs[outputs["out1"]==category]
