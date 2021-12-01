#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import os
import pandas as pd
from colorama import Fore, Style
import plotly.express as px
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# In[3]:


def demo(text):
    file=os.path.dirname(os.getcwd())
    model=pickle.load(open(file+'\\Model\\log_model.pkl', 'rb'))
    Tfidf_reco=pickle.load(open(file+'\\Model\\Tfidf_reco.pkl', 'rb'))
    DTM=Tfidf_reco.transform([text])
    result=model.predict(DTM)[0]
    feature_importances = pd.DataFrame(model.coef_.T,
                                   index = model.feature_names_in_,
                                   columns=['importance']).sort_values('importance', ascending=False)
    feature_importances=feature_importances['importance']*np.var(feature_importances['importance'])
    feature_importances=feature_importances.sort_values(key=abs,ascending=False)
    positive=list(feature_importances[feature_importances>0].index)
    negative=list(feature_importances[feature_importances<0].index)
    print( '***********Result**********')
    if result =='Recommended':
        print('This review is',Fore.LIGHTGREEN_EX+model.predict(DTM)[0])
    else:
        print('THis review is',Fore.LIGHTRED_EX+model.predict(DTM)[0])
    def paint(word, good, bad):
        if word in good:
            return(Fore.LIGHTGREEN_EX + word)
        elif word in bad:
            return(Fore.LIGHTRED_EX+ word)
        else:
            return Style.RESET_ALL + word
    print(Style.RESET_ALL + '***********Text**********')
    print(' '.join(map(lambda word: paint(word, positive, negative),text.split())))
    prob=model.predict_proba(DTM)
    df=pd.DataFrame(prob.T,columns=['Rate'])
    df['Recommendation']=['Not Recommended','Recommended']
    fig = px.bar(df, x='Recommendation', y='Rate',color='Recommendation',
             color_discrete_sequence=['#ff796c','#c7fdb5'],
             title='Recommendation Rate')
    print(Style.RESET_ALL + '***********Visualization**********')
    fig.show()

# In[]:

text=input('Enter the review you want to check:')
demo(text)