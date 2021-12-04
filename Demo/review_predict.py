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


def result(text):
    model=pickle.load(open('log_model.pkl', 'rb'))
    Tfidf_reco=pickle.load(open('Tfidf_reco.pkl', 'rb'))
    DTM=Tfidf_reco.transform([text])
    result=model.predict(DTM)[0]
    feature_importances = pd.DataFrame(model.coef_.T,
                                   index = Tfidf_reco.get_feature_names(),
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
    all_words=list(feature_importances.index)
    origin_words=text.split()
    review_words=[]
    words_value=[]
    for i in origin_words:
        if i in positive or i in negative:
            review_words.append(i)
            words_value.append(feature_importances[all_words.index(i)])
    df2=pd.DataFrame()
    df2['Words']=review_words
    df2['Value']=words_value
    def support(row):
        if row['Value']>0 :
            return('Recommended')
        else:
            return('Not recommended')
    df2['NP']=df2.apply (lambda row: support(row), axis=1)
    ro=['Recommended','Not recommended']
    df2['NP'] = pd.Categorical(df2['NP'], categories=ro, ordered=True)
    df2=df2.sort_values(by='NP',ascending= False)
    fig = px.bar(df2, x='Value',y='Words',color='NP',
                color_discrete_sequence=['#ff796c','#c7fdb5'],
                title='Review Key Words Importance', orientation='h')
    print(Style.RESET_ALL + '***********Review Key Words Importance**********')
    fig.show()
    prob=model.predict_proba(DTM)
    df=pd.DataFrame(prob.T,columns=['Rate'])
    df['Recommendation']=['Not Recommended','Recommended']
    fig = px.bar(df, x='Recommendation', y='Rate',color='Recommendation',
             color_discrete_sequence=['#ff796c','#c7fdb5'],
             title='Recommendation Probability')
    print(Style.RESET_ALL + '***********Recommendation Probability**********')
    fig.show()
# In[]
def demo():
    text=input('Enter the review you want to check. If you want quit, please input quit.')
    if text != 'quit':
        result(text)
        demo()
    else:
        print('Thank you.')
# In[]:
demo()