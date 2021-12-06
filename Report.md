<p style="text-align: center;"><span style="font-size:28px;"><strong>Classification of Game Review - Based on Steam Game Comments</strong></span></p>
<p style="text-align: center;"> Yongrui Chen, Lin Meng, Haoyu Wang</p>
<p style="text-align: center;"> Georgetown University, Washington D.C, 20007, U.S.A</p>


# Introduction 
With the development of computer science technology and the world economy, the game industry experienced an exploding increase in recent years. Predicted by a top consulting agency, the annual revenue of the video game industry is forecast to reach $180.1 billion in 2021. A highly increasing industry nurtured a more advanced user ecosystem. Professional video game platforms such as Steam and Xbox covered more and more market shares and gradually became the first destination for users to discover new games. 

In Steam, users give comments on games, which provide a reference for other new users on game selection. To some extent, the game review is a major factor affecting game sales. Interpreting the sentiment of game reviews becomes important, e.g., if the review content shows the recommendation or not? In this research, a review sentiment recognition system is created based on NLP (Natural Language Processing) methods such as BERT(Bidirectional Encoder Representations from Transformers), Logistics, Naïve Bayes, linear SVC (linear support vector machine), Decision Tree algorithm. A python package is created for recognizing whether the given review input is recommendation or not.

# Project Methodology

## Data Cleaning & Exploratory data analysis(EDA)
The dataset used in the research is from the Steam game reviews which has more than 330 thousand rows. After checking out the basic layout of the dataset, a preprocess function is created to regularly express the review content. The characters except for the alphabet and multiple spaces are replaced. All letters are converted to lowercase. In table 1, it  shows the introduction of all the variables in the data. Also,  there is a small sample of data in figure 1.

<br>

 

<p align="center">
   <img src=Visualizations/SampleData.png alt="Figure 1:The five lines of the dataset"/>
</p>

  
<p align="center">Figure 1:The five lines of the dataset</p>
  


<center>

| Variable Name | Description |
|------------|---------------|
| recommendation | Whether the reviewer recommended the game or not |
| review | The content of the review |<br>
  
 </center>
 
 <center>
Table 1: Data Variable Description

</center>

<br>
To figure out the meaningless reviews, a histogram of text length is created. Figure 2 shows the distribution of review length. There are many reviews under 5 words, which are not long enough to convey sentiments. Hence, only the reviews with more than 5 words will be used in the following analysis.

<center>

![Figure 2:Review lengths distribution](Visualizations/Histogram.png)<br>
Figure 2:Review lengths distribution

</center>

After obtaining the clean dataset, two WordClouds for recommended reviews and non-recommended reviews are created, showing the frequency of words. It can be seen that words like the game, play, time are significant in both recommended and non-recommended reviews. Words like a friend, good, fun which have positive meanings are shown in recommended reviews, while negative words like don’t, bad and Rockstar show in non-recommended reviews.

<center>

![Figure 3:Recommendation word cloud](Visualizations/wordcloud1.jpg)
Figure 3:Recommendation word cloud

![Figure 4:Not recommendation word cloud](Visualizations/wordcloud2.jpg)
Figure 4:Not recommendation word cloud

</center>

## Linear SVC
Linear SVC is a supervised learning model with associated learning algorithms that analyze data for classification and regression. A "best fit" hyperplane that divides, or categorizes with fitting the data can be returned by Linear SVC.

Support Vector Machines with the right setting can deal with the high dimension data. Hence, it can offer a high prediction speed. Moreover, linear SVC can work for both classification and regression data analysis. 

However, there are still some weaknesses of SVM. SVM doesn't provide sophisticated and uninterpretable reports. Moreover, there are not too many choices for model visualization. On the other hand, it doesn’t work well when the total feature of test data exceeds total training data samples. This will lead to this algorithm isn’t suitable for larger datasets. Last, it requires scaling data before training models. This is not necessarily a disadvantage but it will give people extra work for using linear SVC.

## Logistic Regression
Logistic regression is a predictive analysis method like all regression methods.  Logistic regression is used to describe data and explain the relationship between one dependent binary or multiple variables and one or more nominal, ordinal, interval, or ratio-level independent variables.

Logistic regression can offer the probability prediction, not only the classification label. It gives more information for the following analysis. Logistic regression is a fast and resource-friendly algorithm it scales. Hence, the time consumption of logistic regression is less. It’s very friendly with the large size data. Generally, the accuracy of logistic regression is very high.  However, it still gets weak. Logistic regression has several requirements for the data. If the relationships between features are complex, it may not work well using logistic regression, because the logistic regression algorithm is linear and has lower accuracy when dealing with non-linear data. Also, if the data is in a lower amount, the accuracy will be very low.  Moreover, logistic regression can easily overfit high-dimensional calculation. 

## Decision Tree

A decision tree is the most powerful and popular tool for classification and prediction. A decision tree is a flowchart-like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.

There are three major strengths of decision tree methods, that decision trees are able to generate understandable rules, are able to perform classification without requiring much computation, and are able to handle both continuous and categorical variables. Decision trees provide a clear indication of which fields are the most important for prediction or classification.

The weaknesses of the decision tree method are that decision trees are less appropriate for estimation tasks when the goal is to predict the value of a continuous attribute. Decision trees are prone to errors in classification problems with many classes and a relatively small number of training examples.

A decision tree can be computationally expensive to train. The process of growing a decision tree is computationally expensive. At each node, each candidate splitting field must be sorted before the best split can be found. In some algorithms, combinations of fields are used and a search must be made for optimal combining weights. Pruning algorithms can also be expensive since many candidate sub-trees must be formed and compared. When the dataset is super large, the tree plot will be really unreadable and meaningless.

## Naïve Bayes
Naïve Bayes models are called ‘naïve’ algorithms because they make an assumption that the predictor variables are all independent of each other. In other words, the presence of a certain feature in a dataset is completely unrelated to the presence of any other features. Figure 5 shows the algorithms of the Naïve Bayes.

<center>

![Figure 5:Recommendation word cloud](Visualizations/NavieBayes.png)
Figure 5: Bayes Theorem

</center>
The strengths of Naïve Bayes are that first, it’s an easy and quick way to predict classes, both for binary and multiclass classification problems. Second, it performs better even if less training data are applied, when the independence assumption fits. Third, the decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one-dimensional distribution. This helps with problems derived from the curse of dimensionality and improves performance.

The weaknesses of Naïve Bayes are that first NB is known as a poor estimator. The probability of the outputs shouldn’t be taken very seriously. Second, the naïve assumption of independence is very unlikely to match real-world data, since there is a rare situation that all features are independent of each other.

## Bidirectional Encoder Representations from Transformers (BERT)

BERT makes use of a transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. In its vanilla form, the transformer includes two separate mechanisms — an encoder that reads the text input and a decoder that produces a prediction for the task. Since the goal of BERT is to generate a language model, only the encoder mechanism is necessary.

As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. Therefore, it is considered bidirectional, though it would be more accurate to say that it’s non-directional. This character allows the model to learn the context of a word based on all of its surroundings (left and right of the word).

Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a [MASK] token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence. 

The BERT loss function takes into consideration only the prediction of the masked values and ignores the prediction of the non-masked words. As a consequence, the model converges slower than directional models, a characteristic that is offset by its increased context-awareness.
<center>

![Figure 6: BERT Masked LM](Visualizations/BERT1.png)
Figure 6: BERT Masked LM

</center>
In the BERT training process, the model receives pairs of sentences as input and learns to predict if the second sentence in the pair is the subsequent sentence in the original document. During training, 50% of the inputs are a pair in which the second sentence is the subsequent sentence in the original document, while in the other 50% a random sentence from the corpus is chosen as the second sentence. The assumption is that the random sentence will be disconnected from the first sentence.

<center>

![Figure 7: Next Sentence Prediction (NSP)](Visualizations/BERT2.png)
Figure 7: Next Sentence Prediction (NSP)

</center>

# Model results & Performance

## Linear SVC
After applying the Linear SVC model onto the game reviews, the precision and accuracy of the model is 82% and 85% respectively. In the meanwhile, a confusion matrix(Figure 8) and feature importance bar plots(Figure 9) were created to provide a more readable interpretation. 
<center>

![Figure 8: Confusion Matrix of Linear SVC](Visualizations/SVM_cm.png)

Figure 8: Confusion Matrix of Linear SVC

</center>
From the Figure 8 below, it is obvious that the top 5 most crucial words supporting recommendation reviews are “best”, “amazing”, “addicting”, “love” and “highly”, whereas “worst”, “refund”, “unplayed”, “garbage” and “ruined” would explain more for negative reviews. 

<center>

![Figure 9: Feature importance of Linear SVC](Visualizations/SVM_Imp.png)

Figure 9: Feature importance of Linear SVC

</center>

## Logistic Regression

The evaluation of the Logistic Regression model is nearly the same as that of the Linear SVC model (82% for precision and 85% for accuracy). The confusion matrix (Figure 10)  and ROC curve (Figure 11) are created for more detailed information.


<center>

![Figure 9: Feature importance of Linear SVC](Visualizations/SVM_Imp.png)

Figure 9: Confusion Matrix of Logistic Regression

</center>

In accordance with the Figure 10 below, the blue curve stands for the result of our prediction. The further the blue curve is away from the diagonal, the better the model performed. Therefore, it can be concluded that the Logistic Regression model had a relatively excellent performance. 

<center>

![Figure 10: Feature importance of Linear SVC](Visualizations/SVM_Imp.png)

Figure 10: ROC of Logistic Regression

</center>

<center>

![Figure 11: Feature importance of Linear SVC](Visualizations/SVM_Imp.png)

Figure 11: Feature importance of Logistic Regression

</center>


## Decision Tree

When the dataset is super large, the training process of the Decision Tree model will be relatively highly time-consuming and the tree plot could be less understandable. Hence, the visualization of the decision tree falls to accuracy and feature importance of cross-validation. In line with Figure 12, the accuracy of cross-validation is 76% which is lower than that of the Logistic and Linear SVC Model. However, the bar plot of feature importance told us a similar story. 
<center>

![Figure 12: Confusion Matrix of Decision Tree](Visualizations/DT_cm.png)

Figure 12: Confusion Matrix of Decision Tree

</center>

From the Figure 13 below, it can be easily acknowledged that some common words like “money”, “best”, “fun”, “great”, “love” and “superb” were deeply involved in reviews, although they might be within a different order in different cross validation results.

<center>

![Figure 13: Feature importance of Decision Tree](Visualizations/DT_Imp.png)

Figure 13: Feature importance of Decision Tree

</center>

## Naïve Bayes

Using the Naïve Bayes classifier, the evaluation results were slightly lower than those of the Linear SVC and Logistic models, but higher than those of the Decision Tree model (79% for precision and 81% for accuracy). The confusion matrix and feature importance are shown below. 

<center>

![Figure 14: Confusion Matrix of Decision Tree](Visualizations/NB_cm.png)

Figure 14: Confusion Matrix of Naïve Bayes

</center>

Looking at Figure 14, words like “greedy”, “interactive” and “selling” are highly related to the type of the reviews, even if they might be ordered differently in different cross validation results, nonetheless, overlaps did exist. 

<center>

![Figure 15: Feature importance of Decision Tree](Visualizations/NB_Imp.png)

Figure 15: Feature importance of Naïve Bayes

</center>

## BERT

<center>

![Figure 16: Feature importance of Decision Tree](Visualizations/BERT_cm.png)

Figure 16: Confusion matrix of BERT classifier

</center>

<center>

![Figure 17: BERT classification with epochs](Visualizations/BERT3.jpg)

Figure 17: BERT classification with epochs

</center>

# Conclusion

According to the experiment and visualization above, there are some words in review that are highly related to the recommendation or not. It is possible to predict the recommendation by the words in the review. In table 2, the comparison between different model performances is created showing the accuracy, precision, and F1 score of each model. After comparing the evaluation score, time consumption, interpretability, and visualization performance, it can be concluded that logistic regression will be the best choice which is significantly higher performance in the comparing variables and only cost a little time to show the result.

| Model | Accuracy | Precision | F1 Score |
|------------|---------------|---------------|---------------|
| Decision Tree | 0.76 | 0.76 | 0.83 |
| Naive Bayes | 0.79 | 0.81 | 0.85 |
| Bert | 0.81 | 0.78 | 0.77 |
| Linear SVC | 0.82 | 0.85 | 0.87 |
| Logistic Regression | 0.82| 0.85 | 0.87 |
