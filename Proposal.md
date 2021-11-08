# ANLY-580-Final Project Proposal
## Group Member:
Lin Meng (lm1398)<br>
Haoyu Wang (hw468)<br>
Yongrui Chen (yc910)<br>


## Brief: 
Nowadays, many people prefer playing video games to release their press or have fun in their free time. Some of them may be confused about how to choose a game which is worth playing. Steam is one of the most popular game platforms. Review and recommendation will be a primary point which people will be checked. However some reviews and recommendations may be deceptive. We will use a dataset of about 40 thousand reviews of games in the steam. Hence we will analyze the review from users to predict the truth of the review and recommendation. Our final goal is to distinguish  whether the game is recommended or not and whether the recommendation of a game is true or not by entering a review.
 
## Description 

### Variables
There are 8 variables in the origin data. However, there are 4 variables will be used. All the variables and description of each variable are in the below table. The data link is in the below.<br>
Data Link: Stream Reviews: https://www.kaggle.com/luthfim/steam-reviews-dataset<br>
| Variable Name | Description |
|------------|---------------
| helpful | How many other players think the review is helpful  |
| hour_played | How many hours a reviewer play the game when making a review |
| recommendation | Whether the reviewer recommended the game or not |
| review | The content of the review |

### Satitiscal Summary
||helpful|hour_played|
|------------|---------------|---------------|
|Min|0|0|
|1st Quantile|0|62|
|Median|0|190|
|Mean|1.04|364.1|
|3rd Quantile|0|450|
|Max|28171|31962|

### EDA
<img src='newplot.png'>
{% include Visualizations/distribution_of_labels.html %}


#### Label
We plan to use create a label with 6 classes. We will base on the quantile of helpful and hour played to divide all the data into three classes which are truth, doubted, and deceptive. Then we combine these with the recommendation to get our final label with 6 classes which are Recommended_Truth, Recommended_Doubted,Recommended_Deceptive, Not_Recommended_Truth，Not_Recommended_Doubted, and Not_ Recommended_Deceptive.

## Method:
1. What modeling approach do you intend to use?
	<br>1.SVM/Naive Bayes
	<br>2.Decision Tree
	<br>3.Logistic
	<br>4.(Probably) BERT 
	<br>
2. How will your system be evaluated and what are the evaluation criteria?<br>
	We will split data to train data and test data by 70% and 30%. Use the model created by train data to predict the recommendation of test data. If the rate of accuracy, precision, F1-Score is about 80%, we will conclude our model will pass the evaluation.<br>
3. Are there any special computational/hardware considerations?<br>
	We have approximately 40 thousand reviews from steam data. Although the data  is not  very  big, we may need to use a neural networking algorithm.  If it wok slow on our laptop, we may need to use the AWS.
4. What are the biggest unknowns that might dictate the success or failure of this project?<br>
	We classify the review as deceptive and truthful based on the hour played and helpful number. If the number of helpful answers is greater than medium or hour played is greater than 75% quantile, we will say this review is true, otherwise it is deceptive.<br>

## Result:
Report+Presentation (Demo maybe)



