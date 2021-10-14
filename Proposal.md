# ANLY-580-Final Project Proposal
### Lin Meng, Haoyu Wang, Yongrui Chen


## Brief: 
Nowadays, many people prefer playing video games to release their press or have fun in their free time. Some of them may be confused about how to choose a game which is worth playing. Steam is one of the most popular game platforms. Review and recommendation will be a primary point which people will be checked. However some reviews and recommendations may be deceptive. We will use a dataset of about 40 thousand reviews of games in the steam. Hence we will analyze the review from users to predict the truth of the review and recommendation. Our final goal is to distinguish  whether the game is recommended or not and whether the recommendation of a game is true or not by entering a review.
 
## Description 
<img src='img/data.png'>

| Variable Name | Description |
|------------|---------------
| helpful | How many other player think the review is helpful  |
| hour_playred | How many hour a reviewer play the game before make a review |
| recommendation | Whether reviewer recommend the game or not |
| review | The text of user review |

## Method:
1. What modeling approach do you intend to use?
	<br>1.SVM/Naive Bayes
	<br>2.Decision Tree
	<br>3.(Probably) BERT: 
	<br>
2. What data do you intend to use?<br>
	Data Link: Stream Reviews: https://www.kaggle.com/luthfim/steam-reviews-dataset<br>
3. How will your system be evaluated and what are the evaluation criteria?<br>
	We will split data to train data and test data by 70% and 30%. Use the model created by train data to predict the recommendation of test data. If the rate of accuracy, precision, F1-Score is about 80%, we will conclude our model will pass the evaluation.<br>
4. Are there any special computational/hardware considerations?<br>
	We have approximately 40 thousand reviews from steam data. Although the data  is not  very  big, we may need to use a neural networking algorithm.  If it wok slow on our laptop, we may need to use the AWS.
5. What are the biggest unknowns that might dictate the success or failure of this project?<br>
	We classify the review as deceptive and truthful based on the hour played and helpful number. If the number of helpful answers is greater than medium or hour played is greater than 75% quantile, we will say this review is true, otherwise it is deceptive.<br>

## Result:
Report+Presentation (Demo maybe)



