# Fake-News-Classifier
1) Dataset
In this project we have used a simple and realistic dataset that contains 10240 news articles simply classified as True or False. It had marginally more real news than  This dataset will be used for training our classifier. Link to the dataset we used is given below:
https://www.kaggle.com/ksaivenketpatro/fake-news-detection-dataset#train.csv

2) Removing stop words 
Removing the english stop words were done using the remove stop words parameter of the Tf-Idf vectorizer where stop words remover parameter was given the argument  english_stop_words 

3) Removing commonly occurring Words and outliers 
Words which occurred in more than 85% of  the documents were removed. The above process would remove words like “and”, ”the” which can skew the results. Similarly it was done for rarely occurring words which occurred less than in 1% of the documents as it can have a similar effect. 

4) The entire corpus  is then vectorized which results in a 10240x156 sparse matrix, where 10240 are the number of news articles and 156 are the vectors for each article. Most of the values for the vectors are 0 hence the above matrix is a sparse matrix. 
We chose TFIDF vectorizer over other text vectorizing techniques like count vectorizer because, CountVectorizer gives you a vector with the number of times each word appears in the document. Words that carry the topic information appear less frequently in a document. Therefore, if vectorizing technique like Count Vectorizer is used, the vectors generated will not truly represent the dataset. 
What TF-IDF does is it balances out the term frequency (how often the word appears in the document) with its inverse document frequency (how often the term appears across all documents in the data set). Therefore,  TF-IDF will give higher scores to the words that carry the topic information and thus they’ll be the ones that the model identifies as important and tries to learn.

5)The vectorized corpus is then split into the training and testing set where 80% is constitutes the training and 20% constitutes the testing test. In the training and testing test the news article is in one class while the label is in another class 


      


 6.  Passing the training set into different classifiers

Since it is a classification problem, we have used Logistic Regression, Random Forest Classifier, and XGBoost classifier. Logistic regression is a simple algorithm whereas Random Forest and XGBoost are more advance. 

6.1 Logistic Regression

Logistic regression is an effective method for binary classification problems. As the aim of our model is to simply classify the news article as true/false, logistic regression is a good choice. 
The logistic function is an S-shaped curve that can take any real-valued number and map it into a value between 0 and 1, but never exactly at those limits. Input values (x) are combined linearly using weights or coefficient values to predict an output value (y). 
logistic regression equation:
y = e^(b0 + b1*x) / (1 + e^(b0 + b1*x))
Where y is the predicted output, b0 is the bias or intercept term and b1 is the coefficient for the single input value (x).
We first create an object of Logistic Regression

	

The logistic regression classifier is trained by passing the news article training set into the fit function. After it is trained, predictions are made on the test set using the predict function.

F1 score and accuracy is calculated to understand the performance of the classifier. In simple terms, accuracy is the fraction of predictions our model got right. 
Accuracy can tell us immediately whether a model is being trained correctly and how it may perform generally. However, it does not give detailed information regarding its application to the problem. The problem with using accuracy as your main performance metric is that it does not do well when you have a severe class imbalance. But as our dataset realistically skewed with around 50% of the dataset being false news and the other 50% being true news, Accuracy is a realistic measure to understand the performance of the classifier.
accuracy = (number of articles classified correctly) / (total number of news articles.)

F1 score
The F1 Score is the 2*((precision*recall)/(precision+recall)). It is also called the F Score or the F Measure. Put another way, the F1 score conveys the balance between the precision and the recall. F1 is an overall measure of a model’s accuracy that combines precision and recall.
Precision basically tells us about, Out of all the cases that the system says ‘true’, how many are actually ‘true’.
Recall tells us about, out of all the total positive cases, how many positive cases were identified correctly by the system.

6.2 Random Forest Classifier

Random forest algorithm is a supervised classification algorithm. It creates a forest with a number of decision trees. How the decision tree works is:

i.Given the training dataset with targets and features, the decision tree algorithm will come up with some set of rules. The same set rules can be used to perform the prediction on the test dataset.

ii.Therefore in a forest classifier, each tree comes up with a certain set of rules. If more than one tree comes up with the same classification rule, the probability of the classifier using that rule to classify the test input increases.
Formally, how it works is:
Takes the test features and use the rules of each randomly created decision tree to predict the outcome and stores the predicted outcome (target)

iii.Calculate the votes for each predicted target.

iv.Consider the high voted predicted target as the final prediction from the random forest algorithm.


6.3 XGBoost classifier

XGBoost (Extreme Gradient Boosting) belongs to a family of boosting algorithms and uses the gradient boosting (GBM) framework at its core. It is an optimized distributed gradient boosting library.
In this method of classification a number of weak classifiers are made and the results are combined to give a higher prediction accuracy. The entire training set is passed on to the classifier first. It results in a weak classifier. All the data items that were incorrectly classified are combined and passed on to the classifier again. This process is repeated n times which results in an n number of weak classifiers.These weak learners are combined to give an improved prediction accuracy.

7. Cross Validation

Cross validation Score is calculated using cross_val_score(). lr_statement is the the object used to fit the data. In simple terms it is the object used to train the data. We test the vectorized original dataset against the trained dataset to compute a cross validation score. We are performing a 7 fold cross validation.
In 7-fold cross-validation, we split the dataset into 7 subsets hen we perform training on the all the subsets but leave one(k-1) subset for the evaluation of the trained model. In this method, we iterate k times with a different subset reserved for testing purpose each time.
Cross validation is a more accurate estimate of out-of-sample accuracy.

