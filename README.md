# Sentiment Analysis

Classifies movie reviews into positive and negative. Classification is done using different classification algorithms in python 3.

### Dataset

The dataset is referred to as “sentence polarity dataset v1.0“. It was collected by Bo Pang and Lillian Lee and released in 2005. First used in Bo Pang and Lillian Lee, "Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.", Proceedings of the ACL, 2005. The Movie Review data shows good/bad ("fresh"/"rotten") sentiment classification based on a collection of short review excerpts from Rotten Tomatoes. 

#### The data has been cleaned up somewhat, for example:

* The dataset is balanced, comprised of 5331 positive and 5331 negative processed senteces. 
* The dataset is comprised of only English reviews.
* All text has been converted to lowercase.
* There is white space around punctuation like periods, commas, and brackets.
* Text has been split into one snippet per line.


## Data processing pipeline:

* Firstly replace contractions to their equivalents;
* Then remove punctuation and numbers;
* Remove all the single characters;
* Replace multiple spaces with a single space;
* Replace slang words and abbreviations with their equivalents;
* Negation replacement. Dealing with negations (like “not good”) is a critical step in Sentiment Analysis. A negation word can influence the tone of all the words around it, and ignoring negations is one of the main causes of misclassification;
* Remove stopwords. Stop words are the most common words in a language like “the”, “a”, “on”, “is”, “all”. These words usually carry little importance to the sentiment analysis.
* Stemmimg. Reduce inflectional forms to a common base form. 
* Tokenization.


## Classification methods

Next some machine learning approaches are compared to see which is more likely to give an accurate analysis of sentiment. These approaches analyse a corpora of positive and negative Movie Review data by training and thereafter testing to get an accuracy score. 

Generative classifiers like naive Bayes build a model of each class. Given an observation, they return the class most likely to have generated the observation. Discriminative classifiers like logistic regression instead learn what features from the input are most useful to discriminate between the different possible classes. While discriminative systems are often more accurate and hence more commonly used, generative classifiers still have a role. Other classifiers commonly used in language processing include support-vector machines (SVMs), random forests, perceptrons, and neural networks.

I performed a set of experiments to recognize positive or negative sentiment, using single intelligent methods. It was observed that the better results were obtained by: Logistic regression and BernoulliNB, reaching the accuracy of 90% an 76% respectively. SVM also reaches a high accuracy (with linear kernal), but the training time takes much longer.

Logistic regression and BernoulliNB techniques serve as good primers for conducting sentiment analysis using machine learning techniques considering both accuracy and training time. The scikit-learn library simplifies the process into five major steps: training, vectorization, classification, prediction and testing.

### Logistic Regression

Logistic Regression is a ‘Statistical Learning’ technique categorized in ‘Supervised’ Machine Learning (ML) methods dedicated to ‘Classification’ tasks. It has gained a tremendous reputation for last two decades especially in financial sector due to its prominent ability of detecting defaulters. A contradiction appears when we declare a classifier whose name contains the term ‘Regression’ is being used for classification, but this is why Logistic Regression is magical: using a linear regression equation to produce discrete binary outputs.

Logistic Regression is a classification model that is very easy to implement and performs very well on linearly separable classes. It is one of the most widely used algorithms for classification in industry too, which makes it attractive to play with. Logistic regression can, however, be used for multiclass classification, but here we will focus on its simplest application. It is the go-to method for binary classification problems (problems with two class values). 

### BernoulliNB

The Naive Bayes (NB) classifier is widely used in machine learning for its appealing tradeoffs in terms of design effort and performance as well as its ability to deal with missing features or attributes. It is particularly popular for text classification. Multinomial Naive Bayes is in a sense more complex model, while BernoulliNB is particularly designed for binary/boolean features. Because of that, Bernoulli model can be trained using less data and be less prone to overfitting

There are two major advantages of the NB classification when working with binary features. First, the naive assumption of feature independence reduces the number of probabilities that need to be calculated. This, in turn, reduces the requirement on the size of training set. Another advantage of NB classification is that it is still possible to perform classification even if one or more features are missing, in such situations the terms for missing features are simply omitted from calculations.



## Unbalanced dataset problems

Data sets are unbalanced when at least one class is represented by only a small number of training examples (called the minority class) while other classes make up the majority. In this scenario, classifiers can have good accuracy on the majority class but very poor accuracy on the minority class(es) due to the influence that the larger majority class.


#### Imbalanced-learn library

A full-fledged python library designed specifically for dealing with these kinds of problems. This library is a part of sklearn-contrib. One of such examples is the BalancedRandomForestClassifier. The usual Random Forest algorithm performs extremely poorly on imbalanced datasets. However, this Balanced Random Forest Classifier which is a part of imblearn package works wonderfully well. It internally handles the sampling issues. Other features of the library include inbuilt oversamplers, undersamplers, a combination of both and other algorithms tailor-made for handling skewed datasets. 

#### Algorithmic approach

ML algorithms penalize False Positives and False Negatives equally. A way to counter that is to modify the algorithm itself to boost predictive performance on minority class. This can be executed through either recognition-based learning or cost-sensitive learning.

At the algorithmic level, solutions include adjusting the costs of the various classes so as to counter the class imbalance, adjusting the probabilistic estimate at the tree leaf (when working with decision trees), adjusting the decision threshold, and recognition-based (i.e., learning from one class) rather than discrimination-based (two class) learning.

With gradient boosted trees it is possible to train on much more unbalanced data, like credit scoring, fraud detection, and medical diagnostics, where percentage of positives may be less that 1%. For example, with algorithm like XGBoost, you can set the scale_pos_weight hyperparameter of the algorithm to indicate that your dataset has a certain ratio of positive and negative classes and XGBoost will take care of the rest. So, in a example of 40,000 positive and 2,000 negative samples, if we want to train our XGBoost classifier on this dataset we would set the value of scale_pos_weight to be 40,000/2,000 = 20. This works very well in practice. However, one downside is that this restricts you to using XGBoost and other similar algorithms since not all algorithms have this adjustable hyperparameter.

In problems where it is desired to give more importance to certain classes or certain individual samples keywords class_weight and sample_weight can be used. SVC (but not NuSVC) implement a keyword class_weight in the fit method. It’s a dictionary of the form {class_label : value}, where value is a floating point number > 0 that sets the parameter C of class class_label to C * value. SVC, NuSVC, SVR, NuSVR and OneClassSVM implement also weights for individual samples in method fit through keyword sample_weight. Similar to class_weight, these set the parameter C for the i-th example to C * sample_weight[i].
