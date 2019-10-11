# Sentiment Analyses

Classifies movie reviews into positive and negative. Classification is done using different classification methods in python 3.

### Dataset

The dataset is referred to as “sentence polarity dataset v1.0“. It was collected by Bo Pang and Lillian Lee and released in 2005. First used in Bo Pang and Lillian Lee, "Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.", Proceedings of the ACL, 2005. The Movie Review data shows good/bad ("fresh"/"rotten") sentiment classification based on a collection of short review excerpts from Rotten Tomatoes. 

### Data processing pipeline

#### The data has been cleaned up somewhat, for example:

* The dataset is balanced, comprised of 5331 positive and 5331 negative processed senteces. 
* The dataset is comprised of only English reviews.
* All text has been converted to lowercase.
* There is white space around punctuation like periods, commas, and brackets.
* Text has been split into one snippet per line.


#### To further proccess data:

* Firstly replace contractions to their equivalents;
* Then remove punctuation and numbers;
* Remove all the single characters;
* Replace multiple spaces with a single space;
* Replace slang words and abbreviations with their equivalents;
* Negation replacement. Dealing with negations (like “not good”) is a critical step in Sentiment Analysis. A negation word can influence the tone of all the words around it, and ignoring negations is one of the main causes of misclassification;
* Remove stopwords. Stop words are the most common words in a language like “the”, “a”, “on”, “is”, “all”. These words usually carry little importance to the sentiment analyces.
* Lemmatization. The aim of lemmatization, like stemming, is to reduce inflectional forms to a common base form. As opposed to stemming, lemmatization does not simply chop off inflections. Instead it uses lexical knowledge bases to get the correct base forms of words;
* Tokenization.


### Classification methods

I will compare two machine learning approaches to see which is more likely to give an accurate analysis of sentiment. Both approaches analyse a corpora of positive and negative Movie Review data by training and thereafter testing to get an accuracy score. The techniques are Support Vector Machines (SVM) and Logistic Regression.

Generative classifiers like naive Bayes build a model of each class. Given an observation, they return the class most likely to have generated the observation. Discriminative classifiers like logistic regression instead learn what features from the input are most useful to discriminate between the different possible classes. While discriminative systems are often more accurate and hence more commonly used, generative classifiers still have a role. Other classifiers commonly used in language processing include support-vector machines (SVMs), random forests, perceptrons, and neural networks.

I performed a set of experiments to recognize positive or negative sentiment, using single intelligent methods. It was observed that the better results were obtained by: Logistic regression and SVM.

Logistic Regression is a ‘Statistical Learning’ technique categorized in ‘Supervised’ Machine Learning (ML) methods dedicated to ‘Classification’ tasks. It has gained a tremendous reputation for last two decades especially in financial sector due to its prominent ability of detecting defaulters. A contradiction appears when we declare a classifier whose name contains the term ‘Regression’ is being used for classification, but this is why Logistic Regression is magical: using a linear regression equation to produce discrete binary outputs.

Logistic Regression is a classification model that is very easy to implement and performs very well on linearly separable classes. It is one of the most widely used algorithms for classification in industry too, which makes it attractive to play with. Logistic regression can, however, be used for multiclass classification, but here we will focus on its simplest application. It is the go-to method for binary classification problems (problems with two class values). 








