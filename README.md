# Sentiment Analyses

Classifies movie reviews into positive and negative. Classification is done using two different classification methods in python 3. Accuracy achieved: 75 - 80 %

### Data processing pipeline
The data has been cleaned up somewhat, for example:

* The dataset is comprised of only English reviews.
* All text has been converted to lowercase.
* There is white space around punctuation like periods, commas, and brackets.
* Text has been split into one snippet per line.

To further proccess data:
* Remove all the special characters
* Remove all the single characters
* remove all the stopwords (except for not)
* get the stemwords
* tokenize the words

### Movie Review Dataset

The Movie Review data shows good/bad ("fresh"/"rotten") sentiment classification based on a collection of short review excerpts from Rotten Tomatoes. It wsas collected by Bo Pang and Lillian Lee and released in 2005. The dataset is referred to as “sentence polarity dataset v1.0“.

This data was first used in Bo Pang and Lillian Lee, "Seeing stars: Exploiting class relationships for sentiment categorization
with respect to rating scales.", Proceedings of the ACL, 2005.

The dataset is comprised of 5331 positive and 5331 negative processed senteces / snippets of movie reviews drawn from an archive of the rec.arts.movies.reviews newsgroup hosted at IMDB. 
