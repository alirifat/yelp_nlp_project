# Recommend Helpful Customer Reviews to Businesses

<p align="center">
  <img src="https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/figures/yelp-logo.png" width="800" height="500">
</p>

<i>We aim to predict helpful reviews based on the features that are extracted from the review corpus. Our purpose is to enhance customer interaction and help businesses to establish profitable customer relationships.</i>



## Table of Contents

- [Introduction](#Introduction)
	* [About Yelp Data](#About-Yelp-Data)
- [Customer Reviews and eWOM](#Customer-Reviews-and-eWOM)
- [Why Helpful Reviews Matter?](#Why-Helpful-Reviews-Matter?)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
	* [Where the Businesses are Located](#Where-the-Businesses-are-Located)
	* [Businesses in the United States](#Businesses-in-the-United-States)
	* [What are the Industries?](#What-are-the-Industries?)
	* [Businesses in the Restaurant Industry](#Businesses-in-the-Restaurant-Industry)
	* [Statistical Summaries of the Restaurant Industry](#Statistical-Summaries-of-the-Restaurant-Industry)
	* [Distribution of Business Star Ratings and Review Counts](#Distribution-of-Business-Star-Ratings-and-Review-Counts)
	* [Distribution of Helpful Reviews and Their Relationship with Time](#Distribution-of-Helpful-Reviews-and-Their-Relationship-with-Time)
	* [Distribution of Review Star Ratings and Helpful Reviews](#Distribution-of-Review-Star-Ratings-and-Helpful-Reviews)
	
- [Data Cleaning and Feature Extraction](#Data-Cleaning-and-Feature-Extraction)
	* [Basic Text Features](#Basic-Text-Features)
	* [Data Cleaning](#Data-Cleaning)
	
- [Predict Helpful Reviews](#Predict-Helpful-Reviews)
	* [Hyperparameter Optimization](#Hyperparameter-Optimization)
	* [Final Evaluation](#Final-Evaluation)
	
- [Conclusion](#Conclusion)



## Introduction

Customer reviews are an essential source of information for many potential buyers. It is possible to reach thousands of product reviews via different mediums. However, it is not the best practice to read all customer reviews before purchasing a product. This kind of approach will come with a high cost of time.  For this reason, the potential buyers need to reach the most helpful customer reviews with minimum time and effort to use their resources more efficiently. 

#### About Yelp Data

[Yelp academic dataset](https://www.yelp.com/dataset) is available for free and open to the public. It's a well-known dataset in NLP (Natural Language Processing) research due to the detailed information provided to each business, user, customer review. Moreover, the dataset consists of six files: 

* `business.json` contains information about each company such as name and location, attributes, working hours, etc.
* `review.json` has information about each posted review such as user id, star rating, the customer review itself, number of useful votes, etc.
* `user.json` provides information about each Yelp user such as first name, the total number of reviews, the list of friends, the average star rating, etc.
* `checkin.json` has information about check-in for each business, such as business id and date. 
* `tip.json` (the shorter version of reviews and conveys quick suggestions to the businesses) provides information such as the tip itself, the number of compliments and dates, etc. 
* `photo.json` contains information about each photo uploaded to Yelp, such as photo id and photo label, etc. 


---

[See the full report](https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/Final%20Report.pdf)  
[Go to Table of Contents](#Table-of-Contents)

---



## Customer Reviews and eWOM

Online customer reviews can be defined as comments or opinions on a specific product posted on the company or a third-party website by peers [(Mudambi and Schuff, 2010)](https://www.jstor.org/stable/20721420?seq=1). Moreover, they can be seen as an outcome of a customer’s experience with a product and an input for a potential customer’s buying process. For this reason, customer reviews should not be considered as a narrow two-way relationship between the customer and the brand. Still, they may result in a more significant effect on business performance by affecting potential customers. 

On the other hand, in the information search process, customers would like to access word-of-mouth (WOM) to mitigate uncertainty and perceived risk [(Xie, etc., 2014)](https://www.sciencedirect.com/science/article/abs/pii/S027843191400125X). Traditionally, WOM is done by contacting friends, peers, people who have experience with the product, etc. However, as e-commerce is prevalent, online customer reviews can be considered a part of WOM and can be defined as eWOM (electronic word-of-mouth) [(Bronner and Hoog, 2010)](https://journals.sagepub.com/doi/abs/10.1177/0047287509355324). 

---

[Go to Table of Contents](#Table-of-Contents)

---



## Why Helpful Reviews Matter?

Customer reviews are primary sources of information for potential buyers on the internet. However, a company has to do more than just presenting customer reviews to potential buyers. As the number of customer reviews grows, the amount of resources that potential buyers have to allocate to complete the purchasing process also grows. Moreover, low-quality customer reviews can change potential buyers' minds and cause loss of business relationships. For this reason, a business needs to select and present helpful reviews to build and sustain profitable relationships with customers [(Park, 2018)](https://www.mdpi.com/2071-1050/10/6/1735). 

Predicting customer reviews' helpfulness gives business control over the customer reviews by promoting the possible ones. It is also convenient for potential buyers since it significantly reduces the search for material information about the product. Finally, it increases the dataset's efficiency by going through each customer's reviews instead of focusing on a limited number of customer reviews. 

---

[Go to Table of Contents](#Table-of-Contents)

---



## Exploratory Data Analysis

We want to start with a brief description of the Yelp academic dataset. It consists of 209,393 businesses, 1,968,703 Yelp users and 8,021,122 customer reviews. 

The businesses have an average of 36 customer reviews with a maximum of 10,129. Also, the average number of reviews posted by a user is 22, with a maximum of 14,455. Almost half of the customer reviews (46%) has at least one helpful vote, and the average number of helpful votes is 1.32 with a maximum of 1,122. Moreover, the average number of helpful votes given by a user is 40. Harold (Yelp only provides first names) is the user who gave the most significant number of helpful votes (197,130) and a Yelp community member since 2012.  

On the other hand, the average star rating for all businesses is 3.6. However, Yelp marked approximately 20% of the companies as closed. Finally, the restaurant industry, which is the most significant industry, accounts for 30.5% of all businesses and is followed by the shopping industry, 16.5%. 

---

For a detailed version of this section, [please refer to the full report](https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/Final%20Report.pdf)  
[Go to Table of Contents](#Table-of-Contents)

---

#### Where the Businesses are Located

The businesses spread around Canada and the United States. While almost ¼ (43,693 out of 168,903) of all open businesses placed in Canada, Ontario is the most populous state with 36,627 companies. Quebec and Alberta follow it with the number of companies 10,223 and 8,682, respectively.   

<p align="center">
  <img src="https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/figures/business-locations.JPG" width="800" height="400">
</p>

In the United States, the companies grouped around Arizona-Nevada, Ohio-Wisconsin-Illinois, and North Carolina. However, Arizona and Nevada have 48% of all businesses in the US. 

#### Businesses in the United States

The most populous states in terms of the number of businesses are Arizona and Nevada. Ohio and North Carolina have approximately the same number of companies. Pennsylvania is the last state in the US that has more than 10k businesses. 

 In this project, we mainly focus on the businesses operating in the US's restaurant industry. For this reason, it is essential to select firms in the US. Accordingly, we generated a dummy variable, `in_US`, which indicates whether a business operates in the US. Additionally, we take advantage of another dummy variable, `is_open`, provided by Yelp, which shows whether a company still operates. As a result, the United States has 153,843 businesses (28,633 of those are closed) in the Yelp academic data set.

#### What are the Industries?

Yelp's dataset provides industry information for each business. However, 374 firms do not associate with any industry. For this reason, they are dropped from the study. The remaining 124,836 firms are divided into 22 industries. The biggest industry is the restaurant industry, which is followed by the shopping industry. They almost account for half of the businesses in the US.  

<p align="center">
  <img src="https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/figures/industries.JPG" width="600" height="400">
</p>

#### Businesses in the Restaurant Industry

While the restaurant businesses account for 30.5% of all companies, they are the customer reviews dataset's dominant category. Approximately 63% of all customer reviews (5,055,992 out of 8,021,122 - including closed businesses) belong to the restaurant industry. 

| Business           | Number of Branches |
| :----------------- | :----------------: |
| Subway Restaurants |        609         |
| McDonald's         |        536         |
| Taco Bell          |        294         |
| Burger King        |        273         |
| Wendy's            |        232         |

####  Statistical Summaries of the Restaurant Industry

Businesses in the shopping industry have 3.6-star ratings and 22 reviews on average. The least number of reviews submitted for a shopping business is three, and the highest number of reviews is 3,873. Also, 1.2% and 1% of the restaurant businesses are in the shopping and hotels & travel industries. 

On the other hand, the total number of reviews submitted to the restaurant businesses is 3,487,937, with a mean helpful vote of 1.046. Moreover, 40.5% of all customer reviews have at least one helpful vote, 19.1% have at least one funny vote, and 24.1% have at least one cool vote. The highest number of helpful votes given to a review is 758, the highest number of funny votes and cool votes given to a review are 786 and 321. 

#### Distribution of Business Star Ratings and Review Counts

The histogram of star rating is left-skewed, which means that the number of businesses with a higher star rating than 3.0 is less than the number of companies with a lower star. 

<p align="center">
  <img src="https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/figures/histograms.JPG" width="800" height="300">
</p>

The y-axis of the review counts' histogram is logarithmically scaled, which allows showing a wide range of data compactly. In the figure, the markers on the y-axis increases by multiples of 10. In other words, even though the gaps between the markers on the y-axis are equal on the figure, they grow exponentially.  We see that only a few businesses have more than a thousand customer reviews. 

#### Distribution of Helpful Reviews and Their Relationship with Time

The number of reviews decreases as we increase the number of helpful votes. It may require to have a cut-off point to determine the helpful reviews. Because a customer reviews with one helpful vote should differ from the one with a hundred helpful votes. For this reason, implementing a cut-off may help us better predict helpful reviews. 

<p align="center">
  <img src="https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/figures/histogram-heatmap.JPG" width="800" height="300">
</p>

We see a heatmap of correlation among numerical values in the reviews data. To investigate the relationship between helpful, funny, and cool reviews and time, we generated three measures such as year, month, and day using modular arithmetic. We used an anchor (12/15/2020) to extract all reviews' age in terms of years, months, and days.  

Helpful, funny, and cool reviews are highly positively correlated, leading to multicollinearity. It is the situation when one feature can be predicted by using another feature with great accuracy. For this reason, we will discard funny and cool counts of the reviews from the study. 

#### Distribution of Review Star Ratings and Helpful Reviews

We want to investigate how helpful votes are distributed with respect to star ratings since star ratings can determine how many helpful votes a review will have. The figure shows that the number of helpful reviews and unhelpful reviews are pretty close in almost all categories. Additionally, star rating 
distribution is right-skewed, which means that most of the reviews have 5-stars. 

<p align="center">
  <img src="https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/figures/star-helpful.JPG" width="400" height="300">
</p>

---

[See the full report](https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/Final%20Report.pdf)  
[See the notebook](https://nbviewer.jupyter.org/github/alirifat/yelp_nlp_project/blob/main/02_yelp_eda.ipynb)  
[Go to Table of Contents](#Table-of-Contents)

---



## Data Cleaning and Feature Extraction

As a first step, we dropped all duplicated, 9,073, reviews and removed a null value from the reviews. Thus, the reviews corpus is ready for feature extraction, such as number of sentences, number of words, number of unique words, number of punctuations, number of stop words, number of uppercase and title case words, number of letters, and average word length.  

We explored the data based on the extracted features and designed for the data cleaning process in the second step. Additionally, we extracted another set of features, such as number of photos, URLs, price, time and emoticons. 

Finally, we performed data cleaning processes and vectorized the reviews corpus using TF-IDF matrix. Moreover, we implemented a cut-off for the minimum number of documents, such as 0.03. In other words, the matrix only consists of terms that appears more than 3% of all documents (reviews). 

---

[Go to Table of Contents](#Table-of-Contents)

---

#### Basic Text Features

Most reviews have less than 100 sentences, but it is perfectly normal to have incredibly long customer reviews. However, the longest review consists of more than 250 sentences.  

<p align="center">
  <img src="https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/figures/basic-text-features.JPG" width="800" height="800">
</p>

In English, a sentence has, on average, 20 words, and each word has 4.7 letters. Thus, an average sentence in English has 94 letters. Accordingly, any properly constructed review may not be able to have more than 50 sentences. The longest sentence is wordy and does not contain any material 
information about the place, such as price, time, and photo ([see the notebook](https://nbviewer.jupyter.org/github/alirifat/yelp_nlp_project/blob/main/03_yelp_data_cleaning_feature_engineering.ipynb) for further details). 

On the other hand, we focused on the number of dollar signs, exclamation marks, digits, uppercase words, and average word length. Because each feature may reveal essential information to the customers, such as the number of dollar signs and digits increases in a review, it may contain more information about the price. Also, the number of uppercase words and exclamation marks can express customer emotions and be perceived as helpful by others. 

However, the average length of words is an essential feature to identify anomalies in the corpus. Since it is approximately 5 in English, we can locate any non-English review. In the below figure, we can see the relationship between the average length of words in a review and its language. As the average length of words increases, it is more likely to be written in a different language. 

 <p align="center">
  <img src="https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/figures/correlation-extracted_features.JPG" width="800" height="600">
</p>

Almost all features are highly correlated ([see the notebook](https://nbviewer.jupyter.org/github/alirifat/yelp_nlp_project/blob/main/03_yelp_data_cleaning_feature_engineering.ipynb) for further details), and using all features as the determinants of the target variable will cause multicollinearity. For this reason, we only selected the number of unique words as an independent variable and discarded the remaining features. It was possible to extract only the number of unique words initially; however, this would not provide the best opportunity to get familiar with the corpus. As a result, we used the extracted features for experimental purposes and decided to generate another set of predictive variables. 

---

[Go to Table of Contents](#Table-of-Contents)

---

#### Data Cleaning

As a first step, we used a language detection algorithm to locate English reviews. A small portion of the reviews, 0.002, was in a different language and discarded from the study. We also focused on the businesses that have more than a thousand reviews to have a homogeneous corpus.  

Even though we decreased the average length of words from 800 to 30, there were still non-English characters in the corpus. Those reviews used mixed languages, so that we wanted to remove only the non-English characters. 

Before we moved to the text normalization process, we have to generate the predictive variables, such as the number of photos, URLs, price, time, and emoticons. Since text normalization requires all that information to be removed from the text, it must be done in advance.  

To extract those features, we used Regular Expressions, which finds the pre-defined patterns in the corpus. Since the patterns have to be hard-coded, there may be left-over information after the matching process. However, it should not cause any problem because we covered the most used cases in the patterns. 

| **Feature**  | **Pearson R** | **Significance** |
| ------------ | :-----------: | :--------------: |
| **PHOTO**    |     0.07      |       0.00       |
| **URL**      |     0.05      |       0.00       |
| **PRICE**    |     0.13      |       0.00       |
| **TIME**     |     0.07      |       0.00       |
| **EMOTICON** |     0.16      |       0.00       |

 We, finally, moved to the text normalization process in which we performed the following steps: 

* Replace Chinese and Japanese characters with whitespace
* Whitespace formatting 
* Reduce duplicated letters (Ex. Sooooooooooooooooo → So) 
* Replace spaced words (Ex. A M A Z I N G → AMAZING) 
* Fix contractions (Ex. I’m → I am) 
* Remove hashtags (#) and mentions (@) 
* Remove punctuations 
* Remove digits 
* Lowercase terms 
* Remove stop words 
* Lemmatize and 
* Stemmer 

In the last step, we generated a TF-IDF matrix from the cleaned corpus. However, we implemented a cut-off to remove noise from the data. We used 3% as the lower and 90% as the upper threshold. As a result, the number of terms in the corpus decreased from 100,847 to 243, accounting for 53% of the corpus. 

| **Term** | **Frequency** | **Term** | **Frequency** |
| :------: | :-----------: | :------: | :-----------: |
|   food   |    535,503    |   love   |    195,117    |
|   good   |    455,635    |   wait   |    194,598    |
|  place   |    437,241    | restaur  |    193,228    |
|  great   |    383,019    |   eat    |    187,496    |
|   time   |    314,294    |  friend  |    182,940    |
|  order   |    312,467    |   amaz   |    152,153    |
|  servic  |    308,846    |  delici  |    150,732    |
|   make   |    228,983    |   nice   |    148,213    |
|   back   |    218,896    |   tabl   |    140,939    |
|   vega   |    196,308    |  drink   |    138,937    |

---

[See the full report](https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/Final%20Report.pdf)  
[See the notebook](https://nbviewer.jupyter.org/github/alirifat/yelp_nlp_project/blob/main/03_yelp_data_cleaning_feature_engineering.ipynb)  
[Go to Table of Contents](#Table-of-Contents)

---



## Predict Helpful Reviews

We generated a binary target variable by transforming helpful votes. Any review with five or more helpful votes is recorded as a helpful review, and the remaining ones unhelpful. We used Naïve Bayes and Decision Tree algorithms to obtain initial results. 

We used the TF-IDF matrix to predict helpful reviews; however, the results were not promising ([see the notebook](https://nbviewer.jupyter.org/github/alirifat/yelp_nlp_project/blob/main/04_algorithms_with_default_parameters.ipynb) for a detailed explanation). To check if the matrix works, we predicted star ratings of the reviews with the same matrix. We transformed the star ratings into a binary variable by assigning 1 to reviews with four or higher ratings and 0 to the remaining ones. The TF-IDF matrix works when we predict the star rating; however, it does not predict helpful reviews. For this reason, we used only the extracted features to predict helpful reviews.

After deciding the set of features, we did cross-validation to see the algorithms’ performances. The black bars represent the variation in the validation scores.

 <p align="center">
  <img src="https://github.com/alirifat/yelp_nlp_project/blob/main/Documentation/figures/validation-scores.JPG" width="800" height="600">
</p>

Based on the cross-validation scores, we dropped Naïve Bayes from the study. The differences between the training and the set scores are not significant for __ROC__, but __PR__ and __MCC__ scores differ significantly for tree-based algorithms (excluding __XGBoost__). Accordingly, those algorithms may not be good choices for our data.

---

[Go to Table of Contents](#Table-of-Contents)

---

#### Hyperparameter Optimization

We used GridSearchCv. which goes over every possible combination of  parameters in the parameter space to find the best performing set of  parameters. **kNN** has the most significant benefit from hyperparameter tuning. In the final step, we will train and test the algorithms with the optimized parameters and find the best algorithm to predict helpful reviews. 

|                              | **Score (Default Parameters)** | **Score (Optimized Parameters)** | **Change** |
| ---------------------------- | :----------------------------: | :------------------------------: | :--------: |
| **Logistic Regression**      |             0.975              |              0.976               |  + 0.001   |
| **kNN Classifier**           |             0.897              |              0.950               |  + 0.053   |
| **Decision Tree Classifier** |             0.785              |              0.785               |   0.000    |
| **Random Forest Classifier** |             0.969              |              0.976               |  + 0.007   |
| **Extra Trees Classifier**   |             0.965              |              0.972               |  + 0.007   |
| **XGBoost Classifier**       |             0.980              |              0.981               |  + 0.001   |

---

[Go to Table of Contents](#Table-of-Contents)

---

#### Final Evaluation

 In the final evaluation step, we will assess algorithms based on two criteria:  

1. Confusion matrix 
2. Recall rate (over assigned 0.95 probability) 

We aim to predict helpful reviews based on the extracted features but approaching predicted reviews as a pure classification problem may not benefit any parties. In other words, it will result in only two classes of reviews, such as helpful and unhelpful. However, it does not say anything about how likely one review to be a helpful review. For this reason, we will use predicted probabilities as a second criterion and set the threshold as 0.95. By doing so, we hope to get reviews that are more likely to be helpful review. 

We located the reviews in the test set with the highest helpful votes and checked the algorithms' performance on those reviews. Among 146,513 reviews in the test set, there are only 11 reviews with the top 10 helpful votes. In the first place, we aim to decide if an algorithm can distinguish helpful reviews from unhelpful reviews. Also, we want to check how many of the predicted helpful reviews are among the ones that have the top 10 helpful votes. 

|                              | **Recall Rate** | **Predicted Value** |      | **True Value** |
| ---------------------------- | :-------------: | :-----------------: | :------------: | :--: |
| **Logistic Regression**      |    100.00 %     |         11          |     out of     |  11  |
| **KNN Classifier**           |    100.00 %     |         11          |     out of     |  11  |
| **Decision Tree Classifier** |    100.00 %     |         11          |     out of     |  11  |
| **Random Forest Classifier** |     72.73 %     |          8          |     out of     |  11  |
| **Extra Trees Classifier**   |     81.82 %     |          9          |     out of     |  11  |
| **XGBoost Classifier**       |     72.73 %     |          8          |     out of     |  11  |

We set 0.95 as the threshold for the assigned probabilities so that the business owners (the restaurants in this case) will be more likely to provide customers with the reviews that can attract their attention and present what they are looking for. They can also promote those reviews among the others so that it will be easier for the customers to reach the relevant information about the business and the product(s).

Based on the algorithms' performance in the confusion matrices and the top 5% predicted reviews, we can say that __kNN__ is the most practical algorithm. Even though __XGBoost__ has the best performing results, it has some flaws:

1. __kNN__ hits a 100% recall rate for the top 10 helpful reviews, but __XGBoost__ stays at 72.73%.
2. Even though __kNN__ has a lower recall rate in general, it has the most significant number of correctly predicted helpful reviews.
3. __kNN__ provides a broader pool of helpful reviews for the business owner to hand-pick if necessary.

For those reasons, _we believe that __kNN__ is the best algorithm for our purpose_ in this project. We will provide some examples in the next chapters.

## Conclusion

In this project, we aimed to develop a model that can predict if a freshly posted review will be helpful. If so, the businesses can benefit from it by promoting those reviews and letting the customers enjoy them. 

A helpful review's essential components are the number of customers who voted for the review and the amount of time passed since it was posted. By doing so, we hoped to save the amount of time that would require a review to be recognized as a helpful review and provide those reviews for the customers' convenience in advance.

We started with the corpus to identify the helpful reviews. First, we implemented text cleaning steps to ready the text for vectorization. Later, we used the TF-IDF method to vectorize the text and set the following cut-off points for the minimum, and the maximum number of reviews (documents) for a word (term) has to appear as 3% and 90%, respectively. 

However, the TF-IDF vector was not an efficient way to identify helpful reviews as it was detecting the star rating. For this reason, we employed the features extracted from the reviews, such as the number of photos, the number of price information, the average helpful vote that the writer has, etc. As a result, we improved the model performance.

Later, we did hyperparameter optimization using the extracted features to find the parameter values that explain the data best. Finally, we trained the models using the whole training set and evaluated them with the test set.

To increase the recall rate, we focused on the reviews that have at least 0.95 assigned probability. As a result, we got the highest recall rate by using __XGBoost__ Classifier. However, the best result is acquired by the __kNN__ classifier.

As a result, we think that the best algorithm to identify helpful reviews with the given feature set and the given conditions such as 0.95 probability cut-off is the __kNN__ algorithm. We claim __kNN__ can detect the reviews that have the highest number of helpful votes with great accuracy. Even though its recall rate is lower than some of the other algorithms, it is best to recommend the highest number of helpful reviews.

---

[Go to Table of Contents](#Table-of-Contents)

---

