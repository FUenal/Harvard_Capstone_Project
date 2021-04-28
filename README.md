# Harvard_Capstone_Project

This Repository contains the files for my **HarvardX PH125.9x Data Science: Capstone Project**. 

It includes 2 Capstone projects:

**1. The submission for the MovieLens project.**   


Here, I report the results of the first part of my HarvardX PH125.9x Data Science: Capstone Project. The goal of this exercise was to develop and test Machine Learning Algorithms to predict movie ratings. The data for this project came from the [GroupLens research lab](https://grouplens.org/) who generated their own database with over 20 million ratings for over 27,000 movies by more than 138,000 users. The data used here is a subset of this data provided by the course.
I programmed several regression algorithms to predict movie ratings using the available features of movies, users and genres. After an exploratory data analysis in which I analyzed the structure, composition, completeness of the data set as well as the associations between the features and the outcome variable, I controlled for missing values and applied some data wrangling techniques to prepare the data for the modeling phase. The EDA indicated that movie ratings might be predicted by a) Users, Movies, and Genres, and that the data set represents a so-called sparse data set. Based on my observations during the EDA, I modeled several algorithms to predict the outcome variable and compared the RMSE scoreof each alogorithm to determine the best model (lowest RMSE value). Due to the sparse nature of the data set, I also applied regularization of my final model to account for potentially `biased` effect sizes. Regularization permits to penalize `biased` effect sizes such as large estimates that are formed using small sample sizes and thus optimizes the results.    

**The `Regularized Movie + User + Genre + Movie-Genre` Model achieved an RMSE of 0.8616823.**


**2. The second project: Climate Change Worry among European Citizens.**   


Here, I report the results of the second part of my HarvardX PH125.9x Data Science: Capstone Project. The goal of this full-stack data science project was to download, wrangle, explore, and analyze European Citizens' perceptions of climate change using 10 different Machine Learning (ML) Algorithms (*KNN, naiveBayes, Random Forest, adaBOOST, SVM, Ensemble, ctree, GBM, C5.0, rpart*) while also employing hyperparameter tuning (e.g., *tuneGrid* and *tuneLength*) to some of the models to improve the results. The data for this project came from the 8th round [European Social Survey (2016)](https://www.europeansocialsurvey.org/data/download.html?r=8), a publicly available academic data set. The response variable (outcome) in this project was "climate change worry", i.e., whether or not individuals in more than 30 European countries are worried about climate change. The response to this variable was binary ('yes' vs. 'no') and thus, this project dealt with a classification problem
After a contextual introduction into the topic of climate change perceptions, I explained how the data download, selection, and cleaning was done. Then, after splitting the data into *train* and *test* sets, I proceeded with an Exploratory Data Analysis (EDA) in which I explored the train data set and visualized the initial data exploration gaining some insights for modeling. In the Methods & Analysis Section, I employed some feature engineering beyond the steps undertaken in the initial data selection process (i.e., missing value imputation, removing zero variance features, normalization of data). Afterwards, I employed several ML algorithms and checked to what extent each of the algorithms accurately classifies the response variable.    

Out of the 10 individual ML algorithms tested, the best results was (were) achieved by a **GLM Ensemble Machine Learning Algorithm based on the 10 individual algortihms developed and which achieved and overall Accuracy of 0.7775691.**

