#################################################################################################
###                     Harvard Data Science Project: Climate Change Worry                    ###
###                                     Author: Dr Fatih Uenal                                ###
###                                     Date: April 2021                                      ###
#################################################################################################



#   PLEASE NOTE THAT RUNNING THE CODE PROVIDED HERE MIGHT REQUIRE UP TO 60 MIN OR MORE TO RUN   #
#              DEPENDING ON THE PROCESSING POWER OF YOUR PERSONAL COMPUTER                      #



#################################################################################################
##                              Data Download & Preparation                                    ##
#################################################################################################

## Install/Load required packages
# All of the following packages need to be installed to run this document properly
if(!require(devtools)) install.packages("devtools", repos = "http://cran.us.r-project.org")
library(devtools)
if(!require(lares)) devtools::install_github("laresbernardo/lares")
if(!require(pacman)) install.packages("pacman", repos = "http://cran.us.r-project.org")
library(pacman)

pacman::p_load(essurvey, readr, curl, tidyverse, haven, skimr, C50, class, e1071, randomForest, rpart, rpart.plot, 
               partykit, caret, caretEnsemble, lares, GGally, ggplot2, PerformanceAnalytics, fastmatch,
               Metrics, ipred, mlbench, RANN, highcharter)

options(dplyr.summarise.inform = FALSE) # Suppress summary info


### 3.2 Download

## Download ESS Data
# You need to first register an account with ESS. Then use your email which you used for registration 
# to download the data set using the 'ess' package.

## Download ESS Data

# Set email for access
# set_email("ENTER YOUR EMAIL ADRESS HERE")

## Download all countries round 8
# df8 <- import_rounds(8)

## Load data file
# df8 <- read.csv("ESS8e02.1_F1.csv") # If you download the data from edX

## Download data file from my github
# df8 <- read_csv('https://raw.githubusercontent.com/FUenal/Harvard_Capstone_Project/main/ESS8e02.1_F1.csv')
df8 <- read.csv(curl('https://raw.githubusercontent.com/FUenal/Harvard_Capstone_Project/main/ESS8e02.1_F1.csv'))

### 3.3 Preparation

## First peak into raw data
skim(df8)


## Identify and deal with low and high correlations and zero variance variables ('lares' package)
## Check Overall Data Structure
df_str(df8)
df_str(df8, return = "skimr")


## From here on after the cleaned and smaller subset of the data will be used
## Load cleaned data set
# df <- read.csv("ess8_subsample.csv") # If you download the data from edX

## Download data file from my github
# df <- read_csv('https://raw.githubusercontent.com/FUenal/Harvard_Capstone_Project/main/ess8_subsample.csv')
df <- read.csv(curl('https://raw.githubusercontent.com/FUenal/Harvard_Capstone_Project/main/ess8_subsample.csv'))

## Random sample for testing purposes: You can choose to work with an even smaller subsample to have shorter run times
# df <- sample_n(df, 300)

## First peak into cleaned data
df_str(df)
# df_str(df, return = "skimr")


## Check missing values
highmiss <- missingness(df)
head(highmiss)

# Store columns with NaNs > 50%
highmiss50 <- highmiss %>%
        filter(missingness >= 50)

dropmiss50 <- highmiss50[["variable"]]

# Removing variables with NaNs >50% 
df <- df %>%
        select(-one_of(dropmiss50))


# Recheck missing values
df_str(df)
head(missingness(df))

# Check zero variance columns (same value in each column)
zerovar(df)


# Check Missing Values in Response Variable
sum(is.na(df$wrclmch))

# Drop NAs in response variable
df <- df %>%
        drop_na(wrclmch)

# Binarize response variable
df <- df %>%
        mutate(wrclmch = case_when(wrclmch <= 2 ~ "Not_worried",
                                   wrclmch >= 3 ~ "Worried"),
               wrclmch = factor(wrclmch, levels = c("Not_worried", "Worried")))

# Table response variable counts 
table(df$wrclmch)


## Factor Gender
df <- df %>% 
        mutate(gndr = case_when(gndr <= 1 ~ "Male",
                                gndr >= 2 ~ "Female"),
               gndr = factor(gndr, levels = c("Male", "Female")))

table(df$gndr)


# drop idno 
df <- df[,-1]




#################################################################################################
##                              Exploratory Data Analysis (EDA)                                ##
#################################################################################################

# 4. Exploratory Data Analysis (EDA)

## 4.1. Train - Test Data Split


# Split data
# Validation set will be 40% of data
set.seed(123, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = df$wrclmch, times = 1, p = 0.4, list = FALSE)
trainData <- df[-test_index,]
testData <- df[test_index,]


## 4.2. Frequencies & Distributions

## First peak into partitioned trainData
head(df_str(trainData, return = "skimr"))


# Table response variable counts 
# Train Data
table(trainData$wrclmch) 

# Table response variable counts 
# Test Data
table(testData$wrclmch) 


## How many worried about climate change?

# Using the 'lares' package: Works only in the HTML format
trainData %>% freqs(wrclmch, plot = T, results = F, abc = TRUE)

# Same as above but using ggplot
plot1 <- trainData %>% 
        ggplot(aes(wrclmch)) +
        geom_bar(fill = "steelblue") +
        ggtitle("How many are worried about climate change?") +
        labs(x="Climate Change Worry", y="Count") +
        theme_minimal() +
        ggeasy::easy_center_title()
plot1


## climate worry across countries

# Using the 'lares' package: Works only in the HTML format
trainData %>% distr(wrclmch, cntry, clean = TRUE, abc = FALSE)

# Same as above but using ggplot
plot2 <- ggplot(trainData, aes(wrclmch)) + 
        geom_bar(fill = "steelblue") + 
        labs(title = "Climate Change Worry Across Countries",
             x = "Climate Change Worry") +
        facet_wrap(~cntry, dir = 'h') + 
        theme_minimal() +
        ggtitle("Climate change worry across countries.") +
        ggeasy::easy_center_title()

plot2


## Climate worry across political spectrum
# In politics people sometimes talk of 'left' and 'right'. Using this card, where would you place yourself on this scale, where 0 means the left and 10 means the right

# Using the 'lares' package: Works only in the HTML format
trainData %>% distr(wrclmch, lrscale)

# Same as above but using ggplot and binarized political orientation scale
# plot3 <- trainData %>%
#         mutate(lrscale1 = case_when(lrscale <= 5 ~ "Left",
#                                     lrscale >= 6 ~ "Right"),
#                lrscale1 = factor(lrscale1, levels = c("Left", "Right"))) %>% 
#         drop_na(lrscale1) %>% 
#         ggplot(aes(wrclmch, ..count..)) + 
#         geom_bar(aes(fill = lrscale1), position = "dodge", na.rm = TRUE) +
#         labs(title = "Climate Change Worry Across The Political Spectrum", x = "Climate Change Worry", y = "Count", fill = "Political Orientation") +
#         scale_fill_manual(values = c("#d8b365", "#5ab4ac")) +
#         theme_minimal() 
# plot3 + ggeasy::easy_center_title()



## Climate worry across genders (1: Male // 2: Female)

# Using the 'lares' package: Works only in the HTML format
trainData %>% distr(wrclmch, gndr)

# Same as above but using ggplot
plot4 <- ggplot(trainData, aes(wrclmch, ..count..)) + 
        geom_bar(aes(fill = gndr), position = "dodge") +
        labs(title = "Climate Change Worry Across Genders", x = "Climate Change Worry", y = "Count", fill = "Gender") +
        scale_fill_manual(values = c("#d8b365", "#5ab4ac")) +
        theme_minimal() 
plot4 + ggeasy::easy_center_title()




#################################################################################################
##                                     Methods & Analysis                                      ##
#################################################################################################

## Data Pre-processing

# drop country
trainData <- trainData[,-1]


# Relocate response variable as first column
fmatch("wrclmch", names(trainData))
trainData <- trainData %>% relocate(wrclmch, .before = gndr)


# Store X and Y for later use.
x = trainData[, 2:123]
y = trainData$wrclmch


### Missing Value Imputation

## Impute Missing Values with medianImpute Method
# Create the median imputation model on the training data
preProcess_missingdata_model <- preProcess(trainData, method = "medianImpute")
preProcess_missingdata_model


# Use the imputation model to predict the values of missing data points
trainData <- predict(preProcess_missingdata_model, newdata = trainData)

# Check for NaNs
anyNA(trainData)


### Normalization

## Normalize data
preProcess_range_model <- preProcess(trainData, method='range')
trainData <- predict(preProcess_range_model, newdata = trainData)

# Append the Y variable
trainData$wrclmch <- y

# Show first 10 columns of normalized data
apply(trainData[, 1:10], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})


### Feature Selection 
str(trainData)
## Correlation Analysis
# Highest correlation with dependent variable for processing
top20corr <- corr_var(trainData, # name of data set
                      wrclmch, # name of variable to focus on
                      top = 20, # display top 20 correlations
                      ranks = FALSE, # Boolean. Add ranking numbers // MAKE THIS LINE FALSE FOR THE NEXT STEP TO RUN!
                      plot = FALSE # Don't return plot
) 

print(top20corr)

# SELECT TOP 20 CORRELLATIONS
dftopcorr <- top20corr$variables
# str(dftopcorr)

trainData <- trainData %>% 
        select(all_of(dftopcorr))

wrclmch <- y
trainData <- cbind(trainData, wrclmch)
# str(trainData)


# Relocate response variable as first column
fmatch("wrclmch", names(trainData))
trainData <- trainData %>% relocate(wrclmch, .before = wrdpimp)


## Correlation Viz 1

library(ggplot2)
library(GGally)

ggpairs(trainData[,c(2:11,1)], aes(color=wrclmch, alpha=0.75), lower=list(continuous="smooth"))+ theme_bw()+
        labs(title="Climate Worry Mean")+
        theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))


## Correlation Viz 2

ggcorr(trainData[,c(2:11)], name = "corr", label = TRUE)+
        theme(legend.position="none")+
        labs(title="Climate Worry Mean")+
        theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))

## Correlation Viz 3

library(PerformanceAnalytics)

chart.Correlation(trainData[,c(2:10)],histogram=TRUE, col="grey10", pch=1, main="Climate Worry Mean")


## Pre-processing testData

# * remove the 'country' feature, 
# * relocate the response variable to the very beginning, 
# * store my response variable and my feature separately for later usage, 
# * impute the missing values, 
# * normalize the data, and
# * finally, select the same 20 variables as in the **trainData** set. 

# drop country
testData <- testData[,-1]

# Relocate response variable as first column
fmatch("wrclmch", names(testData))
testData <- testData %>% relocate(wrclmch, .before = gndr)

# Store X and Y for later use.
x1 = testData[, 2:123]
y1 = testData$wrclmch

## Prepare the test data set and predict
# Step 1: Impute missing values 
testData2 <- predict(preProcess_missingdata_model, testData)  

# View
# head(testData2[, 1:10])
anyNA(testData2)

## Normalize data
# Step 1: Transform the features to range between 0 and 1
testData3 <- predict(preProcess_range_model, testData2)

## Highest correlation with dependent variable
## SELECT TOP 20 CORRELLATIONS FROM TRAINDATA
testData3 <- testData3 %>% 
        select(all_of(dftopcorr))

wrclmch <- y1
testData3 <- cbind(testData3, wrclmch)

# str(testData3)

# Relocate response variable as first column
fmatch("wrclmch", names(testData3))
testData3 <- testData3 %>% relocate(wrclmch, .before = wrdpimp)


#################################################################################################
##                                      Modeling                                               ##
#################################################################################################

## Modeling

# * C5.0 
# 
# * KNN 
# 
# * SVM 
# 
# * naiveBayes 
# 
# * rpart 
# 
# * ctree 
# 
# * Random Forest 
# 
# * adaBOOST 
# 
# * GBM 
# 
# * Ensemble 


### C5.0  

# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

library(C50)
learn_c50 <- C5.0(trainData[,-1],trainData$wrclmch)
pre_c50 <- predict(learn_c50, testData3[,-1])
cm_c50 <- confusionMatrix(pre_c50, testData3$wrclmch)
cm_c50


## C5.0 Tune 
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

acc_test <- numeric()
accuracy1 <- NULL; accuracy2 <- NULL

for(i in 1:50){
        learn_imp_c50 <- C5.0(trainData[,-1],trainData$wrclmch,trials = i)      
        p_c50 <- predict(learn_imp_c50, testData3[,-1]) 
        accuracy1 <- confusionMatrix(p_c50, testData3$wrclmch)
        accuracy2[i] <- accuracy1$overall[1]
}

acc <- data.frame(t= seq(1,50), cnt = accuracy2)

opt_t <- subset(acc, cnt==max(cnt))[1,]
sub <- paste("Optimal number of trials is", opt_t$t, "(accuracy :", opt_t$cnt,") in C5.0")

# Plot
library(highcharter)

hchart(acc, 'line', hcaes(t, cnt)) %>%
        hc_title(text = "Accuracy With Varying Trials (C5.0)") %>%
        hc_subtitle(text = sub) %>%
        hc_add_theme(hc_theme_google()) %>%
        hc_xAxis(title = list(text = "Number of Trials")) %>%
        hc_yAxis(title = list(text = "Accuracy"))


### Apply optimal trials to show best predict performance in C5.0
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

learn_imp_c50 <- C5.0(trainData[,-1],trainData$wrclmch,trials=opt_t$t)    
pre_imp_c50 <- predict(learn_imp_c50, testData3[,-1])
cm_imp_c50 <- confusionMatrix(pre_imp_c50, testData3$wrclmch)
cm_imp_c50



## KNN - Tune 
library(class)

# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

acc_test <- numeric() 

for(i in 1:30){
        predict <- knn(train=trainData[,-1], test=testData3[,-1], cl=trainData[,1], k=i, prob=T)
        acc_test <- c(acc_test,mean(predict==testData3[,1]))
}

acc <- data.frame(k= seq(1,30), cnt = acc_test)

opt_k <- subset(acc, cnt==max(cnt))[1,]
sub <- paste("Optimal number of k is", opt_k$k, "(accuracy :", opt_k$cnt,") in KNN")

# Plot
hchart(acc, 'line', hcaes(k, cnt)) %>%
        hc_title(text = "Accuracy With Varying K (KNN)") %>%
        hc_subtitle(text = sub) %>%
        hc_add_theme(hc_theme_google()) %>%
        hc_xAxis(title = list(text = "Number of Neighbors(k)")) %>%
        hc_yAxis(title = list(text = "Accuracy"))


## Apply optimal K to show best predict performance in KNN
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

pre_knn <- knn(train = trainData[,-1], test = testData3[,-1], cl = trainData[,1], k=opt_k$k, prob=T)
cm_knn  <- confusionMatrix(pre_knn, testData3$wrclmch)
cm_knn



## naiveBayes
library(e1071)

# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

acc_test <- numeric()
accuracy1 <- NULL; accuracy2 <- NULL

for(i in 1:30){
        learn_imp_nb <- naiveBayes(trainData[,-1], trainData$wrclmch, laplace=i)    
        p_nb <- predict(learn_imp_nb, testData3[,-1]) 
        accuracy1 <- confusionMatrix(p_nb, testData3$wrclmch)
        accuracy2[i] <- accuracy1$overall[1]
}

acc <- data.frame(l= seq(1,30), cnt = accuracy2)

opt_l <- subset(acc, cnt==max(cnt))[1,]
sub <- paste("Optimal number of laplace is", opt_l$l, "(accuracy :", opt_l$cnt,") in naiveBayes")

# Plot
hchart(acc, 'line', hcaes(l, cnt)) %>%
        hc_title(text = "Accuracy With Varying Laplace (naiveBayes)") %>%
        hc_subtitle(text = sub) %>%
        hc_add_theme(hc_theme_google()) %>%
        hc_xAxis(title = list(text = "Number of Laplace")) %>%
        hc_yAxis(title = list(text = "Accuracy"))


# naiveBayes without laplace
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

learn_nb <- naiveBayes(trainData[,-1], trainData$wrclmch)
pre_nb <- predict(learn_nb, testData3[,-1])
cm_nb <- confusionMatrix(pre_nb, testData3$wrclmch)        
cm_nb



## SVM 
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

## SVM 
learn_svm <- svm(wrclmch~., data=trainData)
pre_svm <- predict(learn_svm, testData3[,-1])
cm_svm <- confusionMatrix(pre_svm, testData3$wrclmch)
cm_svm


## SVM -Tune 
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

### Choose ‘gamma, cost’ which shows best predict performance in SVM
gamma <- seq(0,0.1,0.005)
cost <- 2^(0:5)
parms <- expand.grid(cost=cost, gamma=gamma)    ## 231

acc_test <- numeric()
accuracy1 <- NULL; accuracy2 <- NULL

for(i in 1:NROW(parms)){        
        learn_svm <- svm(wrclmch~., data=trainData, gamma=parms$gamma[i], cost=parms$cost[i])
        pre_svm <- predict(learn_svm, testData3[,-1])
        accuracy1 <- confusionMatrix(pre_svm, testData3$wrclmch)
        accuracy2[i] <- accuracy1$overall[1]
}

acc <- data.frame(p= seq(1,NROW(parms)), cnt = accuracy2)

opt_p <- subset(acc, cnt==max(cnt))[1,]
sub <- paste("Optimal number of parameter is", opt_p$p, "(accuracy :", opt_p$cnt,") in SVM")

# Plot
hchart(acc, 'line', hcaes(p, cnt)) %>%
        hc_title(text = "Accuracy With Varying Parameters (SVM)") %>%
        hc_subtitle(text = sub) %>%
        hc_add_theme(hc_theme_google()) %>%
        hc_xAxis(title = list(text = "Number of Parameters")) %>%
        hc_yAxis(title = list(text = "Accuracy"))

# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

### Apply optimal parameters(gamma, cost) to show best predict performance in SVM

learn_imp_svm <- svm(wrclmch~., data=trainData, cost=parms$cost[opt_p$p], gamma=parms$gamma[opt_p$p])
pre_imp_svm <- predict(learn_imp_svm, testData3[,-1])
cm_imp_svm <- confusionMatrix(pre_imp_svm, testData3$wrclmch)
cm_imp_svm




## Hyperparameter tuning to optimize the model for better performance
# trainControl()

# Define the training control
fitControl <- trainControl(
        method = 'cv',                   # k-fold cross validation
        number = 5,                      # number of folds
        savePredictions = 'final',       # saves predictions for optimal tuning parameter
        classProbs = T,                  # should class probabilities be returned
        summaryFunction=twoClassSummary  # results summary function
) 



### rpart 
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

# Train the model using rpart
model_rpart = train(wrclmch ~ ., data=trainData, method='rpart', tuneLength=2, trControl = fitControl)
# model_rpart

# Predict on testData and Compute the confusion matrix
predicted5 <- predict(model_rpart, testData3)
confmat_rpart <- confusionMatrix(reference = testData3$wrclmch, data = predicted5)
confmat_rpart



## Training ctree
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

# Train the model using ctree
model_ctree = train(wrclmch ~ ., data=trainData, method='ctree', tuneLength=2, trControl = fitControl)
# model_ctree

# Predict on testData and Compute the confusion matrix
predicted6 <- predict(model_ctree, testData3)
confmat_ctree <- confusionMatrix(reference = testData3$wrclmch, data = predicted6)
confmat_ctree


## Training Random Forest
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

# Train the model using rf
model_rf = train(wrclmch ~ ., data=trainData, method='rf', tuneLength=2, trControl = fitControl)
# model_rf

# Predict on testData and Compute the confusion matrix
predicted7 <- predict(model_rf, testData3)
confmat_rf <- confusionMatrix(reference = testData3$wrclmch, data = predicted7)
confmat_rf



## Training GBM
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

# Train the model using GBM
model_gbm = train(wrclmch ~ ., data=trainData, method='gbm', tuneLength=2, trControl = fitControl)
# model_gbm

# Predict on testData and Compute the confusion matrix
predicted8 <- predict(model_gbm, testData3)
confmat_gbm <- confusionMatrix(reference = testData3$wrclmch, data = predicted8)
confmat_gbm




##Training Adaboost
# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

# Train the model using Adaboost
model_adaboost = train(wrclmch ~ ., data=trainData, method='adaboost', tuneLength=2, trControl = fitControl)
# model_adaboost

# Predict on testData and Compute the confusion matrix
predicted9 <- predict(model_adaboost, testData3)
confmat_ada <- confusionMatrix(reference = testData3$wrclmch, data = predicted9)
confmat_ada



### Ensembling the predictions
library(caretEnsemble)

# Set the seed for reproducibility
set.seed(123, sample.kind="Rounding")

# Stacking Algorithms - Run multiple algorithms in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=5, 
                             repeats=2,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

algorithmList <- c('knn', 'naive_bayes', 'C5.0', 'rpart', 'ctree',  'rf', 'gbm', 'adaboost', 'svmRadial')


models <- caretList(wrclmch ~ ., data=trainData, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)

# Box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)


### Combine the predictions of multiple models to form a final prediction
# Create the trainControl
set.seed(123, sample.kind="Rounding")
stackControl <- trainControl(method="repeatedcv", 
                             number=5, 
                             repeats=2,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
stack.glm

# Predict on testData and Compute the confusion matrix
predicted12 <- predict(stack.glm, newdata=testData3)
confmat_glm <- confusionMatrix(reference = testData3$wrclmch, data = predicted12, positive = 'Worried')



## Model Accuracy Comparison    

## Visualize to compare the accuracy of all methods
col <- c("#ed3b3b", "#0099ff")
par(mfrow=c(4,4))
fourfoldplot(cm_c50$table, color = col, conf.level = 0, margin = 1, main=paste("C5.0 (",round(cm_c50$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_imp_c50$table, color = col, conf.level = 0, margin = 1, main=paste("Tune C5.0 (",round(cm_imp_c50$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_knn$table, color = col, conf.level = 0, margin = 1, main=paste("KNN (",round(cm_knn$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_nb$table, color = col, conf.level = 0, margin = 1, main=paste("NaiveBayes (",round(cm_nb$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_svm$table, color = col, conf.level = 0, margin = 1, main=paste("SVM (",round(cm_svm$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_imp_svm$table, color = col, conf.level = 0, margin = 1, main=paste("Tune SVM (",round(cm_imp_svm$overall[1]*100),"%)",sep=""))
fourfoldplot(confmat_rpart$table, color = col, conf.level = 0, margin = 1, main=paste("RPart (",round(confmat_rpart$overall[1]*100),"%)",sep=""))
fourfoldplot(confmat_ctree$table, color = col, conf.level = 0, margin = 1, main=paste("CTree (",round(confmat_ctree$overall[1]*100),"%)",sep=""))
fourfoldplot(confmat_rf$table, color = col, conf.level = 0, margin = 1, main=paste("RandomForest (",round(confmat_rf$overall[1]*100),"%)",sep=""))
fourfoldplot(confmat_gbm$table, color = col, conf.level = 0, margin = 1, main=paste("GBM (",round(confmat_gbm$overall[1]*100),"%)",sep=""))
fourfoldplot(confmat_ada$table, color = col, conf.level = 0, margin = 1, main=paste("AdaBoost (",round(confmat_ada$overall[1]*100),"%)",sep=""))
fourfoldplot(confmat_glm$table, color = col, conf.level = 0, margin = 1, main=paste("Ensemble GLM (",round(confmat_glm$overall[1]*100),"%)",sep=""))


## Best Model vis-a-vis accuracy

opt_predict <- c(cm_c50$overall[1], cm_imp_c50$overall[1], cm_knn$overall[1], cm_nb$overall[1], cm_svm$overall[1], cm_imp_svm$overall[1], 
                 confmat_rpart$overall[1], confmat_ctree$overall[1], confmat_rf$overall[1], confmat_gbm$overall[1], confmat_ada$overall[1], confmat_glm$overall[1])
names(opt_predict) <- c('c50', 'imp_c50', 'knn', 'nb', 'svm' , 'imp_svm', 'rpart', 'ctree', 'rf', 'gbm', 'ada', 'glm')
best_predict_model <- subset(opt_predict, opt_predict==max(opt_predict))




#################################################################################################
##                                      Results                                                ##
#################################################################################################

# Box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

# Accuracy best model
best_predict_model

# Ensemble results
stack.glm



# Variable importance

varimp_rf <- varImp(model_rf)
plot(varimp_rf, main="Variable Importance with Random Forest")


#################################################################################################
##                                      Conclusion                                             ##
#################################################################################################


best_predict_model

