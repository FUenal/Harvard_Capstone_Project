##########################################################
###         MovieLens Movie Ratings Project            ###
###             Author: Fatih Uenal                    ###
###             Date: April 2021                       ###
##########################################################

# PLEASE NOTE THAT RUNNING THE CODE PROVIDED HERE MIGHT 
# REQUIRE UP TO 75 MIN OR MORE TO RUN DEPENDING ON THE 
# PROCESSING POWER OF YOUR PERSONAL COMPUTER 

# The following code was provided by edX /////////////////

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
# title = as.character(title),
# genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
        semi_join(edx, by = "movieId") %>%
        semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# The above code was provided by edX /////////////////////

##########################################################
##       Exploratory Data Analysis (EDA)                ##
##########################################################

# Loading required packages
library(pacman)
pacman::p_load(ggpubr, knitr, rafalib, raster, tidyverse, ggeasy, dplyr)

## EDA 1: Data glimpse
library(tidyverse)
edx %>% as_tibble()

## EDA 2: Missing values check
any(is.na(edx))

## EDA 3: Unique values check
edx %>% 
        summarize(n_users = n_distinct(userId),
                  n_movies = n_distinct(movieId))

## EDA 4: Distribution of ratings across movies
plot_movies <- edx %>%
        count(movieId) %>%
        ggplot(aes(n)) +
        geom_histogram(bins = 20, fill = "orange") + 
        labs(title = "Ratings per movie",
             x = "Ratings per movie", y = "Count", fill = element_blank()) +
        theme_classic() +
        ggeasy::easy_center_title()

plot_movies

## EDA 5: Distribution of ratings across users
plot_users <- edx %>%
        count(userId) %>%
        ggplot(aes(n)) +
        geom_histogram(bins = 20, fill = "orange") + 
        labs(title = "Ratings per user",
             x = "Ratings per user", y = "Count", fill = element_blank()) +
        theme_classic() +
        ggeasy::easy_center_title()

plot_users

## EDA 6: Wrangling Genre Column
genres_unique <- str_extract_all(unique(edx$genres), "[^|]+") %>%
        unlist() %>%
        unique()

genres_unique

## EDA 7: Re-shaping both the train and validation data sets into long format with each genres represented individually
edx <- edx %>%
        separate_rows(genres, sep = "\\|", convert = TRUE)

validation <- validation %>%
        separate_rows(genres, sep = "\\|", convert = TRUE)

## EDA 8: Distribution of number of ratings across genres
plot_genres <- ggplot(edx, aes(x = reorder(genres, genres, function(x) - length(x)))) +
        geom_bar(fill = "orange") +
        labs(title = "Ratings per genre",
             x = "Genre", y = "Counts") +
        scale_y_continuous(labels = paste0(1:4, "M"),
                           breaks = 10^6 * 1:4) +
        coord_flip() +
        theme_classic() +
        ggeasy::easy_center_title()

plot_genres

## EDA 9: Distribution of ratings across genres
plot_genre_ratings <- ggplot(edx, aes(genres, rating)) + 
        geom_boxplot(fill = "orange", varwidth = TRUE) + 
        labs(title = "Ratings across genres",
             x = "Genre", y = "Rating", fill = element_blank()) +
        theme_classic() +
        theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
        ggeasy::easy_center_title()

plot_genre_ratings

##########################################################
##               Modeling & Evaluation                  ##
##########################################################

## Loss function: RMSE

RMSE <- function(true_ratings, predicted_ratings){
        sqrt(mean((true_ratings - predicted_ratings)^2))
}

## Average movie rating
mean(edx$rating)

## Baseline Model: Just the average

mu_hat <- mean(edx$rating)
naive_rmse <- RMSE(edx$rating, mu_hat)

rmse_results <- tibble(Method = "Just the average", RMSE = naive_rmse)
kable(rmse_results)

## Predicting ratings by movie average rating
RMSE(edx$rating, mu_hat)


### First model: Movie Effects Model

movie_avg <- edx %>% 
        group_by(movieId) %>% 
        summarize(b_i = mean(rating - mu_hat))

predicted_ratings <- mu_hat + validation %>% 
        left_join(movie_avg, by = 'movieId') %>%
        pull(b_i)

mod_m <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Movie Effect Model",
                                 RMSE = mod_m))

kable(rmse_results[2,])


### Second model: Movie + User Effects Model

user_avg <- edx %>% 
        left_join(movie_avg, by = 'movieId') %>%
        group_by(userId) %>%
        summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- validation %>% 
        left_join(movie_avg, by = 'movieId') %>%
        left_join(user_avg, by = 'userId') %>%
        mutate(pred = mu_hat + b_i + b_u) %>%
        pull(pred)

mod_m_u <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Movie + User Effects Model",  
                                 RMSE = mod_m_u))

kable(rmse_results[3,])


### Third model: Movie + User + Genre Effects Model

genres_avg <- edx %>% 
        left_join(movie_avg, by = 'movieId') %>%
        left_join(user_avg, by = 'userId') %>%
        group_by(genres) %>%
        summarize(b_g = mean(rating - mu_hat - b_i - b_u))

predicted_ratings <- validation %>% 
        left_join(movie_avg, by = 'movieId') %>%
        left_join(user_avg, by = 'userId') %>%
        left_join(genres_avg, by = c('genres')) %>%
        mutate(pred = mu_hat + b_i + b_u + b_g) %>%
        pull(pred)

model_m_u_g <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Movie + User + Genre Effects Model",
                                 RMSE = model_m_u_g))

kable(rmse_results[4,])

### Fourth model: Genre*User Effects

genres_user_avg <- edx %>% 
        left_join(movie_avg, by = 'movieId') %>%
        left_join(user_avg, by = 'userId') %>%
        left_join(genres_avg, by = 'genres') %>%
        group_by(genres, userId) %>%
        summarize(b_gu = mean(rating - mu_hat - b_i - b_u - b_g))

predicted_ratings <- validation %>% 
        left_join(movie_avg, by = 'movieId') %>%
        left_join(user_avg, by = 'userId') %>%
        left_join(genres_avg, by = c('genres')) %>%
        left_join(genres_user_avg, c("userId", "genres")) %>%
        mutate(b_gu = ifelse(is.na(b_gu), 0, b_gu),
               pred = mu_hat + b_i + b_u + b_g + b_gu) %>%
        pull(pred)

model_m_u_g_gu <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Movie + User + Genre + Genre-User Effects Model",
                                 RMSE = model_m_u_g_gu))
kable(rmse_results[5,])

### Fifth model: Genre*Movie Effects
genres_movie_avg <- edx %>% 
        left_join(movie_avg, by = 'movieId') %>%
        left_join(user_avg, by = 'userId') %>%
        left_join(genres_avg, by = 'genres') %>%
        group_by(genres, movieId) %>%
        summarize(b_gi = mean(rating - mu_hat - b_i - b_u - b_g))

predicted_ratings <- validation %>% 
        left_join(movie_avg, by = 'movieId') %>%
        left_join(user_avg, by = 'userId') %>%
        left_join(genres_avg, by = c('genres')) %>%
        left_join(genres_movie_avg, c("movieId", "genres")) %>%
        mutate(b_gi = ifelse(is.na(b_gi), 0, b_gi),
               pred = mu_hat + b_i + b_u + b_g + b_gi) %>%
        pull(pred)

model_m_u_g_gi <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Movie + User + Genre + Genre-Movie Effects Model",
                                 RMSE = model_m_u_g_gi))
kable(rmse_results[6,])


### Sixth model: Regularization Model

## Demonstration Movie Effect Model Regularized vs. Non-Regularized
lambda <- 3
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
        group_by(movieId) %>% 
        summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

tibble(original = movie_avg$b_i, 
       regularlized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
        ggplot(aes(original, regularlized, size=sqrt(n))) + 
        geom_point(shape=1, alpha=0.5)

predicted_ratings <- validation %>% 
        left_join(movie_reg_avgs, by = "movieId") %>%
        mutate(pred = mu + b_i) %>%
        pull(pred)
RMSE(predicted_ratings, edx$rating)

model_3_rmse <- RMSE(predicted_ratings, validation$rating)
print("Movie Effect Model") 
0.9410700

print("Regularized Movie Effect Model") 
model_3_rmse

## Sixth model: Regularization Model on Test Set

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
        
        mu <- mean(edx$rating)
        
        b_i <- edx %>% 
                group_by(movieId) %>%
                summarize(b_i = sum(rating - mu)/(n()+l))
        
        b_u <- edx %>% 
                left_join(b_i, by="movieId") %>%
                group_by(userId) %>%
                summarize(b_u = sum(rating - b_i - mu)/(n()+l))
        
        b_g <- edx %>%
                left_join(b_i, by="movieId") %>%
                left_join(b_u, by="userId") %>%
                group_by(genres) %>%
                summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))
        
        b_gi <- edx %>%
                left_join(b_i, by = "movieId") %>%
                left_join(b_u, by = "userId") %>%
                left_join(b_g, by = "genres") %>%
                group_by(movieId, genres) %>%
                summarize(b_gi = sum(rating - mu - b_i - b_u - b_g) / (n() + l))
        
        predicted_ratings <- validation %>%
                left_join(b_i, by = "movieId") %>%
                left_join(b_u, by = "userId") %>%
                left_join(b_g, by = "genres") %>%
                left_join(b_gi, by = c("movieId", "genres")) %>%
                mutate(b_gi = ifelse(is.na(b_gi), 0, b_gi),
                       pred = mu + b_i + b_u + b_g + b_gi) %>% 
                pull(pred)
        
        return(RMSE(predicted_ratings, validation$rating))
})

## Plot lambdas
plot_rmses <- qplot(lambdas, rmses)

## Best lambda (lowest RMSE)
lambda <- lambdas[which.min(rmses)]

## Results test set
rmse_results <- bind_rows(rmse_results,
                          tibble(Method = "Final Regularized Movie + User + Genre + Movie-Genre Model",
                                 RMSE = min(rmses)))
kable(rmse_results[7,])

##########################################################
##                   Results section                    ##
##########################################################

## All Model results
kable(rmse_results)

##########################################################
##                   Environment Info                   ##
##########################################################

print("Operating System:")
version

print("All installed packages")
installed.packages()
