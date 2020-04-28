################################
# Install packages and libraries
################################

install.packages("tidyverse")
install.packages("caret")
install.packages("rmarkdown")
install.packages('tinytex')
tinytex::install_tinytex()

library(rmarkdown)
library(ggplot2)
library(lubridate)
library(tidyverse)

################################
# Create edx set, validation set
################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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

################################
# Data exploration
################################

# Structure of the dataset
str(edx)

# First 6 rows and header 
head(edx)

# Summary of statitics
summary(edx)

# Number of ratings given
nrow(edx)

# Total number of users giving ratings
n_distinct(edx$userId)

# Plot of user bias
edx%>%
count(userId) %>%
ggplot(aes(n)) +
geom_histogram(bins= 30, color = "white") + 
scale_x_log10() + 
ggtitle("Number of ratings by users")

# Total number of movies rated
n_distinct(edx$movieId)

# Plot of movie bias
edx%>%
count(movieId) %>%
ggplot(aes(n)) +
geom_histogram(bins= 30, color = "white") + 
scale_x_log10() + 
ggtitle("Number of ratings for movies")

# Number of movies rated for 10 times or less
edx%>%count(movieId)%>%filter(n<=10)

################################
# Data Analysis - Modelling Approach
################################

# Define RMSE function
RMSE <- function(true_ratings,predicted_ratings){
sqrt(mean((true_ratings-predicted_ratings)^2))}

### Simplest model - using mean rating only ###

# Mean of ratings
mu <- mean(edx$rating) 
mu

# Calculate RMSE of this simplest model
simplest_rmse <- RMSE(validation$rating, mu) 
simplest_rmse

# Print the result in a table
rmse_results <- tibble(method = "Using mean rating", RMSE = simplest_rmse)
rmse_results %>% knitr::kable()

### Model incorporating the movie effects ###

# Calculate the estimates of b_i
movie_avgs <- edx %>% 
group_by(movieId) %>% 
summarize(b_i = mean(rating - mu))

# Plot the estimates of b_i
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 20, data = ., color = I("white"))

# Predict ratings using the test dataset 
predicted_ratings_m <- mu + validation %>% 
left_join(movie_avgs, by='movieId') %>%
pull(b_i)

# Calculate the RMSE
movie_effects_rmse <- RMSE(validation$rating, predicted_ratings_m)

# Print the result in a table
rmse_results <- bind_rows(rmse_results,
tibble(method="Movie Effect Model",  
RMSE = movie_effects_rmse))
rmse_results %>% knitr::kable()

### Model incorporating both user and movie effects ###

# Plot of average rating for user u
edx %>% 
group_by(userId) %>% 
summarize(b_u = mean(rating - mu)) %>% 
ggplot(aes(b_u)) + 
geom_histogram(bins = 20, color = "white")

# Calculate the estimates of b_u
user_avgs <- edx %>% 
left_join(movie_avgs, by='movieId') %>%
group_by(userId) %>%
summarize(b_u = mean(rating - mu - b_i))

# Predict ratings using the test dataset 
predicted_ratings_m_u <- validation %>% 
left_join(movie_avgs, by='movieId') %>%
left_join(user_avgs, by='userId') %>%
mutate(prediction = mu + b_i + b_u) %>%
pull(prediction)

# Calculate the RMSE
user_movie_effects_rmse <- RMSE(validation$rating, predicted_ratings_m_u)

# Print results in a table
rmse_results <- bind_rows(rmse_results,
tibble(method="Movie and User Effects Model",  
RMSE = user_movie_effects_rmse))
rmse_results %>% knitr::kable()

### Regularization Model ###

# Use cross validation to find a lambda that minimize RMSE
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

predicted_ratings_reg <- 
edx %>% 
left_join(b_i, by = "movieId") %>%
left_join(b_u, by = "userId") %>%
mutate(prediction = mu + b_i + b_u) %>%
pull(prediction)

return(RMSE(validation$rating, predicted_ratings_reg))
})

# Plot RMSE results against lambdas
qplot(lambdas, rmses) 

# Print the lambda that results in the lowest RMSE
lambda <- lambdas[which.min(rmses)]
lambda

# Calculate the new RMSE using the optimal lambda and print results in a table
rmse_results <- bind_rows(rmse_results,
tibble(method="Regularization Model",  
RMSE = min(rmses)))
rmse_results %>% knitr::kable()

# Summarize the RMSE results for each model discussed above

rmse_results %>% knitr::kable()
