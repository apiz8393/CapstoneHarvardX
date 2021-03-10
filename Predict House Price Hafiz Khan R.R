
# MovieLens Project (R Script) by Hafiz Khan
# 2/14/2020

######################### Data Download and Cleaning #########################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(gam)

# House Price Prediction dataset:
# https://www.kaggle.com/shree1992/housedata

############################# Data Download #############################

url   <- "https://raw.githubusercontent.com/apiz8393/CapstoneHarvardX/main/house%20price%20dataset.csv"
house <- read_csv(url)


############################ Data Exploration ############################

# Show internal structure of a R object
str(house)  

# Distribution of house price to see any potential error
boxplot(house$price, main="Original House Price Boxplot")


# Remove the 2 highest house price dataset since they does not seems to have correct house price with the criteria (error)
house[which.max(house$price),]
house <- house %>% filter(price<max(price))
house[which.max(house$price),]
house <- house %>% filter(price<max(price))

# Distribution of house price to see any potential error
boxplot(house$price, main="Cleaned House Price Boxplot")

house %>% ggplot(aes(bedrooms))  + geom_histogram(bins = 10, color = "black") + ggtitle("Bedrooms Distribution")
house %>% ggplot(aes(bathrooms)) + geom_histogram(bins = 10, color = "black") + ggtitle("Bathrooms Distribution")
house %>% ggplot(aes(sqft_living)) + geom_histogram(bins = 10, color = "black") + ggtitle("Living Sq-ft Distribution")
house %>% ggplot(aes(yr_renovated)) + geom_histogram(bins = 5, color = "black") + ggtitle("Year Renovated Distribution")


############################# Data Cleaning #############################
# House price summary
summary(house$price)
# Calculate median house price of the dataset
median_house <- as.numeric(summary(house$price)[3])
# Create 'price_fac' variable that has 1 if house price greater than median and 0 otherwise
house$price_fac = as.factor(ifelse(house$price>median_house,1,0))
# Remove unnecessary/high corelation/low variability variable to improve processing time
house <- subset(house,select=-c(price,country,yr_renovated,street,date,statezip))


############################# Data Partition #############################

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index  <- createDataPartition(y = house$price_fac, times = 1, p = 0.1, list = FALSE)
train_house <- house[-test_index,]
test_house  <- house[test_index,]


######################## Model Development ##############################

# Model 1: GLM model
# train glm model
train_glm             <- train(price_fac ~ ., method="glm", data=train_house)
pred_glm              <- predict(train_glm, test_house, "raw")
# calculate accuracy using glm method and store in data frame
accuracy_glm          <- confusionMatrix(pred_glm,factor(test_house$price_fac))$overall[["Accuracy"]]
accuracy_results      <- data_frame(method = "GLM Method", Accuracy = accuracy_glm)
accuracy_glm          <- data_frame(method = "GLM Method", Accuracy = accuracy_glm)
accuracy_glm %>% knitr::kable()


# Model 2: KNN model
modelLookup("knn")
# 5 fold cross-validation tuning
control               <- trainControl(method = "cv", number = 5, p = .9)
# train knn model by using tune grid from 5 to 31 (odd number)
train_knn             <- train(price_fac ~ ., method = "knn", data = train_house, tuneGrid = data.frame(k = seq(5,19,2)), trControl = control)
# plot parameter that provides the best accuracy
ggplot(train_knn, highlight = TRUE)
train_knn$bestTune
pred_knn              <- predict(train_knn, test_house, type="raw")
accuracy_knn          <- confusionMatrix(pred_knn, factor(test_house$price_fac))$overall[["Accuracy"]] 
accuracy_results      <- bind_rows(accuracy_results,data_frame(method="KNN Method", Accuracy = accuracy_knn))
accuracy_knn          <- data_frame(method = "KNN Method", Accuracy = accuracy_knn)
accuracy_knn %>% knitr::kable()


# Model 3: GamLoess model
modelLookup("gamLoess")
# Define grid. Using length of 3 due to limit capacity of my laptop
grid                  <- expand.grid(span = seq(0.15, 0.65, len = 3), degree = 1)
train_loess           <- train(price_fac ~ ., method = "gamLoess", tuneGrid=grid,data = train_house)
# plot parameter that provides best accuracy
ggplot(train_loess, highlight = TRUE)
pred_gamLoess         <- predict(train_loess, test_house)
accuracy_gamLoess     <- confusionMatrix(data = predict(train_loess, test_house), reference = test_house$price_fac)$overall["Accuracy"]
accuracy_results      <- bind_rows(accuracy_results,data_frame(method="GamLoess Method", Accuracy = accuracy_gamLoess))
accuracy_gamLoess     <- data_frame(method = "GamLoess Method", Accuracy = accuracy_gamLoess)
accuracy_gamLoess %>% knitr::kable()


# Model 4: Classification and Regression Trees (CART) model
modelLookup("rpart") 
# train rpart model with selected parameters
train_rpart           <- train(price_fac ~ ., data = train_house, method = "rpart",tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)))
# plot parameter that provides the best accuracy
ggplot(train_rpart, highlight = TRUE)
train_rpart$bestTune
pred_rpart            <- predict(train_rpart, test_house, type="raw")
accuracy_rpart        <- confusionMatrix(pred_rpart, factor(test_house$price_fac))$overall[["Accuracy"]]
accuracy_results      <- bind_rows(accuracy_results,data_frame(method="CART Method", Accuracy = accuracy_rpart))
accuracy_rpart        <- data_frame(method = "CART Method", Accuracy = accuracy_rpart)
accuracy_rpart %>% knitr::kable()


# Model 5: Ensemble
# Pick top 3 method that provides highest accuracy and ensemble them together
combine               <- as.numeric(as.character(pred_rpart)) + as.numeric(as.character(pred_gamLoess)) +  as.numeric(as.character(pred_glm))
# If there are 2 out of the 3 top models predict 1, it will assign 1 to it
pred_ensemble         <- factor(ifelse(combine>=2,1,0))
accuracy_ensemble     <- confusionMatrix(pred_ensemble, factor(test_house$price_fac))$overall[["Accuracy"]]
accuracy_results      <- bind_rows(accuracy_results,data_frame(method="Ensemble Method", Accuracy = accuracy_ensemble))
accuracy_ensemble     <- data_frame(method = "Ensemble Method", Accuracy = accuracy_ensemble)
accuracy_ensemble %>% knitr::kable()


############################# Results ###################################

summary_ensemble     <- confusionMatrix(pred_ensemble, factor(test_house$price_fac))
summary_ensemble
accuracy_results %>% knitr::kable()
