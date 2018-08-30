library(corrplot)
library(ggplot2)
library(dplyr)
library(rcompanion)
library(mlr)
library(caTools)
library(MASS)
library(Metrics)
library(randomForest)


# reading csv file
raw_data <- read.csv("day.csv", header = TRUE)
head(raw_data)
dim(raw_data)
str(raw_data)

#checking for missing values
sapply(raw_data, function(x) {
  sum(is.na(x))
})
# There are no missing values

#convering categorical variables into factor
raw_data$season <- as.factor(raw_data$season)
levels(raw_data$season)[levels(raw_data$season) == 1] <- 'springer'
levels(raw_data$season)[levels(raw_data$season) == 2] <- 'summer'
levels(raw_data$season)[levels(raw_data$season) == 3] <- 'fall'
levels(raw_data$season)[levels(raw_data$season) == 4] <- 'winter'

raw_data$yr <- as.factor(raw_data$yr)
levels(raw_data$yr)[levels(raw_data$yr) == 0] <- 2011
levels(raw_data$yr)[levels(raw_data$yr) == 1] <- 2012
raw_data$mnth <- as.factor(raw_data$mnth)
raw_data$holiday <- as.factor(raw_data$holiday)
raw_data$weekday <- as.factor(raw_data$weekday)
raw_data$workingday <- as.factor(raw_data$workingday)
raw_data$weathersit <- as.factor(raw_data$weathersit)
levels(raw_data$weathersit)[levels(raw_data$weathersit) == 1] <-
  'Good'
levels(raw_data$weathersit)[levels(raw_data$weathersit) == 2] <-
  'Cloudy'
levels(raw_data$weathersit)[levels(raw_data$weathersit) == 3] <-
  'Bad'


#--------------------------------------------------------------------------------------------------#
#                                                                                                  #
#                        Outlier Analysis                                                          #
#                                                                                                  #
#--------------------------------------------------------------------------------------------------#
outlierKD <- function(dt, var) {
  var_name <- eval(substitute(var), eval(dt))
  na1 <- sum(is.na(var_name))
  m1 <- mean(var_name, na.rm = T)
  par(mfrow = c(2, 2), oma = c(0, 0, 3, 0))
  boxplot(var_name, main = "With outliers")
  hist(var_name,
       main = "With outliers",
       xlab = NA,
       ylab = NA)
  outlier <- boxplot.stats(var_name)$out
  mo <- mean(outlier)
  var_name <- ifelse(var_name %in% outlier, NA, var_name)
  boxplot(var_name, main = "Without outliers")
  hist(var_name,
       main = "Without outliers",
       xlab = NA,
       ylab = NA)
  title("Outlier Check", outer = TRUE)
  na2 <- sum(is.na(var_name))
  cat("Outliers identified:", na2 - na1, "n")
  cat("Propotion (%) of outliers:", round((na2 - na1) / sum(!is.na(var_name)) *
                                            100, 1), "n")
  cat("Mean of the outliers:", round(mo, 2), "n")
  m2 <- mean(var_name, na.rm = T)
  cat("Mean without removing outliers:", round(m1, 2), "n")
  cat("Mean if we remove outliers:", round(m2, 2), "n")
  
}


outlierKD(raw_data, temp) #no outliers
outlierKD(raw_data, atemp) #no outliers
outlierKD(raw_data, hum) # no extreme outlier detected
outlierKD(raw_data, windspeed) #some extreme values are present but canot be considered as outlier
outlierKD(raw_data, casual) # no logical outliers
outlierKD(raw_data, registered)# no ouliers
outlierKD(raw_data, cnt)# no ouliers


#--------------------------------------------------------------------------------------------------#
#                                                                                                  #
#                        Correlation Analysis                                                      #
#                                                                                                  #
#--------------------------------------------------------------------------------------------------#
par(mfrow = c(1, 1))
numeric_predictors <- unlist(lapply(raw_data, is.numeric))
numVarDataset <- raw_data[, numeric_predictors]
corr <- cor(numVarDataset)
corrplot(
  corr,
  method = "color",
  outline = TRUE,
  cl.pos = 'n',
  rect.col = "black",
  tl.col = "indianred4",
  addCoef.col = "black",
  number.digits = 2,
  number.cex = 0.60,
  tl.cex = 0.70,
  cl.cex = 1,
  col = colorRampPalette(c("green4", "white", "red"))(100)
)

# Findings :
# 1. temp and atemp are highly correlated

# Looking at target variable
ggplot(data = raw_data, aes(cnt)) +
  geom_histogram(aes(
    y = ..density..,
    binwidth = .5,
    colour = "black"
  ))
# Target variable looks like normal distribution

#--------------------------------------------------------------------------------------------------#
#                                                                                                  #
#                        Univariate Analysis                                                       #
#                                                                                                  #
#--------------------------------------------------------------------------------------------------#
# 1. Continous predictors
univariate_continuous <- function(dataset, variable, variableName) {
  var_name = eval(substitute(variable), eval(dataset))
  print(summary(var_name))
  ggplot(data = dataset, aes(var_name)) +
    geom_histogram(aes(binwidth = .5, colour = "black")) +
    labs(x = variableName) +
    ggtitle(paste("count of", variableName))
}

univariate_continuous(raw_data, cnt, "cnt")
univariate_continuous(raw_data, temp, "temp")
univariate_continuous(raw_data, atemp, "atemp")
univariate_continuous(raw_data, hum, "hum") # skwed towards left
univariate_continuous(raw_data, windspeed, "windspeed") #skewed towards right
univariate_continuous(raw_data, casual, "casual") # skwed towards right
univariate_continuous(raw_data, registered, "registered")

#2. categorical variables
univariate_categorical  <- function(dataset, variable, variableName) {
  variable <- enquo(variable)
  
  percentage <- dataset %>%
    dplyr::select(!!variable) %>%
    group_by(!!variable) %>%
    summarise(n = n()) %>%
    mutate(percantage = (n / sum(n)) * 100)
  print(percentage)
  
  dataset %>%
    count(!!variable) %>%
    ggplot(mapping = aes_(
      x = rlang::quo_expr(variable),
      y = quote(n),
      fill = rlang::quo_expr(variable)
    )) +
    geom_bar(stat = 'identity',
             colour = 'white') +
    labs(x = variableName, y = "count") +
    ggtitle(paste("count of ", variableName)) +
    theme(legend.position = "bottom") -> p
  plot(p)
}

univariate_categorical(raw_data, season, 'season')
univariate_categorical(raw_data, yr, "yr")
univariate_categorical(raw_data, mnth, "mnth")
univariate_categorical(raw_data, holiday, "holiday")
univariate_categorical(raw_data, weekday, "weekday")
univariate_categorical(raw_data, workingday, "workingday")
univariate_categorical(raw_data, weathersit, "weathersit")

# ------------------------------------------------------------------------------------------------ #
#
#                                     bivariate Analysis
#
#------------------------------------------------------------------------------------------------- #

# bivariate analysis for categorical variables
bivariate_categorical <-
  function(dataset, variable, targetVariable) {
    variable <- enquo(variable)
    targetVariable <- enquo(targetVariable)
    
    ggplot(
      data = dataset,
      mapping = aes_(
        x = rlang::quo_expr(variable),
        y = rlang::quo_expr(targetVariable),
        fill = rlang::quo_expr(variable)
      )
    ) +
      geom_boxplot() +
      theme(legend.position = "bottom") -> p
    plot(p)
    
  }

bivariate_continous <-
  function(dataset, variable, targetVariable) {
    variable <- enquo(variable)
    targetVariable <- enquo(targetVariable)
    ggplot(data = dataset,
           mapping = aes_(
             x = rlang::quo_expr(variable),
             y = rlang::quo_expr(targetVariable)
           )) +
      geom_point() +
      geom_smooth() -> q
    plot(q)
    
  }

bivariate_categorical(raw_data, season, cnt)
bivariate_categorical(raw_data, yr, cnt)
bivariate_categorical(raw_data, mnth, cnt)
bivariate_categorical(raw_data, holiday, cnt)
bivariate_categorical(raw_data, weekday, cnt)
bivariate_categorical(raw_data, workingday, cnt)
bivariate_categorical(raw_data, weathersit, cnt)

bivariate_continous(raw_data, temp, cnt)
bivariate_continous(raw_data, atemp, cnt)
bivariate_continous(raw_data, hum, cnt)
bivariate_continous(raw_data, windspeed, cnt)
bivariate_continous(raw_data, casual, cnt)
bivariate_continous(raw_data, registered, cnt)


# removing instant and dteday
raw_data$instant <- NULL
raw_data$dteday <- NULL
raw_data$casual <- NULL
raw_data$registered <- NULL


# ------------------------------------------------------------------------------------------------ #
#
#                                     Feature scaling or Normalization                             #
#
#------------------------------------------------------------------------------------------------- #

scaledData <- normalizeFeatures(raw_data,'cnt')





# Function for calculating Mean Absolute Error
MAE <- function(actual,predicted){
  error = actual - predicted
  mean(abs(error))
}
# ----------------- Model 1 Linear Regression -----------------------------------------------------#


set.seed(654)
split <- sample.split(raw_data$cnt, SplitRatio = 0.70)
training_set <- subset(raw_data, split == TRUE)
test_set <- subset(raw_data, split == FALSE)


model1 <- lm(cnt ~ ., data = training_set)


# step wise model selection

modelAIC <- stepAIC(model1, direction = "both")
summary(modelAIC)




# Apply prediction on test set
test_prediction <- predict(modelAIC, newdata = test_set)


test_rmse <- rmse(test_set$cnt, test_prediction)
print(paste("root-mean-square error for linear regression model is ", test_rmse))
print(paste("Mean Absolute Error for linear regression model is ",MAE(test_set$cnt,test_prediction)))
print("summary of predicted count values")
summary(test_prediction)
print("summary of actual count values")
summary(test_set$cnt)

# From the summary we can observe negative prediction values
#We will perform log transformation of trarget variable
model2 <- lm(log(cnt)~., data = training_set)

stepwiseLogAICModel <- stepAIC(model2,direction = "both")
test_prediction_log<- predict(stepwiseLogAICModel, newdata = test_set)
predict_test_nonlog <- exp(test_prediction_log)

test_rmse2 <- rmse(test_set$cnt, predict_test_nonlog)
print(paste("root-mean-square error between actual and predicted", test_rmse))
print(paste("Mean Absolute Error for linear regression model is ",
            MAE(test_set$cnt,predict_test_nonlog)))

summary(predict_test_nonlog)
summary(test_set$cnt)



par(mfrow = c(2, 2))
plot(stepwiseLogAICModel)

# ----------------- Model 2 Random forest -----------------------------------------------------#

rf_model_1 <- randomForest(cnt ~.,
                           data = training_set,ntree = 500, mtry = 8, importance = TRUE)
print(rf_model_1)
par(mfrow = c(1,1))
plot(rf_model_1)

cv_RF <- cv

# 300 trees selected from the plot

tumedmodel <- tuneRF(training_set[,1:11], training_set[,12], stepFactor = 0.5, plot = TRUE, 
                     ntreeTry = 250, trace = TRUE, improve = 0.05)

# selected mtry = 6 from the plot

tuned_randomForest <-  randomForest(cnt ~. - atemp,
                                    data = training_set,ntree = 250, mtry = 6, importance = TRUE)
tuned_randomForest
# predicting using random forest model 1
rf1_prediction <- predict(tuned_randomForest,test_set[,-12])
rmse(rf1_prediction,test_set$cnt)
print(paste("Mean Absolute Error for Random forest regressor  is ",
            MAE(test_set$cnt,rf1_prediction)))

#745

varImpPlot(tuned_randomForest)


# Random forest is performing better than linear regression.

# Model input and output for linear regression and Random forest
write.csv(test_set, file = "InputLinearRegressionR.csv")
write.csv(test_set, file = "InputRandomForest.csv")
write.csv(predict_test_nonlog, file="outputLogisticRegressionR.csv")
write.csv(predict_test_nonlog, file="outputLogisticRegressionR.csv")