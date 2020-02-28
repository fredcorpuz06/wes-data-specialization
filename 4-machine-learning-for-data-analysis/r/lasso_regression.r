library(tidyverse)
library(caret) # machine learning
library(glmnet)

# Load dataset
df <- read_csv("tree_addhealth.csv") %>%
    select(-id) %>%
    mutate(TREG1 = factor(TREG1)) %>%
    na.omit()

head(df)


# Split into train and test
set.seed(1234)
trainIndex <- createDataPartition(df$TREG1, p = 0.6, list = FALSE, times = 1)

train <- df[trainIndex, ]
test <- df[-trainIndex, ]

# Build model on training data
tc <- trainControl(method = "cv", number = 10)
cv_classifier <- train(
    TREG1 ~ ., 
    data = training,
    method = 'binomial',
    trControl = tc,
    preProcess = c("center", "scale")
)


# Predict and evaluate
pred <- predict(classifier, newdata = testing)
confusionMatrix(pred, testing$TREG1)


##--------------------------------------------------------
## glmnet package
##--------------------------------------------------------
# predictor variables without the intercept
# any categorical variables are converted to
# dummy codes
x <- model.matrix(TREG1 ~ ., train)[,-1]
y <- train$TREG1

# fit model
model.lasso <- glmnet(x, y, family="binomial")

# 10-fold cross validation of lambda values on accuracy
set.seed(1234)
cv.lambda <- cv.glmnet(x, y, family="binomial")
bestlam <- cv.lambda$lambda.min
bestlam

# coefficients for final model
coef(model.lasso, s=bestlam)


x <- model.matrix(TREG1 ~ ., test)[,-1]

# predict class assuming prob cutoff of .5 
pred <- predict(model.lasso, s=bestlam, newx=x, type="class")


actual    <- ifelse(test$TREG1 == 1, "1", "0")
confusionMatrix(pred, actual, positive="1")