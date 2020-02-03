library(tidyverse)
library(caret) # machine learning
library(rpart) # decision trees
library(rpart.plot) # visualize trees

# Load dataset
df <- read_csv("tree_addhealth.csv") %>%
    select(-id) %>%
    na.omit()

summary(df)
head(df)


# Split into train and test
set.seed(1234)
trainIndex <- createDataPartition(
    df$TREG1, p = 0.6,
    list = FALSE,
    times = 1
)

train <- df[trainIndex, ]
test <- df[-trainIndex, ]

# Build model on train data
classifier <- rpart(TREG1 ~ ., data = train, method = "class")

# Predict and evaluate
pred <- predict(classifier, newdata = test, type = "class")
confusionMatrix(pred, factor(test$TREG1))

# Display decision tree
rpart.plot(classifier)
