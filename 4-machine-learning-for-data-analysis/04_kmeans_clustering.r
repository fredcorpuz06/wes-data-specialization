library(tidyverse)
library(caret) # machine learning
# library(rpart)
library(rpart.plot) # visualize trees

# Load dataset
df <- read_csv("../data/tree_addhealth.csv") %>%
    na.omit()

summary(df)
head(df)


# Split into training and test
set.seed(1234)
trainIndex <- createDataPartition(
    df$TREG1, p = 0.6,
    list = FALSE,
    times = 1
)

training <- df[trainIndex, ]
testing <- df[-trainIndex, ]

# Build model on training data
classifier <- train(
    TREG1 ~ ., 
    data = training,
    method = 'knn',
    preProcess = c("center", "scale"),
    trControl = trainControl(method="cv", number=10),
    metric = "Accuracy",
    tuneLength = 10
)
plot(classifier)

# Predict and evaluate
pred <- predict(classifier, newdata = testing)
confusionMatrix(pred, testing$TREG1)
