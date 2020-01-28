library(tidyverse)
library(caret) # machine learning
# library(rpart)
library(rpart.plot) # visualize trees

# Load dataset
df <- read_csv("../data/treeaddhealth.csv") %>%
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
    method = 'rpart'
)

# Predict and evaluate
pred <- predict(classifier, newdata = testing)
confusionMatrix(pred, testing$TREG1)

# Display decision tree
rpart.plot(classifier)
pdf("picture_out1.pdf") # save file
