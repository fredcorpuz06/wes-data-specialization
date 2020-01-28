library(tidyverse)
library(caret) # machine learning
library(randomForest)

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
    method = 'rf',
    ntree = 25
)

# Predict and evaluate
pred <- predict(classifier, newdata = testing)
confusionMatrix(pred, testing$TREG1)

# Display variable importance
varImpPlot(classifier, type = 2, main = "Variable Importance") 

# Running a different number of trees to see effect on accuracy
# tc <- trainControl(method = "cv", number = )
# cv_classifier <- train(
#     TREG1 ~ ., 
#     data = training,
#     method = 'rf',
#     trControl = tc
#     tuneGrid = c()
# )

n_trees = 1:25

my_random_forest <- function(n_tree){
    classifier <- train(
        TREG1 ~ ., 
        data = training,
        method = 'rf',
        ntree = n_tree
    )
    
    pred <- predict(classifier, newdata = testing)
    results <- confusionMatrix(pred, testing$TREG1)
    return(results$accuracy)
}

accuracy <- map(n_trees, my_random_forest)
plot(accuracy)