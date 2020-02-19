library(tidyverse) # data manipulation
library(caret) # machine learning 
library(randomForest) # random forest

# Load dataset
df <- read_csv("tree_addhealth.csv") %>%
    select(-id) %>%
    mutate(TREG1 = factor(TREG1)) %>%
    na.omit()

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

# Build random forest model on train data
classifier <- randomForest(TREG1 ~ ., data = train, ntree = 25)

# Predict and evaluate
pred <- predict(classifier, newdata = test, type = "class")
confusionMatrix(pred, test$TREG1)

# Display variable importance
varImpPlot(classifier, type = 2, main = "Variable Importance") 

# Running a different number of trees, observe accuracy
n_trees = 1:25

my_random_forest <- function(n_tree){
    classifier <- randomForest(TREG1 ~ ., data = train, ntree = n_tree)
    pred <- predict(classifier, newdata = test)
    results <- confusionMatrix(pred, test$TREG1)
    return(results$overall['Accuracy'])
}

accuracies <- map(n_trees, my_random_forest)
plot(n_trees, accuracies)
