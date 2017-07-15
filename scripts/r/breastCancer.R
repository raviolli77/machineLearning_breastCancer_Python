# LOAD PACKAGES
setwd('set/approp/wd')

require(data.table)
require(caret)
require(ggcorrplot)
require(GGally)
require(class)
require(randomForest)
require(nnet)
require(e1071)
require(pROC)
require(class)

# EXPLORATORY ANALYSIS
breastCancer <- read.csv('wdbc.data.txt', header = TRUE)
head(breastCancer)
dim(breastCancer)


# REMOVING 'id_number'
breastCancer$id_number <- NULL

table(breastCancer$diagnosis)
summary(breastCancer)

# Scatterplot Matrix
p <- ggpairs(data = breastCancer,
        columns = c('concave_points_worst', 'concavity_mean', 
                    'perimeter_worst', 'radius_worst', 
                    'area_worst', 'diagnosis'),
        mapping = aes(color = diagnosis)) + 
  theme(panel.background = element_rect(fill = '#fafafa')) + 
  ggtitle('Scatter Plot Matrix')

# MANUALLY CHANGING COLORS OF PLOT
# BORROWED FROM: https://stackoverflow.com/questions/34740210/how-to-change-the-color-palette-for-ggallyggpairs
for(i in 1:p$nrow) {
  for(j in 1:p$ncol){
    p[i,j] <- p[i,j] + 
      scale_fill_manual(values=c("red", "#875FDB")) +
      scale_color_manual(values=c("red", "#875FDB"))  
  }
}

p
# Pearson Correlation 
corr <- round(cor(breastCancer[, 2:31]), 2)
ggcorrplot(corr,
           colors = c('red', 'white', '#875FDB')) + 
  ggtitle('Peasron Correlation Matrix')

# Box Plot
ggplot(data = stack(breastCancer), 
       aes(x = ind, y = values)) + 
  geom_boxplot() + 
  coord_flip(ylim = c(-.05, 50)) +
  theme(panel.background = element_rect(fill = '#fafafa')) + 
  ggtitle('Box Plot of Unprocessed Data')

# NORMALIZING 
preprocessparams <- preProcess(breastCancer[, 3:31], method=c('range'))

breastCancerNorm <- predict(preprocessparams, breastCancer[, 3:31])

breastCancerNorm <- data.table(breastCancerNorm, diangosis = breastCancer$diagnosis)

summary(breastCancerNorm)
# Box Plot of Normalized data
ggplot(data = stack(breastCancerNorm), 
       aes(x = ind, y = values)) + 
  geom_boxplot() + 
  coord_flip(ylim = c(-.05, 1.05)) + 
  theme(panel.background = element_rect(fill = '#fafafa'))  + 
  ggtitle('Box Plot of Normalized Data')

# TRAINING AND TEST SET
breastCancer$diagnosis <- gsub('M', 1, breastCancer$diagnosis)
breastCancer$diagnosis <- gsub('B', 0, breastCancer$diagnosis)

breastCancer$diagnosis <- as.numeric(breastCancer$diagnosis)

set.seed(42)
trainIndex <- createDataPartition(breastCancer$diagnosis, 
                                  p = .8, 
                                  list = FALSE, 
                                  times = 1)

training_set <- breastCancer[ trainIndex, ]
test_set <- breastCancer[ -trainIndex, ]
## Kth Nearest Neighbor

# TRAINING AND TEST SETS ARE SET UP DIFFERENTLY FOR KNN
# SO HERE WE'RE DOING THAT
# Intialize a class set as vector
class_set <- as.vector(training_set$diagnosis)

test_set_knn <- test_set
training_set_knn <- training_set
test_set_knn$diagnosis <- NULL
training_set_knn$diagnosis <- NULL

head(test_set_knn)

# FITTING MODEL 
fit_knn <- knn(training_set_knn, test_set_knn, class_set, k = 7)

# TEST SET EVALUATIONS
table(test_set$diagnosis, fit_knn)

# TEST ERROR RATE: 0.063

## RANDOM FOREST
# FITTING MODEL
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

set.seed(42)
fit_rf <- train(as.factor(diagnosis) ~ ., 
                data = training_set, 
                method = "rf", 
                trControl = fitControl)

fit_rf$finalModel

# VARIABLE IMPORTANCE
varImportance <- varImp(fit_rf, scale = FALSE)

varImportanceScores <- data.table(varImportance$importance, names = colnames(breastCancer[, 2:31]))

varImportanceScores

# VISUAL OF VARIABLE IMPORTANCE
ggplot(varImportanceScores, 
       aes(reorder(names, Overall), Overall)) + 
  geom_bar(stat='identity', 
           fill = '#875FDB') + 
  theme(panel.background = element_rect(fill = '#fafafa')) + 
  coord_flip() + 
  labs(x = 'Feature', y = 'Importance') + 
  ggtitle('Feature Importance for Random Forest Model')

# TEST SET EVALUATIONS
predict_values <- predict(fit_rf, newdata = test_set)

table(predict_values, test_set$diagnosis)

# TEST ERROR RATE: 0.027

# NEURAL NETWORKS

# CREATING NORMALIZED TRAINING AND TEST SET
set.seed(42)
trainIndex_norm <- createDataPartition(breastCancerNorm$diangosis, 
                                  p = .8, 
                                  list = FALSE, 
                                  times = 1)

training_set_norm <- breastCancerNorm[ trainIndex, ]
test_set_norm <- breastCancerNorm[ -trainIndex, ]

training_set_norm

fit_nn <- train(as.factor(diangosis) ~ ., 
                data = training_set_norm,
                method = "nnet", 
                hidden = 3, 
                algorithm = 'backprop')  

fit_nn$finalModel
plot(fit_nn)
predict_val_nn <- predict(fit_nn, newdata = test_set_norm)

table(predict_val_nn, test_set$diagnosis)

# TEST ERROR RATE: 0.035