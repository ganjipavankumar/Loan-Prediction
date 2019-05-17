#Loan Prediction
#install.packages('data.table')
library(data.table)

#importing data
#test_data <- fread('test_data.csv', stringsAsFactors = T)
train_data <- fread('train_data.csv', stringsAsFactors = T)

head(train_data)
sapply(train_data, class)

#checking Missing values
sort(sapply(train_data, function(x) { sum(is.na(x)) }), decreasing=TRUE)
#sort(sapply(test_data, function(x) { sum(is.na(x)) }), decreasing=TRUE)

#Imputing Missing Values for training set
library(mice)
imputed_data <- mice(train_data[,c(9,10,11)], m=5, maxit = 50, method = 'rf', seed = 500)
train_data[,c(9,10,11)] <- complete(imputed_data, 2)

# #imputing Missing Values for test set
# imputed_data <- mice(test_data[,c(9,10,11)], m=5, maxit = 50, method = 'rf', seed = 500)
# test_data[,c(9,10,11)] <- complete(imputed_data, 2)

#Encoding Categorical variables into Numeric for training data
train_data$Gender <- factor(train_data$Gender, labels = c(1,2), levels = c('Female','Male'))
train_data$Married <- factor(train_data$Married, labels = c(0,1), levels = c('No','Yes'))
train_data$Education <- factor(train_data$Education, labels = c(1,2), levels = c('Graduate','Not Graduate') )
train_data$Self_Employed <- factor(train_data$Self_Employed, labels = c(1,2,3), c('','Yes','No'))
train_data$Property_Area <- factor(train_data$Property_Area, labels = c(1,2,3), levels = c('Rural','Urban','Semiurban'))
train_data$Loan_Status <- factor(train_data$Loan_Status, labels = c(0,1), levels = c('N','Y') )

# #Encoding Categorical variables into factors
# test_data$Gender <- factor(test_data$Gender, labels = c(1,2), levels = c('Female','Male'))
# test_data$Married <- factor(test_data$Married, labels = c(0,1), levels = c('No','Yes'))
# test_data$Education <- factor(test_data$Education, labels = c(1,2), levels = c('Graduate','Not Graduate') )
# test_data$Self_Employed <- factor(test_data$Self_Employed, labels = c(1,2,3), c('','Yes','No'))
# test_data$Property_Area <- factor(test_data$Property_Area, labels = c(1,2,3), levels = c('Rural','Urban','Semiurban'))

#set column level
levels(train_data$Dependents)[levels(train_data$Dependents) ==  "3+"] <- "3"

#Imputing Missing Value for Gender and Married
imputed_data <- mice(train_data[,c(2,3)], m=5, maxit = 50, method = 'rf', seed = 500)
train_data[,c(2,3)] <- complete(imputed_data, 2)

# imputed_data <- mice(test_data[,c(2,3)], m=5, maxit = 50, method = 'rf', seed = 500)
# test_data[,c(2,3)] <- complete(imputed_data, 2)

#Converting categoical to integer for train data
train_data$Gender <- as.integer(train_data$Gender)
train_data$Loan_ID <- as.integer(train_data$Loan_ID)
train_data$Married <- as.integer(train_data$Married)
train_data$Dependents <- as.integer(train_data$Dependents)
train_data$Education <- as.integer(train_data$Education)
train_data$Self_Employed <- as.integer(train_data$Self_Employed)
train_data$Property_Area <- as.integer(train_data$Property_Area)
#train_data$Loan_Status <- as.integer(train_data$Loan_Status)

# #Converting categorical to integer for test data
# test_data$Gender <- as.integer(test_data$Gender)
# test_data$Loan_ID <- as.integer(test_data$Loan_ID)
# test_data$Married <- as.integer(test_data$Married)
# test_data$Dependents <- as.integer(test_data$Dependents)
# test_data$Education <- as.integer(test_data$Education)
# test_data$Self_Employed <- as.integer(test_data$Self_Employed)
# test_data$Property_Area <- as.integer(test_data$Property_Area)

#feature Scaling
train_data$Loan_ID <- scale(train_data$Loan_ID)
train_data$Gender <- scale(train_data$Gender)
train_data$Married <- scale(train_data$Married)
train_data$Dependents <- scale(train_data$Dependents)
train_data$Education <- scale(train_data$Education)
train_data$Self_Employed <- scale(train_data$Self_Employed)
train_data$ApplicantIncome <- scale(train_data$ApplicantIncome)
train_data$CoapplicantIncome <- scale(train_data$CoapplicantIncome)
train_data$LoanAmount <- scale(train_data$LoanAmount)
train_data$Loan_Amount_Term <- scale(train_data$Loan_Amount_Term)
train_data$Credit_History <- scale(train_data$Credit_History)
train_data$Property_Area <- scale(train_data$Property_Area)

#test_data <- scale(test_data)

#Splitting train data in to two datasets for validation
library(caTools)
set.seed(123)
split <- sample.split(train_data$Loan_Status, SplitRatio = 0.80)
trainset <- subset(train_data, split == TRUE)
testset <- subset(train_data, split == FALSE)


#Model building

#Applying kernel PCA for dimensinality reduction
# install.packages('kernlab')
library(kernlab)
kpca = kpca(~., data = trainset, kernel = 'rbfdot', features = 10)
training_set_pca = as.data.frame(predict(kpca, trainset))
training_set_pca$Loan_Status = trainset$Loan_Status
test_set_pca = as.data.frame(predict(kpca, testset))
test_set_pca$Loan_Status = testset$Loan_Status

#Fitting training data to KNN
library(class)
knn_classifier <- knn(train = trainset[,-13], test = testset[,-13], cl = trainset$Loan_Status, k=5, prob = TRUE)

#Fitting training data to SVM
library(e1071)
svm_classifier = svm(formula = Loan_Status ~ .,
                 data = trainset,
                 type = 'C-classification',
                 kernel = 'sigmoid')

#Fitting training data to naive bayes
library(e1071)
nb_classifier = naiveBayes(x = trainset[,-13],
                        y = trainset$Loan_Status)

#Fitting training data to decision tree
library(rpart)
dt_classifier <- rpart(formula = Loan_Status ~ ., data = train_data)

#Fitting training data to Random Forest
library(randomForest)
set.seed(123)
rf_classifier <- randomForest(x = trainset[,-13], y = trainset$Loan_Status, ntree = 500)

#Fitting training data to XGBoost Model
library(xgboost)
xg_classifier <- xgboost(data = as.matrix(trainset[,-13]), label = trainset$Loan_Status, nrounds = 10)

#Fitting training data to Gradient boost model
#install.packages('gbm)
library(gbm)
trainset <- trainset[,-14]
gb_classifier <- gbm(Loan_Status ~ ., data = trainset,distribution = "gaussian",n.trees = 10000, interaction.depth = 4, shrinkage = 0.01)

#cross validation
library(caret)
train_control<- trainControl(method="cv", number=10, savePredictions = TRUE)
model<- train(Loan_Status~., data=trainset, trControl=train_control, method="rpart")

#Predicting the test results
svm_pred <- predict(svm_classifier, newdata = testset[,-13])
nb_pred <-  predict(nb_classifier, newdata = testset[,-13])
dt_pred <-  predict(dt_classifier, newdata = testset[,-13], type='class')
rf_pred <-  predict(rf_classifier, newdata = testset[,-13])
xg_pred <-  predict(xg_classifier, newdata = as.matrix(testset[,-13]))
xg_pred <- (xg_pred >= 0.5)
n.trees = seq(from=100 ,to=10000, by=100)
gb_pred <- predict(gb_classifier, newdata = testset[,-13], n.trees = n.trees)
kf_pred <- predict(model, newdata = testset[,-13])

#confusion matrix
cm_knn <- table(testset$Loan_Status,knn_classifier)     #102/101 correct predictions 20/21 incorrect predictions
cm_svm <- table(testset$Loan_Status, svm_pred)          #99/101 correct predictions 24/21 incorrect predictions
cm_nb <- table(testset$Loan_Status, nb_pred)            #103/101 correct predictions 19/21 incorrect predictions
cm_dt <- table(testset$Loan_Status, dt_pred)            #101/104 correct predictions 21/18 incorrect predictions
cm_rf <- table(testset$Loan_Status, rf_pred)            #98/107 correct predictions 24/15 incorrect predictions
cm_xg <- table(testset$Loan_Status, xg_pred)            #84/84 correct predictions 38/38 incorrect predictions
cm_gb <- table(testset$Loan_Status, gb_pred)
cm_kf <- table(testset$Loan_Status, kf_pred)            #101/99 correct predictions 21/23 incorrect predictions



####################################################################################################################


#After Applying PCA
#Fitting training data to KNN
library(class)
knn_classifier <- knn(train = training_set_pca[,-11], test = test_set_pca[,-11], cl = trainset$Loan_Status, k=5, prob = TRUE)

#Fitting training data to SVM
library(e1071)
svm_classifier = svm(formula = Loan_Status ~ .,
                     data = training_set_pca,
                     type = 'C-classification',
                     kernel = 'sigmoid')

#Fitting training data to naive bayes
library(e1071)
nb_classifier = naiveBayes(x = training_set_pca[,-11],
                           y = training_set_pca$Loan_Status)

#Fitting training data to decision tree
library(rpart)
dt_classifier <- rpart(formula = Loan_Status ~ ., data = training_set_pca)

#Fitting training data to Random Forest
library(randomForest)
set.seed(123)
rf_classifier <- randomForest(x = training_set_pca[,-11], y = training_set_pca$Loan_Status, ntree = 500)

#Fitting training data to XGBoost Model
library(xgboost)
xg_classifier <- xgboost(data = as.matrix(training_set_pca[,-11]), label = training_set_pca$Loan_Status, nrounds = 10)

#Fitting training data to Gradient boost model
#install.packages('gbm)
library(gbm)
gb_classifier <- gbm(Loan_Status ~ ., data = training_set_pca,distribution = "gaussian",n.trees = 10000, interaction.depth = 4, shrinkage = 0.01)


cm = table(test_fold[, 3], y_pred)
accuracy <- (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])


#Predicting the test results
svm_pred <- predict(svm_classifier, newdata = test_set_pca[,-11])
nb_pred <-  predict(nb_classifier, newdata = test_set_pca[,-11])
dt_pred <-  predict(dt_classifier, newdata = test_set_pca[,-11], type='class')
rf_pred <-  predict(rf_classifier, newdata = test_set_pca[,-11])
xg_pred <-  predict(xg_classifier, newdata = as.matrix(test_set_pca[,-11]))
xg_pred <- (xg_pred >= 0.5)
n.trees = seq(from=100 ,to=10000, by=100)
gb_pred <- predict(gb_classifier, newdata = test_set_pca[,-11], n.trees = n.trees)


#confusion matrix
cm_knn_pca <- table(test_set_pca$Loan_Status,knn_classifier)     #101 correct predictions 21 incorrect predictions
cm_svm_pca <- table(test_set_pca$Loan_Status, svm_pred)          #101 correct predictions 21 incorrect predictions
cm_nb_pca <- table(test_set_pca$Loan_Status, nb_pred)            #101 correct predictions 21 incorrect predictions
cm_dt_pca <- table(test_set_pca$Loan_Status, dt_pred)            #104 correct predictions 18 incorrect predictions
cm_rf_pca <- table(test_set_pca$Loan_Status, rf_pred)            #107 correct predictions 15 incorrect predictions
cm_xg_pca <- table(test_set_pca$Loan_Status, xg_pred)            #84 correct predictions 38 incorrect predictions


