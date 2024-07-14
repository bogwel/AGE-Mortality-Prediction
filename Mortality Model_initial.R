# Load the package
library(randomForest)
library(caret)
library(e1071)
library(dplyr)
library(haven)
library(MLmetrics)
library(xgboost)
library(mice)
library(class)
library(Amelia)
library(neuralnet)
library(fastAdaboost)
library(adabag)
library(Boruta)
library(ROSE)
library(DMwR)
library(caretEnsemble)
library(pROC)
library(modelplotr)
library(SuperLearner)
library(kernlab)

#Install packages
#install.packages('kernlab')






SDH_data <- read_dta("C:/IEIP/DATA MANAGEMENT/requests/IPD SIAYA/Data/rota_SCRH_08Dec2020.dta")


#filtering predictors that had a chi-square of P<=0.2
SDH_data1 <- SDH_data %>% select(agecat,ipdtem,ipdhg,ipdpuls,ipdmdrhd,ipdmvom,ipdletha,ipdmuncons,ipdmfev,ipdpneg,ipdalert,
                                 ipdresirit,ipdsunkeye,ipdskinp,ipdcaprefi,ipddrink,ipdsunkf,ipdreyes
                                 ,ipdwast,ipdtfiv,dehydrationstatus,stunting,underweight,nutri_wasting,
                                 ipdmadmin,ipdhu,ipdchestin,ipdstrid,ipdnasal,ipdlbpar, bgluco, ipdadmdehy, death) %>%
  mutate(death=factor(death,levels=c(1, 0), labels=c("Yes", "NO"))) %>%
  filter(!is.na( death )) 

#visualize the missing data
missmap(SDH_data1)


#Use mice package to impute missing values

SDH_miss <- SDH_data1 %>%
  select(ipdpuls,ipdwast,ipdreyes,ipdsunkf,ipdletha,ipdpneg,ipdmuncons,ipdcaprefi,ipdmfev,nutri_wasting,
         underweight,ipdmdrhd,ipddrink,ipdtfiv,stunting,ipdalert,ipdresirit, ipdtem, ipdmvom,ipdsunkeye,
         ipdmadmin,ipdhu,ipdchestin,ipdstrid,ipdnasal,ipdlbpar, bgluco, ipdadmdehy)

mice_mod <- mice(SDH_miss, method='rf', drop = FALSE, rfPackage = "randomForest")
mice_complete <- complete(mice_mod)

#Transfer the predicted missing values into the main data set
SDH_data1$ipdpuls <- mice_complete$ipdpuls
SDH_data1$ipdwast <- mice_complete$ipdwast
SDH_data1$ipdreyes <- mice_complete$ipdreyes
SDH_data1$ipdsunkf <- mice_complete$ipdsunkf
SDH_data1$ipdletha <- mice_complete$ipdletha
SDH_data1$ipdpneg <- mice_complete$ipdpneg
SDH_data1$ipdmuncons <- mice_complete$ipdmuncons
SDH_data1$ipdcaprefi <- mice_complete$ipdcaprefi
SDH_data1$ipdmfev <- mice_complete$ipdmfev
SDH_data1$nutri_wasting <- mice_complete$nutri_wasting
SDH_data1$underweight <- mice_complete$underweight
SDH_data1$ipdmdrhd <- mice_complete$ipdmdrhd
SDH_data1$ipddrink <- mice_complete$ipddrink
SDH_data1$ipdtfiv <- mice_complete$ipdtfiv
SDH_data1$stunting <- mice_complete$stunting
SDH_data1$ipdalert <- mice_complete$ipdalert
SDH_data1$ipdresirit <- mice_complete$ipdresirit
SDH_data1$ipdtem <- mice_complete$ipdtem
SDH_data1$ipdmvom <- mice_complete$ipdmvom
SDH_data1$ipdsunkeye <- mice_complete$ipdsunkeye
SDH_data1$ipdmadmin <- mice_complete$ipdmadmin
SDH_data1$ipdhu <- mice_complete$ipdhu
SDH_data1$ipdchestin <- mice_complete$ipdchestin
SDH_data1$ipdstrid <- mice_complete$ipdstrid
SDH_data1$ipdnasal <- mice_complete$ipdnasal
SDH_data1$ipdlbpar <- mice_complete$ipdlbpar
SDH_data1$bgluco <- mice_complete$bgluco
SDH_data1$ipdadmdehy <- mice_complete$ipdadmdehy

missmap(SDH_data1)
#table(drh_duration_CM$mal_end)


row.has.na <- apply(SDH_data1, 1, function(x){any(is.na(x))})
SDH_data1_no_NA <- SDH_data1[!row.has.na, ]


# Perform Boruta search
boruta_output <- Boruta(death ~ ., data=na.omit(SDH_data1_no_NA), doTrace=1)  
# Get significant variables including tentatives
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

# Do a tentative rough fix
roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif1 <- getSelectedAttributes(roughFixMod)
print(boruta_signif1)

# Variable Importance Scores
imps <- attStats(boruta_output)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])  # descending sort

# Plot variable importance
Mortality_features <- plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  

Mortality_features

SDH_data2 <- SDH_data1_no_NA %>% select(-ipddrink, -ipdmadmin, -ipdhu, -ipdletha, -ipdreyes,
                                        -agecat, -ipdcaprefi, -ipdmdrhd, -bgluco)


set.seed(4567)
test_index_M <- createDataPartition(SDH_data2$death, times = 1, p = 0.75, list = FALSE)
train_set_M  <- SDH_data2[test_index_M, ]
test_set_M <- SDH_data2[-test_index_M, ]


# Define the control
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- F1_Score(y_pred  = data$pred, y_true = data$obs, positive = lev[1])
  c(F1 = f1_val)
}

trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid",
                          summaryFunction=f1,
                          classProbs = TRUE,
                          sampling = "up")

trControl1 <- trainControl(method = "cv",
                           number = 10,
                           search = "grid",
                           summaryFunction=f1,
                           classProbs = TRUE,
                           sampling = "down")

trControl2 <- trainControl(method = "cv",
                           number = 10,
                           search = "grid",
                           summaryFunction=f1,
                           classProbs = TRUE,
                           sampling = "rose")


trControl3 <- trainControl(method = "cv",
                           number = 10,
                           search = "grid",
                           summaryFunction=twoClassSummary,
                           classProbs = TRUE)

trControl4 <- trainControl(
  search = "grid",
  summaryFunction=mnLogLoss,
  classProbs = TRUE)

set.seed(1234)
# Run the  default model
rf_default_M <- train(death ~ ., method = "rf", data=train_set_M, metric= "F1",
                       trControl = trControl,threshold = 0.3)
print(rf_default_M)

set.seed(1234)
rf_default_M1 <- train(death ~ ., method = "rf", data=train_set_M, metric= "F1",
                      trControl = trControl1,threshold = 0.3)
print(rf_default_M1)

set.seed(1234)
rf_default_M2 <- train(death ~ ., method = "rf", data=train_set_M, metric= "F1",
                       trControl = trControl2,threshold = 0.3)
print(rf_default_M2)


rf_prediction_M <-predict(rf_default_M, test_set_M,type = "raw")
confusionMatrix(rf_prediction_M, test_set_M$death)

rf_prediction_M1 <-predict(rf_default_M1, test_set_M,type = "raw")
confusionMatrix(rf_prediction_M1, test_set_M$death)

rf_prediction_M2 <-predict(rf_default_M2, test_set_M,type = "raw")
confusionMatrix(rf_prediction_M2, test_set_M$death)


varImp(rf_default_M1)
FI_val <- F_meas(data = rf_prediction_CM, reference = factor(test_no_NA$mal_end))
FI_val

# Computing business value of model using modelplotr

scores_and_ntiles <- prepare_scores_and_ntiles(datasets=list("train_set_M","test_set_M"),
                                               dataset_labels = list("train data","test data"),
                                               models = list("rf_default_M1"),  
                                               model_labels = list("random forest"), 
                                               target_column="death",
                                               ntiles = 100)

plot_input <- plotting_scope(prepared_input = scores_and_ntiles)

#cummulative gains 
plot_cumgains(data = plot_input)

#Cumulative lift
plot_cumlift(data = plot_input)

#Response plot
plot_response(data = plot_input)

#Cumulative response plot
plot_cumresponse(data = plot_input)

#multiple plots
plot_multiplot(data = plot_input)


#reruning model with variables having a variable importance of >10%
set.seed(1234)
# Run the  default model
rf_default_M1 <- train(death ~ (ipdpuls+ipdtem+ipdhg+ipdmdrhd+ipdmdrhd+ipdalert+
                                  underweight+agecat+ipdlbpar+ipdresirit+nutri_wasting+
                                  stunting+ipddrink+ipdmuncons+ipdskinp+ipdtfiv+
                                  ipdmvom+ipdletha+ipdnasal+ipdadmdehy+ipdmfev), method = "rf", data=train_set_M, metric= "F1",
                      trControl = trControl,threshold = 0.3)

print(rf_default_M1)

rf_prediction_M <-predict(rf_default_M1, test_set_M,type = "raw")
confusionMatrix(rf_prediction_M, test_set_M$death)


#*************************************************************************************************************************
#*************************************************************************************************************************
#Gradient Boosting Algorithm

# Fit the model on the training set

set.seed(12345)
gbm_mortality_model <- train( death ~ ., 
                        data = train_set_M, method = "xgbTree", shrinkage = 0.01,
                        trControl = trControl1,metric= "F1")

#print(gbm_RSV_model)
gbm_prediction <-predict(gbm_mortality_model, test_set_M,type = "raw")
confusionMatrix(gbm_prediction, test_set_M$death)
#summary(gbm_RSV_model$feature_names)
#xgb.importance(colnames(train_set_M), model = gbm_RSV_model)
varImp(gbm_mortality_model)

#*************************************************************************************************************************
#*************************************************************************************************************************
#Naive Bayes Algorithm (sen=77, Sep=44, F1=55)

set.seed(1234)
nb_mortality_model <- train(
  death ~ ., 
  data = train_set_M, method = "naive_bayes",
  trControl = trControl, metric= "F1")
print(nb_RSV_model)

#check for variable importance
nb_prediction <-predict(nb_mortality_model, test_set_M,type = "raw")
confusionMatrix(nb_prediction, test_set_M$death)
varImp(nb_mortality_model)

#*************************************************************************************************************************
#*************************************************************************************************************************
#LDA Algorithm (sen=77, Sep=44, F1=55)

set.seed(1234)
lda_Mortality_model <- train(
  death ~ ., 
  data = train_set_M, method = "lda",
  trControl = trControl,metric= "F1")
print(lda_Mortality_model)

#check for variable importance
lda_prediction <-predict(lda_Mortality_model, test_set_M,type = "raw")
confusionMatrix(lda_prediction, test_set_M$death)

varImp(lda_Mortality_model)

########################################################################################################
#Run the kNN Algorithm 
set.seed(1234)
Mortality_knn <- train(death ~ ., 
                 data = train_set_M, method = "knn",
                 trControl = trControl,metric= "F1")

print(RSV_knn)

#check for variable importance
knn_prediction <-predict(Mortality_knn, test_set_M,type = "raw")
confusionMatrix(knn_prediction, test_set_M$death)

varImp(Mortality_knn)

######################################################################################################################
#Run the nueralnet Algorithm 
set.seed(77777)
Mortality_nn <- train(death ~ ., 
                data = train_set_M, method = "nnet",
                trControl = trControl2,metric= "F1")

print(Mortality_nn)

#check for variable importance
nn_prediction <-predict(Mortality_nn, test_set_M,type = "raw")
confusionMatrix(nn_prediction, test_set_M$death)

#########################################################################################################################
##################Implementing stacked ensemble using Super learner##############################################

listWrappers()
