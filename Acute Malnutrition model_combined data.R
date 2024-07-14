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

#library(h2o)

#install.packages("h2o")
#install.packages("adabag")



drh_duration <- read_dta("C:/IEIP/School/PhD Thesis/Analysis/Data/VIDA_GEMS_anal.dta")


#filetring predictors that had a chi-square of P<=0.2
drh_AM <- drh_duration %>% select(agegroup,ppl_sleep_gt4,yng_children_gt2,breast_feed,vs_daily_max,f4a_drh_cough,
                                         f4a_cur_restless,f4a_hometrt_ors,resp,f4b_eyes,f4b_skin_,f4b_mental,f4b_rectal,
                                         bipedal,abn_hair,under_nutr,skin_flaky,f4b_admit,f4b_outcome_dys,f4b_outcome_pneu,
                                         f4b_outcome_mlnt,any_ab,rotavax,trt_zinc,trt_iv,base_stunting,base_wasting,
                                         base_malnut,fever,wat_basic,no_san,san_share_3,piped_wat,fuel_clean
                                  ,who_dehyd,SAM_end,SAM) %>%
  mutate(SAM_end=factor(SAM_end,levels=c(1, 0), labels=c("Yes", "NO"))) %>% 
  filter(!is.na( SAM_end )) %>%
  select(-SAM)

#*************************************************************************************************************************
#*************************************************************************************************************************
#Random Forest Algorithm

# Define the control
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- F1_Score(y_pred  = data$pred, y_true = data$obs, positive = lev[1])
  c(F1 = f1_val)
}

trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid",
                          summaryFunction=f1,
                          classProbs = TRUE)

trControl1 <- trainControl(method = "cv",
                           number = 10,
                           search = "grid",
                           summaryFunction=twoClassSummary,
                           classProbs = TRUE)


#visualize the missing data
missmap(drh_AM)

#Use mice package to impute missing values

drh_AM_miss <- drh_AM %>%
  select(breast_feed,rotavax,base_wasting,base_stunting,trt_zinc,base_malnut,fuel_clean,f4b_admit,
         f4b_skin_,f4b_outcome_dys,f4b_outcome_pneu,f4b_outcome_mlnt,trt_iv,f4a_cur_restless)


mice_mod <- mice(drh_AM_miss, method='rf')
mice_complete <- complete(mice_mod)

#Transfer the predicted missing values into the main data set
drh_AM$breast_feed <- mice_complete$breast_feed
drh_AM$rotavax <- mice_complete$rotavax
drh_AM$base_wasting <- mice_complete$base_wasting
drh_AM$base_stunting <- mice_complete$base_stunting
drh_AM$trt_zinc <- mice_complete$trt_zinc
drh_AM$base_malnut <- mice_complete$base_malnut
drh_AM$fuel_clean <- mice_complete$fuel_clean
drh_AM$f4b_admit <- mice_complete$f4b_admit
drh_AM$f4b_skin_ <- mice_complete$f4b_skin_
drh_AM$f4b_outcome_dys <- mice_complete$f4b_outcome_dys
drh_AM$f4b_outcome_pneu <- mice_complete$f4b_outcome_pneu
drh_AM$f4b_outcome_mlnt <- mice_complete$f4b_outcome_mlnt
drh_AM$trt_iv <- mice_complete$trt_iv
drh_AM$f4a_cur_restless <- mice_complete$f4a_cur_restless

missmap(drh_AM)

row.has.na <- apply(drh_AM, 1, function(x){any(is.na(x))})
drh_AM_no_NA <- drh_AM[!row.has.na, ]


# Perform Boruta search
boruta_output <- Boruta(SAM_end ~ ., data=na.omit(drh_AM_no_NA), doTrace=1)  
# Get significant variables including tentatives
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

# Do a tentative rough fix
roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

# Variable Importance Scores
imps <- attStats(boruta_output)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])  # descending sort

# Plot variable importance
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  

#partioning data
test_index1 <- createDataPartition(drh_AM_no_NA$SAM_end, times = 1, p = 0.75, list = FALSE) 

train_set_no_NA <- drh_AM_no_NA[test_index1, ]
test_set_no_NA <- drh_AM_no_NA[-test_index1, ]


#running default model
set.seed(1234)
# Run the  default model using F1
rf_default_AM <- train(SAM_end ~ ., method = "rf", data=train_set_no_NA, metric= "F1",
                    trControl = trControl)
print(rf_default_AM)
# Run the  default model using ROC
set.seed(1234)
rf_default_AM1 <- train(SAM_end ~ ., method = "rf", data=train_set_no_NA, metric= "ROC",
                    trControl = trControl1)

print(rf_default_AM1)

#model validation
rf_prediction <-predict(rf_default_AM, test_set_no_NA,type = "raw")
confusionMatrix(rf_prediction, test_set_no_NA$SAM_end)

varlist<-as.data.frame(varImp(rf_default_AM)[1]) 
varlist<-varlist[order(varlist$Overall),]

varImp(rf_default_AM)
FI_val <- F_meas(data = rf_prediction, reference = factor(test_set_no_NA$SAM_end))

#re-running model with predictor with importance >10%

set.seed(1234)
# Run the  default model using F1
rf_default_AM1 <- train(SAM_end ~ (base_malnut+resp+base_stunting+base_wasting+breast_feed+
                                    f4b_mental+vs_daily_max+san_share_3+agegroup+ppl_sleep_gt4+
                                     who_dehyd+f4a_cur_restless+yng_children_gt2+any_ab+
                                     trt_zinc+f4a_drh_cough
                                     ), method = "rf", data=train_set_no_NA, metric= "F1",
                       trControl = trControl)
print(rf_default_AM1)
#  
#model validation
rf_prediction <-predict(rf_default_AM1, test_set_no_NA,type = "raw")
confusionMatrix(rf_prediction, test_set_no_NA$SAM_end)
FI_val

#*************************************************************************************************************************
#*************************************************************************************************************************
#Gradient Boosting Algorithm

# Fit the model on the training set
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "random",
                          summaryFunction=f1,
                          classProbs = TRUE)
set.seed(12345)
gbm_AM_model <- train( SAM_end ~ ., 
  data = train_set_no_NA, method = "xgbTree", shrinkage = 0.01,
  trControl = trControl,metric= "F1")


#print(gbm_AM_model)
gbm_prediction <-predict(gbm_AM_model, test_set_no_NA,type = "raw")
confusionMatrix(gbm_prediction, test_set_no_NA$SAM_end)

FI_val <- F_meas(data = gbm_prediction, reference = factor(test_set$diarr_type_bin))
FI_val


#*************************************************************************************************************************
#*************************************************************************************************************************
#Naive Bayes Algorithm (sen=77, Sep=44, F1=55)

set.seed(1234)
nb_AM_model <- train(
  SAM_end ~ ., 
  data = train_set_no_NA, method = "nb",
  trControl = trControl,metric= "F1")
print(nb_AM_model)

#check for variable importance
nb_prediction <-predict(nb_AM_model, test_set_no_NA,type = "raw")
confusionMatrix(nb_prediction, test_set_no_NA$SAM_end)


#*************************************************************************************************************************
#*************************************************************************************************************************
#SVM Algorithm
set.seed(123)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
svm_AM_model1 = train(formula = SAM_end ~ ., 
                   data = train_set_no_NA, 
                   method="svmLinear",
                   trControl=trctrl,
                   preProcess = c("center", "scale"),
                   tuneLength = 10
                   ) 

set.seed(1234)
svm_AM_model = svm(formula = SAM_end ~ ., 
                   data = train_set_no_NA, 
                   kernel = 'linear') 

print(svm_AM_model1)
svm_prediction <-predict(svm_AM_model, test_set_no_NA,type = "raw")
confusionMatrix(svm_prediction, test_set_no_NA$SAM_end)

#*************************************************************************************************************************
#*************************************************************************************************************************
#ANN Algorithm
set.seed(1234)
ann_AM_model = neuralnet(formula = SAM_end ~ ., 
                      data = train_set_no_NA, 
                      linear.output=TRUE, 
                      likelihood = TRUE,
                      hidden = 3) 
print(ann_AM_model)
ann_AM_model$result.matrix
plot(ann_AM_model)

set.seed(1234)
nnet_model <- train(formula = SAM_end ~ ., 
                    data = train_set_no_NA, 
                    method = "nnet",
                    trControl = trControl,metric= "F1",
                    preProcess=c("scale","center"),
                    na.action = na.omit
)
#pred_neuralnet <-predict(ann_model, test_set,type = "class")

test_set1 <- test_set_no_NA  %>% select(-SAM_end)
pred_neuralnet<-compute(ann_AM_model,test_set1)
pred_neuralnet$net.result
ann_prediction1 <- round((pred_neuralnet$net.result[,1]),0)
ann_prediction2 <- round((pred_neuralnet$net.result[,2]),0)

results <- data.frame(actual = test_set_no_NA$SAM_end, prediction1 = ann_prediction1, prediction2 = ann_prediction2)

results1 <- results %>%
  mutate(prediction1=factor(prediction1, levels=c(1, 0), labels=c("Yes", "NO")),
         prediction2=factor(prediction2, levels=c(1, 0), labels=c("Yes", "NO")))

confusionMatrix(results1$prediction1,  results1$actual) 
confusionMatrix(results1$prediction2,  results1$actual) 

FI_val <- F_meas(data = svm_prediction, reference = factor(test_set_no_NA$SAM_end))
FI_val

#*************************************************************************************************************************
#Enmseble model:stacking

#Defining the training control
fitControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction=f1,
  savePredictions = 'final', # To save out of fold predictions for best parameter combinantions
  classProbs = T # To save the class probabilities of the out of fold predictions
)



#Generate base models for ensemble 

set.seed(1234)
# Run the  random forest  model using F1
AM_rf <- train(x=train_set_no_NA[,-36],y=train_set_no_NA$SAM_end, method = "rf", metric= "F1",
                       trControl = fitControl, tuneLength = 5
               )

# Run the  random forest  model using F1
set.seed(1234)
AM_gbm <- train(x=train_set_no_NA[,-36],y=train_set_no_NA$SAM_end, method = "xgbTree", metric= "F1",
                       trControl = fitControl, tuneLength = 5)

#Run the Naive Bayes Algorithm 
set.seed(1234)
AM_nb <- train(x=train_set_no_NA[,-36],y=train_set_no_NA$SAM_end, method = "nb", metric= "F1",
                trControl = fitControl, tuneLength = 5)

#Run the nueralnet Algorithm 
set.seed(1234)
AM_nn <- train(x=train_set_no_NA[,-36],y=train_set_no_NA$SAM_end, method = "nnet", metric= "F1",
               trControl = fitControl, tuneLength = 5)

#Predicting the out of fold prediction probabilities for training data
train_set_no_NA$OOF_pred_rf<-AM_rf$pred$Y[order(AM_rf$pred$rowIndex)]
train_set_no_NA$OOF_pred_gbm<-AM_gbm$pred$Y[order(AM_gbm$pred$rowIndex)]
train_set_no_NA$OOF_pred_nb<-AM_nb$pred$Y[order(AM_nb$pred$rowIndex)]
train_set_no_NA$OOF_pred_nn<-AM_nn$pred$Y[order(AM_nn$pred$rowIndex)]

#Predicting probabilities for the test data
test_set_no_NA$OOF_pred_rf<-predict(AM_rf,test_set_no_NA[,-36],type='prob')$Y
test_set_no_NA$OOF_pred_gbm<-predict(AM_gbm,test_set_no_NA[,-36],type='prob')$Y
test_set_no_NA$OOF_pred_nb<-predict(AM_nb,test_set_no_NA[,-36],type='prob')$Y
test_set_no_NA$OOF_pred_nn<-predict(AM_nn,test_set_no_NA[,-36],type='prob')$Y


#****************Creating Top layer model-level1*********************************************
#Predictors for top layer models 
predictors_top<-c('OOF_pred_rf','OOF_pred_gbm', 'OOF_pred_nb','OOF_pred_nn') 

#visualize the missing data
missmap(train_set_no_NA)

#Use mice package to impute missing values

train_set_no_NA_miss <- train_set_no_NA %>%
  select(OOF_pred_nb)


mice_mod <- mice(train_set_no_NA_miss, method='rf')
mice_complete <- complete(mice_mod)

#Transfer the predicted missing values into the main data set
train_set_no_NA$OOF_pred_nb <- mice_complete$OOF_pred_nb

#GBM as top layer model 
set.seed(1234)
model_gbm<- 
  train(x=train_set_no_NA[,predictors_top],y=train_set_no_NA$SAM_end,method='gbm',trControl=fitControl,tuneLength=5, metric= "F1")


#Logistic as top layer model 
set.seed(1234)
model_glm<- 
  train(train_set_no_NA[,predictors_top],y=train_set_no_NA$SAM_end,method='glm',trControl=fitControl,tuneLength=5, metric= "F1")

#predict using GBM top layer model
test_set_no_NA$gbm_stacked<-predict(model_gbm,test_set_no_NA[,predictors_top])

#predict using logictic regression top layer model
test_set_no_NA$glm_stacked<-predict(model_glm,test_set_no_NA[,predictors_top])

confusionMatrix(test_set_no_NA$glm_stacked, test_set_no_NA$SAM_end)
confusionMatrix(test_set_no_NA$gbm_stacked, test_set_no_NA$SAM_end)