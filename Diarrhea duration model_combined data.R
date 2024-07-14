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
library(Boruta)
library(ROSE)
library(DMwR)
library(caretEnsemble)
library(pROC)
library(gridExtra)
library(grid)
library(ggplotify)

#install.packages("Amelia")
#install.packages("neuralnet")



drh_duration <- read_dta("C:/IEIP/School/PhD Thesis/Analysis/Data/VIDA_GEMS_anal.dta")


#filetring predictors that had a chi-square of P<=0.2
drh_duration1 <- drh_duration %>% select(agegroup, status_enroll_ord_collap1, yng_children_gt2, breast_feed, vs_dur_drh,
                                         f4a_drh_stools, vs_daily_max, f4a_any_vomit, f4a_drh_bellypain, f4a_drh_cough,
                                         f4a_drh_breath, f4a_cur_thirsty, f4a_cur_fastbreath, f4a_hometrt_ors, resp,
                                         f4b_mental, f4b_rectal, under_nutr, skin_flaky, fuel_clean, any_ab, rotavax,
                                         trt_zinc, base_wasting, base_malnut, vs_score,
                                         diarr_type_bin) %>%
  mutate(diarr_type_bin=factor(diarr_type_bin,levels=c(1, 0), labels=c("Yes", "NO"))) %>%
  filter(!is.na( diarr_type_bin )) 
  #select(-improv_water, -improv_sanit)


prop.table(table(drh_duration1$diarr_type_bin))

#visualize the missing data
missmap(drh_duration1)

#Use mice package to impute missing values

drh_duration_miss <- drh_duration1 %>%
  select(rotavax,breast_feed, f4a_drh_bellypain,f4a_cur_thirsty,base_wasting,trt_zinc,base_malnut, fuel_clean)
         

mice_mod <- mice(drh_duration_miss, method='rf')
mice_complete <- complete(mice_mod)

#Transfer the predicted missing values into the main data set
drh_duration1$rotavax <- mice_complete$rotavax
drh_duration1$breast_feed <- mice_complete$breast_feed
drh_duration1$f4a_drh_bellypain <- mice_complete$f4a_drh_bellypain
drh_duration1$f4a_cur_thirsty <- mice_complete$f4a_cur_thirsty
drh_duration1$base_wasting <- mice_complete$base_wasting
drh_duration1$trt_zinc <- mice_complete$trt_zinc
drh_duration1$base_malnut <- mice_complete$base_malnut
drh_duration1$fuel_clean <- mice_complete$fuel_clean


missmap(drh_duration1)

table(drh_duration1$rotavax)
#**********************************************Feature selection using Boruta******************************************
# Perform Boruta search
boruta_output <- Boruta(diarr_type_bin ~ ., data=na.omit(drh_duration1), doTrace=1)  
# Get significant variables including tentatives
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

# Do a tentative rough fix
#roughFixMod <- TentativeRoughFix(boruta_output1)
#boruta_signif1 <- getSelectedAttributes(roughFixMod)
#print(boruta_signif1)

# Variable Importance Scores
imps <- attStats(boruta_output)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])  # descending sort

# Plot variable importance
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Diarrhea Duration Feature Selection")  


row.has.na <- apply(drh_duration1, 1, function(x){any(is.na(x))})
drh_duration2_no_NA <- drh_duration1[!row.has.na, ]
prop.table(table(drh_duration2_no_NA$diarr_type_bin))

#subsetting variables from boruta
drh_duration2 <- drh_duration2_no_NA %>% select(vs_dur_drh,rotavax,agegroup,breast_feed,base_malnut,trt_zinc,vs_daily_max,resp,
               base_wasting,f4a_any_vomit,under_nutr,skin_flaky,vs_score,status_enroll_ord_collap1,
               f4a_drh_bellypain,f4a_hometrt_ors,yng_children_gt2,diarr_type_bin)

#partitioning data
test_index <- createDataPartition(drh_duration2$diarr_type_bin, times = 1, p = 0.75, list = FALSE)

train_set_drh  <- drh_duration2[test_index, ]
test_set_drh <- drh_duration2[-test_index, ]

#*************************************************************************************************************************
#*************************************************************************************************************************

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


##*********************************************************************************************************
#### Random Forest
set.seed(1234)

# Run the  default model  model using F1
rf_default <- train(diarr_type_bin ~ ., method = "rf", data=train_set_drh, metric= "F1",
                    trControl = trControl)

# Run the  default model using ROC
rf_default <- train(diarr_type_bin ~ ., method = "rf", data=train_set_drh, metric= "ROC",
                    trControl = trControl1)

print(rf_default)

#model validation
rf_prediction <-predict(rf_default, test_set_drh,type = "raw")
confusionMatrix(rf_prediction, test_set_drh$diarr_type_bin)

varImp(rf_default)
FI_val <- F_meas(data = rf_prediction, reference = factor(test_set_drh$diarr_type_bin))
FI_val


#*************************************************************************************************************************
#*************************************************************************************************************************
#Gradient Boosting Algorithm

# Fit the model on the training set

set.seed(12345)
gbm_dur_model <- train( diarr_type_bin ~ ., 
                          data = train_set_drh, method = "xgbTree", shrinkage = 0.01,
                          trControl = trControl,metric= "F1")
print(gbm_dur_model)

gbm_prediction <-predict(gbm_dur_model, test_set_drh,type = "raw")
confusionMatrix(gbm_prediction, test_set_drh$diarr_type_bin)
varImp(gbm_dur_model)

#*************************************************************************************************************************
#*************************************************************************************************************************
#Naive Bayes Algorithm (sen=77, Sep=44, F1=55)

set.seed(1234)
nb_dur_model <- train(
  diarr_type_bin ~ ., 
  data = train_set_drh, method = "naive_bayes",
  trControl = trControl, metric= "F1")
print(nb_dur_model)

nb_prediction <-predict(nb_dur_model, test_set_drh,type = "raw")
confusionMatrix(nb_prediction, test_set_drh$diarr_type_bin)
varImp(nb_dur_model)

#*************************************************************************************************************************
#*************************************************************************************************************************
#Logistic Regression Algorithm (sen=77, Sep=44, F1=55)

set.seed(1234)
glm_dur_model <- train(
  diarr_type_bin ~ ., 
  data = train_set_drh, method = "glm",
  trControl = trControl,metric= "F1")

#check for variable importance
glm_prediction <-predict(glm_dur_model, test_set_drh,type = "raw")
confusionMatrix(glm_prediction, test_set_drh$diarr_type_bin)

########################################################################################################
#Run the kNN Algorithm 
set.seed(1234)
dur_knn <- train(diarr_type_bin ~ ., 
                   data = train_set_drh, method = "knn",
                   trControl = trControl,metric= "F1")

#check for variable importance
knn_prediction <-predict(dur_knn, test_set_drh,type = "raw")
confusionMatrix(knn_prediction, test_set_drh$diarr_type_bin)
varImp(RSV_knn)

######################################################################################################################
#Run the nueralnet Algorithm 
set.seed(7777)
dur_nn <- train(diarr_type_bin ~ ., 
                  data = train_set_drh, method = "nnet",
                  trControl = trControl1,metric= "ROC")

#check for variable importance
nn_prediction <-predict(dur_nn, test_set_drh,type = "raw")
confusionMatrix(nn_prediction, test_set_drh$diarr_type_bin)


#########################################################################################################################
#**************************************************Stacking approach 2**************************************************
set.seed(7777)

control_stacking <- trainControl(method = "cv",
                                 number = 10,
                                 summaryFunction=f1,
                                 savePredictions = 'final', # To save out of fold predictions for best parRSVeter combinantions
                                 classProbs = T, # To save the class probabilities of the out of fold predictions,
                                 sampling = "down")


algorithms_to_use <- c('rf','nnet', "nb","xgbTree","svmLinear") # "xgbTree", 'knn'

dur_stacked_models <- caretList(diarr_type_bin ~ ., data=train_set_drh,
                                trControl=control_stacking, 
                                methodList=algorithms_to_use)

dur_stacking_results <- resamples(dur_stacked_models)

summary(dur_stacking_results)



# stack using glm
stackControl <- trainControl(method = "cv",
                             number = 10,
                             
                             savePredictions = 'final', # To save out of fold predictions for best parRSVeter combinantions
                             classProbs = T,  # To save the class probabilities of the out of fold predictions,
                             sampling = "down")

fitGrid_2 <- expand.grid(mfinal = (1:3)*3,         
                         maxdepth = c(1, 3),      
                         coeflearn = c("Breiman"))

set.seed(777)
dur_glm_stack <- caretStack(dur_stacked_models, method="glm", metric="Accuracy", trControl=stackControl)
dur_gbm_stack <- caretStack(dur_stacked_models, method="gbm", metric="Accuracy", trControl=stackControl)

save.image("C:/IEIP/DATA MANAGEMENT/Analysis/RSV prediction ML/Data/rsv_glm_stack.RData")
#print(rsv_glm_stack)

#predict using logictic regression top layer model
dur_stack_prediction <-predict(dur_glm_stack, test_set_drh,type = "raw")
confusionMatrix(dur_stack_prediction, test_set_drh$rsv_result)

dur_stack_prediction <-predict(dur_gbm_stack, test_set_drh,type = "raw")
confusionMatrix(dur_stack_prediction, test_set_drh$rsv_result)

