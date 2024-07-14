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
#library(fastAdaboost)
library(adabag)
library(Boruta)
library(ROSE)
#library(DMwR)
library(caretEnsemble)
library(pROC)
library(modelplotr)
library(SuperLearner)
#library(h2o)
library(caTools)
library(epiR)
library(rminer)
library(iml)
library(DALEX)
library(ResourceSelection) #computing hosmer-lemeshow values
library(rms) #Asessing calibration:Brier score and Spiegelhalter
library(gridExtra)
library(githubinstall)
library(shapper)
library(precrec) #for computing ROC & PR AUC & CI


#install.packages('gridExtra')

#SHAP TreeSHAP



SDH_data <- read_dta("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Data/rota_SCRH_08Dec2020.dta")


#filtering predictors that had a chi-square of P<=0.2
SDH_data1 <- SDH_data %>% select(agecat,ipdtem,ipdhg,ipdpuls,ipdmvom,ipdletha,ipdmuncons,ipdmfev,ipdpneg,ipdmwgt,
                                 ipdalert,ipdresirit,ipdsunkeye,ipdskinp,ipdcaprefi,ipddrink,ipdbulgf,ipdsunkf,
                                 ipdreyes,ipdtfiv,dehydrationstatus,stunting,nutri_wasting,ipdmadmin,ipdhu,
                                 ipdchestin,ipdstrid,ipdnasal,ipdmdrhd,maxvom,vesikari_int,death) %>%
  mutate(death=factor(death,levels=c(1, 0), labels=c("Yes", "NO"))) %>%
  filter(!is.na( death )) 

#visualize the missing data
missmap(SDH_data1)

#Plotting patterns in missing data using VIM
mice_plot <- aggr(SDH_data1, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(SDH_data1), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))



#Use mice package to impute missing values

SDH_miss <- SDH_data1 %>%
  select(ipdpuls,ipdreyes,ipdsunkf,ipdletha,ipdpneg,ipdmuncons,ipdcaprefi,ipdmfev,nutri_wasting,
         ipdmdrhd,ipddrink,ipdtfiv,stunting,ipdalert,ipdresirit, ipdtem, ipdmvom,ipdsunkeye,
         ipdmadmin,ipdhu,ipdchestin,ipdstrid,ipdnasal,ipdbulgf,ipdmwgt,ipdhg,maxvom,ipdskinp) #underweight ipdlbpar,bgluco ipdadmdehy

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
SDH_data1$ipdbulgf <- mice_complete$ipdbulgf
SDH_data1$ipdmwgt <- mice_complete$ipdmwgt
SDH_data1$ipdhg <- mice_complete$ipdhg
SDH_data1$maxvom <- mice_complete$maxvom
SDH_data1$ipdskinp <- mice_complete$ipdskinp



missmap(SDH_data1)
#table(drh_duration_CM$mal_end)


row.has.na <- apply(SDH_data1, 1, function(x){any(is.na(x))})
SDH_data1_no_NA <- SDH_data1[!row.has.na, ]

save(SDH_data1_no_NA, file="C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/SDH_data1_no_NA1.Rda") 


#write_dta(RSV_no_NA, "C:/IEIP/DATA MANAGEMENT/Analysis/RSV prediction ML/Data/RSV_no_NA1.dta")

#load("C:/Users/bogwel.KEMRICDC/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/SDH_data1_no_NA.Rda")
load("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/SDH_data1_no_NA.Rda")
#load("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/SDH_data1_no_NA_old.Rda")
load("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/SDH_data1_no_NA1.Rda")



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
Mortality_features <- plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Diarrheal Mortality Feature Selection")  

Mortality_features

SDH_data2 <- SDH_data1_no_NA %>% select(-ipdtfiv,-ipdletha,-ipdreyes,-ipdcaprefi,-ipdmadmin, 
                                        -ipdhu,-ipdmwgt, -ipdbulgf,-ipddrink) #-bgluco


set.seed(1111111) #4567
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

X<- train_set_M %>% select(-death)
Y<- train_set_M %>% select(death)  %>% mutate(death=as.numeric(death))
X1 <- test_set_M %>% select(-death)
Y1 <- test_set_M %>% select(death) %>% mutate(death=as.numeric(death))

#Function for Reliability Plot
plot(c(0,1),c(0,1), col="grey",type="l",xlab = "Mean Prediction",ylab="Observed Fraction")
reliability.plot <- function(obs, pred, bins=10, scale=T) {
  # Plots a reliability chart and histogram of a set of predicitons from a classifier
  #
  # Args:
  # obs: Vector of true labels. Should be binary (0 or 1)
  # pred: Vector of predictions of each observation from the classifier. Should be real
  # number
  # bins: The number of bins to use in the reliability plot
  # scale: Scale the pred to be between 0 and 1 before creating reliability plot
  require(plyr)
  library(Hmisc)
  min.pred <- min(pred)
  max.pred <- max(pred)
  min.max.diff <- max.pred - min.pred
  if (scale) {
    pred <- (pred - min.pred) / min.max.diff
  }
  bin.pred <- cut(pred, bins)
  k <- ldply(levels(bin.pred), function(x) {
    idx <- x == bin.pred
    c(sum(obs[idx]) / length(obs[idx]), mean(pred[idx]))
  })
  is.nan.idx <- !is.nan(k$V2)
  k <- k[is.nan.idx,]
  return(k)
}



set.seed(7777777)
# Run the  default model
rf_default_M <- train(death ~ ., method = "rf", data=train_set_M, metric= "F1",
                      trControl = trControl,threshold = 0.3)
print(rf_default_M)

set.seed(1234567)
rf_default_M1 <- train(death ~ ., method = "rf", data=train_set_M, metric= "F1",
                       trControl = trControl1,threshold = 0.3)
print(rf_default_M1)

set.seed(7777777)
rf_default_M2 <- train(death ~ ., method = "rf", data=train_set_M, metric= "F1",
                       trControl = trControl2,threshold = 0.3)
print(rf_default_M2)


rf_prediction_M <-predict(rf_default_M, test_set_M,type = "raw")
confusionMatrix(rf_prediction_M, test_set_M$death)

rf_prediction_M1 <-predict(rf_default_M1, test_set_M,type = "raw")
confusionMatrix(rf_prediction_M1, test_set_M$death, mode='everything')

rf_prediction_M1a <-predict(rf_default_M1, test_set_M,type = "prob")

rf_prediction_M2 <-predict(rf_default_M2, test_set_M,type = "raw")
confusionMatrix(rf_prediction_M2, test_set_M$death)

#computing 95% CI for performance metrics
rf_data <- as.table(matrix(c(31,142,10,384), nrow = 2, byrow = TRUE))
rf_rval <- epi.tests(rf_data, conf.level = 0.95,digits = 3)
print(rf_rval)


auc(roc_rf)
ci.auc(roc_rf)

#Assessing model calibration
test_set_M1 <- test_set_M %>% 
  mutate(death=as.numeric(test_set_M$death)) %>%
  mutate(death=if_else(death==2,0,death))

val.prob(rf_prediction_M1a$Yes,as.numeric(test_set_M1$death))

varImp(rf_default_M1)
FI_val <- F_meas(data = rf_prediction_CM, reference = factor(test_no_NA$mal_end))
FI_val

roc_rf <- roc(test_set_M$death,
              predict(rf_default_M1, test_set_M, type = "prob")[,1],
              levels = rev(levels(test_set_M$death)))
roc1<-ggroc(roc_rf, colour="Red") +
  ggtitle(paste0("Random Forest ROC Curve","(AUC=", round(auc(roc_rf),digits=4), ")")) +
  theme_minimal()

roc1

## Now plot
plot(roc_rf, print.thres = c(.5), type = "S", 
     
     print.thres.cex = .8,
     legacy.axes = TRUE)

#computing shapley values using DALEX
explainer_rf <- explain( model = rf_default_M1, data = X, y = Y,label = "Random forest",type = "classification")

explainer_rf_1 <- update_data(explainer_rf, data = X1, y = Y1)


resids_rf <- model_performance(explainer_rf_1)
plot(resids_rf)

mp_rf <- model_parts(explainer_rf_1, type = "difference")
plot (mp_rf, show_boxplots = FALSE)
axis(2,labels=format(mp_rf,scientific=FALSE))

#Using shapper package using output from DALEX

# ive_rf <- shap(explainer_rf, new_observation = X1, nsamples=as.integer(200)) #, 

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

# Run the  default model
set.seed(7777777)
rf_parsimonuos <- train(death ~ (vesikari_int+ipdalert+dehydrationstatus+nutri_wasting+ipdmuncons+ipdhg+ipdpuls+ipdtem+
                                   ipdchestin+ipdnasal+ipdstrid+ipdpneg)
                                   , method = "rf", data=train_set_M, metric= "F1",
                       trControl = trControl,threshold = 0.3) #                               

set.seed(7777777)
rf_parsimonuos1 <- train(death ~ (vesikari_int+ipdalert+dehydrationstatus+nutri_wasting+ipdmuncons+ipdhg+ipdpuls+ipdtem+
                                    ipdchestin+ipdnasal+ipdstrid+ipdpneg)
                        , method = "rf", data=train_set_M, metric= "F1",
                        trControl = trControl1,threshold = 0.3) #ipdchestin ipdnasal ipdstrid ipdpneg                                

print(rf_parsimonuos1)

rf_prediction_M <-predict(rf_parsimonuos, test_set_M,type = "raw")
confusionMatrix(rf_prediction_M, test_set_M$death)

rf_prediction_M1 <-predict(rf_parsimonuos1, test_set_M,type = "raw")
confusionMatrix(rf_prediction_M1, test_set_M$death)

roc_rf <- roc(test_set_M$death,
              predict(rf_default_M1, test_set_M, type = "prob")[,1],
              levels = rev(levels(test_set_M$death)))
auc(roc_rf)


#*************************************************************************************************************************
#*************************************************************************************************************************
#Gradient Boosting Algorithm

# Fit the model on the training set

set.seed(7777777)
gbm_mortality_model <- train( death ~ ., 
                              data = train_set_M, method = "xgbTree", shrinkage = 0.01,
                              trControl = trControl1,metric= "F1")

#print(gbm_RSV_model)
gbm_prediction <-predict(gbm_mortality_model, test_set_M,type = "raw")
confusionMatrix(gbm_prediction, test_set_M$death)

gbm_prediction1 <-predict(gbm_mortality_model, test_set_M,type = "prob")

#summary(gbm_RSV_model$feature_names)
#xgb.importance(colnames(train_set_M), model = gbm_RSV_model)
varImp(gbm_mortality_model)

#computing 95% CI for performance metrics
gbm_data <- as.table(matrix(c(30,135,11,391), nrow = 2, byrow = TRUE))
gbm_rval <- epi.tests(gbm_data, conf.level = 0.95,digits = 3)
print(gbm_rval)

roc_gb <- roc(test_set_M$death,
              predict(gbm_mortality_model, test_set_M, type = "prob")[,1],
              levels = rev(levels(test_set_M$death)))
auc(roc_gb)
ci.auc(roc_gb)

roc2<-ggroc(roc_gb, colour="Red") +
  ggtitle(paste0("Gradient Boosting ROC Curve","(AUC=", round(auc(roc_gb),digits=4), ")")) +
  theme_minimal()

roc2

#Assessing model calibration
val.prob(gbm_prediction1$Yes,as.numeric(test_set_M1$death))

#computing shapley values using DALEX
explainer_gbm <- explain( model = gbm_mortality_model, data = X, y = Y,label = "Gradient Boosting",type = "classification")

explainer_gbm1 <- update_data(explainer_gbm, data = X1, y = Y1)


resids_gbm <- model_performance(explainer_gbm1)
plot(resids_gbm)

mp_gbm <- model_parts(explainer_gbm1, type = "difference")

#*************************************************************************************************************************
#*************************************************************************************************************************
#Naive Bayes Algorithm (sen=77, Sep=44, F1=55)

set.seed(7777777)
nb_mortality_model <- train(
  death ~ ., 
  data = train_set_M, method = "naive_bayes",
  trControl = trControl, metric= "F1")
print(nb_mortality_model)

#check for variable importance
nb_prediction <-predict(nb_mortality_model, test_set_M,type = "raw")
confusionMatrix(nb_prediction, test_set_M$death)
varImp(nb_mortality_model)

nb_prediction1 <-predict(nb_mortality_model, test_set_M,type = "prob")

#computing 95% CI for performance metrics
nb_data <- as.table(matrix(c(31,121,10,405), nrow = 2, byrow = TRUE))
nb_rval <- epi.tests(nb_data, conf.level = 0.95,digits = 3)
print(nb_rval)

roc_nb <- roc(test_set_M$death,
               predict(nb_mortality_model, test_set_M, type = "prob")[,1],
               levels = rev(levels(test_set_M$death)))
auc(roc_nb)
ci.auc(roc_nb)

roc3<-ggroc(roc_nb, colour="Red") +
  ggtitle(paste0("Naive Bayes ROC Curve","(AUC=", round(auc(roc_nb),digits=4), ")")) +
  theme_minimal()

roc3

#Assessing model calibration
val.prob(nb_prediction1$Yes,as.numeric(test_set_M1$death))

#computing shapley values using DALEX
explainer_nb <- explain( model = nb_mortality_model, data = X, y = Y,label = "Naive Bayes",type = "classification")

explainer_nb1 <- update_data(explainer_nb, data = X1, y = Y1)


resids_nb <- model_performance(explainer_nb1)
plot(resids_nb)

mp_nb <- model_parts(explainer_nb1, type = "difference")
 
#*************************************************************************************************************************
#*************************************************************************************************************************
#Logistic Regression Algorithm (sen=77, Sep=44, F1=55)

set.seed(7777777)
glm_Mortality_model <- train(
  death ~ ., 
  data = train_set_M, method = "glm",
  trControl = trControl,metric= "F1")

glm_prediction <-predict(glm_Mortality_model, test_set_M,type = "raw")
confusionMatrix(glm_prediction, test_set_M$death)

glm_prediction1 <-predict(glm_Mortality_model, test_set_M,type = "prob")

#computing 95% CI for performance metrics
glm_data <- as.table(matrix(c(31,123,10,403), nrow = 2, byrow = TRUE))
glm_rval <- epi.tests(glm_data, conf.level = 0.95,digits = 3)
print(glm_rval)

#ROC curve & AUC
roc_glm <- roc(test_set_M$death,
               predict(glm_Mortality_model, test_set_M, type = "prob")[,1],
               levels = rev(levels(test_set_M$death)))
auc(roc_glm)
ci.auc(roc_glm)

roc4<-ggroc(roc_glm, colour="Red") +
  ggtitle(paste0("Logistic Regresion ROC Curve","(AUC=", round(auc(roc_glm),digits=4), ")")) +
  theme_minimal()

roc4
## Now plot
plot(roc_glm, print.thres = c(.5), type = "S", 
     
     print.thres.cex = .8,
     legacy.axes = TRUE)

#Assessing model calibration
val.prob(glm_prediction1$Yes,as.numeric(test_set_M1$death))

#computing shapley values using DALEX
explainer_glm <- explain( model = glm_Mortality_model, data = X, y = Y,label = "Logistic Regression",type = "classification")

explainer_glm1 <- update_data(explainer_glm, data = X1, y = Y1)


resids_glm <- model_performance(explainer_glm1)
plot(resids_glm)

mp_glm <- model_parts(explainer_glm1, type = "difference")

########################################################################################################
#Run the SVM Algorithm 
set.seed(11111111)
Mortality_svm <- train(death ~ ., 
                       data = train_set_M, method = "svmLinear",
                       trControl = trControl2, preProcess = c("center","scale"),
                       probability=TRUE,
                       metric= "F1")
print(Mortality_svm)

set.seed(1234567)
Mortality_svm1 <- train(death ~ ., 
                       data = train_set_M, method = "svmLinear",
                       trControl = trControl, preProcess = c("center","scale"),
                       probability=TRUE,
                       metric= "F1")

set.seed(1234567)
Mortality_svm2 <- train(death ~ ., 
                       data = train_set_M, method = "svmLinear",
                       trControl = trControl1, preProcess = c("center","scale"),
                       probability=TRUE,
                       metric= "F1")


set.seed(7777777)
Mortality_svm1 <- train(death ~ ., 
                       data = train_set_M, method = "svmRadial",
                       trControl = trControl, preProcess = c("center","scale"),
                       metric= "F1")

print(Mortality_svm1)

#check for variable importance
svm_prediction <-predict(Mortality_svm, test_set_M,type = "raw")
confusionMatrix(svm_prediction, test_set_M$death)

svm_prediction1 <-predict(Mortality_svm, test_set_M,type = "prob")

svm_prediction1 <-predict(Mortality_svm1, test_set_M,type = "raw")
confusionMatrix(svm_prediction1, test_set_M$death)

svm_prediction2 <-predict(Mortality_svm2, test_set_M,type = "raw")
confusionMatrix(svm_prediction2, test_set_M$death)

#saving model
saveRDS(Mortality_svm, file="C:/Users/bogwel.KEMRICDC/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/Mortality_svm.rda")

#loading model 
Mortality_svm= readRDS("C:/Users/bogwel/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/Mortality_svm.rda")
#computing 95% CI for performance metrics
svm_data <- as.table(matrix(c(31,88,10,438), nrow = 2, byrow = TRUE))
svm_rval <- epi.tests(svm_data, conf.level = 0.95,digits = 3)
print(svm_rval)

varImp(Mortality_svm)
hltest(Mortality_svm)
hs <- hoslem.test(svm_prediction, Y1)
val.prob(svm_prediction1$Yes,as.numeric(test_set_M1$death))

svm.imp <- Importance(Mortality_svm, data=train_set_M)

roc_svm <- roc(test_set_M$death,
               predict(Mortality_svm, test_set_M, type = "prob")[,1],
               levels = rev(levels(test_set_M$death)))
auc(roc_svm)
ci.auc(roc_svm)

roc5<-ggroc(roc_svm, colour="Red") +
  ggtitle(paste0("SVM ROC Curve","(AUC=", round(auc(roc_svm),digits=4), ")")) +
  theme_minimal()

roc5


#Calculating AUPRC
svm_prediction <-predict(Mortality_svm, test_set_M,type ="prob")[,2]

svm_s1<- svm_prediction[1:189]
svm_s2 <- svm_prediction[190:378]
svm_s3 <- svm_prediction[379:567]

svm_l1 <- test_set_M$death[1:189]
svm_l2 <- test_set_M$death[190:378]
svm_l3 <- test_set_M$death[379:567]


svm_s<- join_scores(svm_s1,svm_s2,svm_s3)
svm_l <- join_labels(svm_l1,svm_l2,svm_l3)

svm_mdat <- mmdata(scores=svm_s, labels= svm_l, modnames=c('m1'), dsids=1:3)

svm_curves <- evalmod(scores=svm_prediction, labels= test_set_M$death)
svm_curves1 <- evalmod(svm_mdat)

auc(svm_curves)
auc(svm_curves1)
auc_ci(svm_curves1,alpha = 0.05, dtype = "normal")


#computing shapley values using DALEX
explainer_svm <- explain( model = Mortality_svm, data = X, y = Y,label = "SVM",type = "classification")

explainer_svm1 <- update_data(explainer_svm, data = X1, y = Y1)


resids_svm <- model_performance(explainer_svm1)
plot(resids_svm)

mp_svm <- model_parts(explainer_svm1, type = "difference")


# Computing business value of model using modelplotr

scores_and_ntiles <- prepare_scores_and_ntiles(datasets=list("train_set_M","test_set_M"),
                                               dataset_labels = list("train data","test data"),
                                               models = list("Mortality_svm"),  
                                               model_labels = list("SVM"), 
                                               target_column="death",
                                               ntiles = 100)

plot_input <- plotting_scope(prepared_input = scores_and_ntiles)

#cummulative gains 
cum_gains<-plot_cumgains(data = plot_input)
cum_gains
#Cumulative lift
cum_lift<-plot_cumlift(data = plot_input)
cum_lift

#Response plot
resp_plot<-plot_response(data = plot_input)
resp_plot
#Cumulative response plot
cum_resp<-plot_cumresponse(data = plot_input)
cum_resp

grid.arrange(cum_gains,cum_lift,resp_plot,cum_resp, ncol = 2 )
#multiple plots
plot_multiplot(data = plot_input)

##################Calibrating Model using Platt Scaling####################

set.seed(4567)
train_index_M <- createDataPartition(train_set_M$death, times = 1, p = 0.85, list = FALSE)
training  <- SDH_data2[train_index_M, ]
CV <- SDH_data2[-train_index_M, ]

#Training Model before Platt scaling
set.seed(7777777)
Mortality_svm2 <- train(death ~ ., 
                       data = training, method = "svmLinear",
                       trControl = trControl2, preProcess = c("center","scale"),
                       probability=TRUE,
                       metric= "F1")

#predicting on the cross validation dataset before Platt Scaling

svm_prediction2 <-predict(Mortality_svm2, CV,type = "raw")
confusionMatrix(svm_prediction2, CV$death)

svm_prediction2a <-predict(Mortality_svm2, CV,type = "prob")


#calculating Log Loss without Platt Scaling
LogLoss(as.numeric(CV$death),svm_prediction2a$Yes) 

# performing platt scaling on the dataset
dataframe<-data.frame(svm_prediction2a$Yes,CV$death)
colnames(dataframe)<-c("x","y")

# training a logistic regression model on the cross validation dataset
set.seed(7777777)
Mortality_svm2_log <- train(
  y ~ x, 
  data = dataframe, method = "glm",
  trControl = trControl,metric= "F1")

Mortality_svm2_log1<-glm(y~x,data = dataframe,family = binomial)

#predicting on the cross validation after platt scaling
svm_prediction_platt <-predict(Mortality_svm2_log, dataframe[-2],type = "raw")
confusionMatrix(svm_prediction_platt, CV$death)

svm_prediction_platt1a <-predict(Mortality_svm2_log1, dataframe[-2],type = "response")

# plotting reliability plots

# The line below computes the reliability plot data for cross validation dataset without platt scaling
k <-reliability.plot(as.numeric(CV$death),svm_prediction2a$Yes,bins = 5)
lines(k$V2, k$V1, xlim=c(0,1),
      ylim=c(0,1), xlab="Mean Prediction", 
      ylab="Observed Fraction",
      col="red", type="o", 
      main="Reliability Plot")

#This line below computes the reliability plot data for cross validation dataset with platt scaling
k <-reliability.plot(as.numeric(CV$death),svm_prediction_platt1a,bins = 5)
lines(k$V2, k$V1, xlim=c(0,1), ylim=c(0,1),
      xlab="Mean Prediction",
      ylab="Observed Fraction",
      col="blue", type="o",
      main="Reliability Plot")

legend("topright",lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","red"),legend = c("platt scaling","without plat scaling"))


# Predicting on the test dataset without Platt Scaling
svm_prediction3 <-predict(Mortality_svm2, test_set_M,type = "raw")
confusionMatrix(svm_prediction3, test_set_M$death)

svm_prediction3a <-predict(Mortality_svm2, test_set_M,type = "prob")

# Predicting on the test dataset using Platt Scaling
dataframe1<-data.frame(svm_prediction3a$Yes)
colnames(dataframe1)<-c("x")

svm_prediction4 <-predict(Mortality_svm2_log, dataframe1,type = "raw")
confusionMatrix(svm_prediction4, test_set_M$death)


result_test_platt<-predict(model_log,dataframe1,type="response")

########################################################################################################
#Run the kNN Algorithm 
set.seed(7777777)
Mortality_knn <- train(death ~ ., 
                       data = train_set_M, method = "knn",
                       trControl = trControl,metric= "F1")

print(Mortality_knn)

#check for variable importance
knn_prediction <-predict(Mortality_knn, test_set_M,type = "raw")
confusionMatrix(knn_prediction, test_set_M$death)

knn_prediction1 <-predict(Mortality_knn, test_set_M,type = "prob")


varImp(Mortality_knn)

#computing 95% CI for performance metrics
knn_data <- as.table(matrix(c(15,121,26,405), nrow = 2, byrow = TRUE))
knn_rval <- epi.tests(knn_data, conf.level = 0.95,digits = 3)
print(knn_rval)

roc_knn <- roc(test_set_M$death,
              predict(Mortality_knn, test_set_M, type = "prob")[,1],
              levels = rev(levels(test_set_M$death)))
auc(roc_knn)
ci.auc(roc_knn)

roc6<-ggroc(roc_knn, colour="Red") +
  ggtitle(paste0("KNN ROC Curve","(AUC=", round(auc(roc_knn),digits=4), ")")) +
  theme_minimal()

roc6

#Assessing model calibration
val.prob(knn_prediction1$Yes,as.numeric(test_set_M1$death))

#computing shapley values using DALEX
explainer_knn <- explain( model = Mortality_knn, data = X, y = Y,label = "KNN",type = "classification")

explainer_knn1 <- update_data(explainer_knn, data = X1, y = Y1)


resids_knn <- model_performance(explainer_knn1)
plot(resids_knn)

mp_knn <- model_parts(explainer_knn1, type = "difference")

######################################################################################################################
#Run the nueralnet Algorithm 
set.seed(1234567)
Mortality_nn <- train(death ~ ., 
                      data = train_set_M, method = "nnet",
                      trControl = trControl,metric= "F1")

print(Mortality_nn)

#check for variable importance
nn_prediction <-predict(Mortality_nn, test_set_M,type = "raw")
confusionMatrix(nn_prediction, test_set_M$death)

nn_prediction1 <-predict(Mortality_nn, test_set_M,type = "prob")

#computing 95% CI for performance metrics
nn_data <- as.table(matrix(c(31,152,10,374), nrow = 2, byrow = TRUE))
nn_rval <- epi.tests(nn_data, conf.level = 0.95,digits = 3)
print(nn_rval)

roc_nn <- roc(test_set_M$death,
               predict(Mortality_nn, test_set_M, type = "prob")[,1],
               levels = rev(levels(test_set_M$death)))
auc(roc_nn)
ci.auc(roc_nn)

roc7<-ggroc(roc_nn, colour="Red") +
  ggtitle(paste0("Neuralnet ROC Curve","(AUC=", round(auc(roc_nn),digits=4), ")")) +
  theme_minimal()

roc7

#Assessing model calibration
val.prob(nn_prediction1$Yes,as.numeric(test_set_M1$death))


#computing shapley values using DALEX
explainer_nn <- explain( model = Mortality_nn, data = X, y = Y,label = "ANN",type = "classification")

explainer_nn1 <- update_data(explainer_nn, data = X1, y = Y1)


resids_nn <- model_performance(explainer_nn1)
plot(resids_nn)

mp_nn <- model_parts(explainer_nn1, type = "difference")


p1<- plot (mp_rf,mp_gbm,mp_nb,mp_glm, show_boxplots = FALSE)
axis(2,labels=format(mp_rf,scientific=FALSE))

p2<- plot (mp_svm,mp_knn,mp_nn, show_boxplots = FALSE)
axis(2,labels=format(mp_svm,scientific=FALSE))

# create comparison plot of residuals for each model


gridExtra::grid.arrange(p1, p2, nrow 
                        = 1)


#Merging ROC Curves into one graph

grid.arrange(roc1,roc2,roc3,roc4,roc5,roc6,roc7, ncol = 3 )

## Now plot
plot(roc_rf,  type = "S", 
     print.thres.cex = .8,
     legacy.axes = TRUE) #print.thres = c(.5),

plot(roc_gbm,add=TRUE, col='red')
plot(roc_nb,add=TRUE, col='blue')
plot(roc_glm,add=TRUE, col='green')
plot(roc_glm,add=TRUE, col='yellow')
plot(roc_svm,add=TRUE, col='purple')
plot(roc_knn,add=TRUE, col='orange')
plot(roc_nn,add=TRUE, col='grey')


#######################################################################################################
######################Stacked Ensemble using CaretEnsemble########################################

# See available algorithms in caret
modelnames <- paste(names(getModelInfo()), collapse=',  ')
modelnames

control_stacking <- trainControl(method = "cv",
                                 number = 10,
                                 summaryFunction=f1,
                                 savePredictions = 'final', # To save out of fold predictions for best parRSVeter combinantions
                                 classProbs = T, # To save the class probabilities of the out of fold predictions,
                                 sampling = "up")


algorithms_to_use <- c('rf','nnet',"lda","gam", "naive_bayes", "knn","xgbTree","svmLinear", "adaboost") # "xgbTree", 'knn'

dth_stacked_models <- caretList(death ~ ., data=train_set_M,
                                trControl=control_stacking, 
                                methodList=algorithms_to_use)

dth_stacking_results <- resamples(dth_stacked_models)

summary(dth_stacking_results)

modelCor(resamples(dth_stacked_models))

# stack using glm
stackControl <- trainControl(method = "cv",
                             number = 10,
                             savePredictions = 'final', # To save out of fold predictions for best parRSVeter combinantions
                             classProbs = T,  # To save the class probabilities of the out of fold predictions,
                             sampling = "up")

fitGrid_2 <- expand.grid(mfinal = (1:3)*3,         
                         maxdepth = c(1, 3),      
                         coeflearn = c("Breiman"))

set.seed(777)
dur_glm_stack <- caretStack(dth_stacked_models, method="glm", metric="Accuracy", trControl=stackControl)
dur_gbm_stack <- caretStack(dth_stacked_models, method="gbm", metric="Accuracy", trControl=stackControl)

dur_glm_stack


model_preds <- lapply(dth_stacked_models, predict, newdata=test_set_M)
model_preds <- lapply(model_preds, function(x) x[,"M"])
model_preds <- data.frame(model_preds)
ens_preds <- predict(dur_glm_stack, newdata=test_set_M)
model_preds$ensemble <- ens_preds
confusionMatrix(ens_preds, test_set_M$death)

roc_ens <- roc(test_set_M$death,
              predict(dur_glm_stack, test_set_M, type = "prob")[,1],
              levels = rev(levels(test_set_M$death)))
auc(roc_ens)

caTools::colAUC(model_preds, testing$Class)

save.image("C:/IEIP/DATA MANAGEMENT/Analysis/RSV prediction ML/Data/rsv_glm_stack.RData")


########################################################################################################
##################Implementing stacked ensemble using superLearner*****************************************

listWrappers()

#splitting test and training dataset as required by superLearner
y <- train_set_M %>% select(death) %>% 
  mutate(death=as.numeric(death)) %>%
  mutate(death=if_else(death==2,0,death)) 

y <- as.numeric(unlist(y))

x <- train_set_M %>% select(-death)

y_test <- test_set_M %>% select(death)%>% 
  mutate(death=as.numeric(death)) %>%
  mutate(death=if_else(death==2,0,death)) 

y_test <- as.numeric(unlist(y_test))

x_test <- test_set_M %>% select(-death)


set.seed(7777777)

# Fit the ensemble model
model_death <- SuperLearner(y,
                      as.data.frame(x),
                      family=binomial(),
                      SL.library=list("SL.randomForest",
                                      "SL.ksvm",
                                      "SL.ipredbagg",
                                      "SL.knn",
                                      "SL.gam",
                                      "SL.lda",
                                      "SL.nnet",
                                      "SL.gbm", 
                                      "SL.bayesglm"))
# Return the model
model_death 

summary(cv.model_death)
#returns error-debug
plot(model_death )

model_death1 <- model_death[lapply(model_death,length)>0]

predictions <- predict.SuperLearner(model_death , newdata=as.data.frame(x_test), onlySL = T)

# Recode probabilities
conv.preds <- ifelse(predictions$pred>=0.5,1,0)

# Create the confusion matrix
ensemble1_pred <- confusionMatrix(conv.preds, y_test)
ensemble1_pred


#########################################################################################################
##############Implementing Stacked ensemble using H2)######################################

h2o.init()


write.csv(train_set_M, 'C:/Users/bogwel.KEMRICDC/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/train_set_M.csv', row.names = FALSE)
write.csv(test_set_M, 'C:/Users/bogwel.KEMRICDC/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/test_set_M.csv', row.names = FALSE)

# Import a sample binary outcome train/test set into H2O
train <- h2o.importFile('C:/Users/bogwel.KEMRICDC/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/train_set_M.csv')
test <- h2o.importFile('C:/Users/bogwel.KEMRICDC/OneDrive/Billy/School/PhD/Analysis/Mortality/Data/test_set_M.csv')

# Identify predictors and response
y <- "death"
x <- setdiff(names(train), y)

# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5

# Train & Cross-validate a GBM
gbm_mortality <- h2o.gbm(x = x,
                         y = y,
                         training_frame = train,
                         distribution = "bernoulli",
                         ntrees = 10,
                         max_depth = 3,
                         min_rows = 2,
                         learn_rate = 0.2,
                         nfolds = nfolds,
                         balance_classes = TRUE,
                         max_after_balance_size=5,
                         keep_cross_validation_predictions = TRUE,
                         seed = 77777777)

# Train & Cross-validate a RF
rf_mortality <- h2o.randomForest(x = x,
                                 y = y,
                                 training_frame = train,
                                 ntrees = 50,
                                 nfolds = nfolds,
                                 balance_classes = TRUE,
                                 max_after_balance_size=5,
                                 keep_cross_validation_predictions = TRUE,
                                 seed = 77777777)

# Build and train NB model:
nb_mortality <- h2o.naiveBayes(x = x,
                               y = y,
                               training_frame = train,
                               laplace = 0,
                               nfolds = nfolds,
                               balance_classes = TRUE,
                               max_after_balance_size=5,
                               keep_cross_validation_predictions = TRUE,
                               seed = 77777777)

# Build and train NN model:
nn_mortality <- h2o.deeplearning(x = x,
                                 y = y,
                                 distribution = "bernoulli",
                                 hidden = c(1),
                                 epochs = 1000,
                                 train_samples_per_iteration = -1,
                                 reproducible = TRUE,
                                 activation = "Tanh",
                                 single_node_mode = FALSE,
                                 nfolds = nfolds,
                                 balance_classes = TRUE,
                                 max_after_balance_size=5,
                                 keep_cross_validation_predictions = TRUE,
                                 force_load_balance = FALSE,
                                 seed = 77777777,
                                 tweedie_power = 1.5,
                                 score_training_samples = 0,
                                 score_validation_samples = 0,
                                 training_frame = train,
                                 stopping_rounds = 0)

#Build and train GLM model;
glm_mortality <- h2o.glm(family = "binomial",
                         x = x,
                         y = y,
                         training_frame = train,
                         lambda = 0,
                         compute_p_values = TRUE,
                         nfolds = nfolds,
                         balance_classes = TRUE,
                         max_after_balance_size=5,
                         keep_cross_validation_predictions = TRUE,
                         seed = 77777777)

svm_mortality <- h2o.psvm(gamma = 0.01,
                          rank_ratio = 0.1,
                          x = x,
                          y = y,
                          training_frame = train,
                          seed=1)


# Compare to base learner performance on the test set
perf_gbm_test <- h2o.performance(gbm_mortality, newdata = test)
perf_rf_test <- h2o.performance(rf_mortality, newdata = test)
perf_nn_test <- h2o.performance(nn_mortality, newdata = test)
perf_glm_test <- h2o.performance(glm_mortality, newdata = test)


perf_svm_test <- h2o.performance(svm_mortality, newdata = test)

perf_gbm_test
perf_rf_test
perf_nn_test
perf_glm_test
perf_svm_test
h2o.confusionMatrix(perf_gbm_test)

# Train a stacked ensemble using the GBM and RF above
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                base_models = list(gbm_mortality, rf_mortality,
                                                   nn_mortality,glm_mortality))

# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test)

perf

set.seed(7777)
