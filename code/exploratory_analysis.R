# loading in data "full.data.for.models.Rdata", exploratory analysis to get info on number and location
# of facilities, mapping facility locations, correlation matrix of different factors with inspections,
# basic analysis (logistic regression) to predict violations with accuracy of results.

library(ggplot2)
library(maps)
library(mapdata)
library(glmnet)
load("full.data.for.models.Rdata")

violations <- random.full[random.full$DV == 1,]

usa <- map_data('usa')
state <- map_data("state") 

ggplot(data = state, aes(x=long, y=lat, group = group)) + 
  geom_polygon(fill='lightblue', color = 'white') + 
  #geom_point(data = random.full, aes(x = FAC_LONG, y = FAC_LAT, color = "green"))+
  geom_point(data = violations, aes(x = FAC_LONG, y = FAC_LAT, group= NA),size = 1,color = "red")+
  xlim(c(-130, -60))+
  ylim(25, 50)

##### split into train/test groups ------------------------------------------
smplmain <- sample(nrow(random.full),
                   round(8 * nrow(random.full) / 10),
                   replace = FALSE)

train.random <- random.full[smplmain, ]
test.random <- random.full[-smplmain, ] #this is the 20% hold-out

##### Write function for calculating risk scores -------------------------------

# this function runs logit, lasso, elastic net, single tree, and regression forest
# outputs of all models are returned all as a list

getPropensity <- function(Wvar, data, covariates, numTrees) {
  # data = the data frame to be used
  # covariates = vector of covariates for the eqution
  # Wvar = name of the treatment variable
  # numTrees = the number of trees for the gradient forest
  
  
  # Create propensity score equation
  sumx <-  paste(covariates, collapse = " + ") 
  
  propensity.eq <- paste(Wvar, paste(sumx, sep = " + "), sep = " ~ ")
  propensity.eq <- as.formula(propensity.eq)
  
  interx <- paste(" (",sumx, ")^2", sep="")  # "(X1 + X2 + X3 + ...)^2" 
  lasso.eq <- paste(Wvar, paste(interx, sep = " + "), sep = " ~ ")
  lasso.eq <- as.formula(lasso.eq)
  
  # Standard Logit
  prop.logit <- glm(propensity.eq, family = binomial, data = data)
  
  # LASSO
  lasso.model.matrix <- model.matrix(lasso.eq, data)[,-1]
  W <- as.matrix(data[,Wvar], ncol = 1)
  lasso <- cv.glmnet(lasso.model.matrix, W,  
                     alpha = 1, family = 'binomial', nfolds = 10)
  
  # Elastic Net
  elastic.net <- cv.glmnet(lasso.model.matrix, W,  
                           alpha = 0.5, family = 'binomial', nfolds = 10)
  
  # Single Tree
  singletree <- rpart(formula = propensity.eq, data = data, 
                      method = "class", y = TRUE,
                      control = rpart.control(cp = 1e-04, minsplit = 30))
  
  # prune the tree
  op.index <- which.min(singletree$cptable[, "xerror"])
  cp.vals <- singletree$cptable[, "CP"]
  treepruned <- prune(singletree, cp = cp.vals[op.index])
  
  # Regression Forest
  forest.model.matrix <- model.matrix(propensity.eq, data)[,-1]
  reg.forest <- regression_forest(forest.model.matrix, W, 
                                  num.trees = numTrees, 
                                  ci.group.size = 4, honesty = TRUE)
  
  
  results <- list(propensity.eq, lasso.eq, prop.logit, lasso, elastic.net, 
                  treepruned, reg.forest)
  names <- c("propensity.eq","lasso.eq", "logit", "lasso", "elasticnet", "tree", "forest")
  names(results) <- names
  return(results)
}


##### Load function for analyzing models + producing plots/tables --------------------------------------

# this function takes in the test DF (with predicted values) and a sample of data (A, B or C)
# and computes the MSEs, outputs the table, produces all of the sensitivity, specificity, AUC
# statistics and corresponding output, and then spits out the lowest MSE method at the end

modelAnalysis <- function(test, sample = c("A1", "A2", "A3", "A4", "A5", 
                                           "B1", "B2","B3","B4","B5",
                                           "C1", "C2", "C3", "C4", "C5"), 
                          threshold = 0.5){
  
  
  ##### Write function to compute residuals & MSEs for all 5 methods --------------------------------
  
  #This function computes the MSE based on predicted and actual outcomes
  getMSE <- function(predicted, actual) {
    MSE <- mean((actual - predicted)^2)
    results <- list(residuals, MSE)
    return(MSE)
  }
  
  pred.violations <- list(test$pscore.logit,
                          test$pscore.lasso,
                          test$pscore.en,
                          test$pscore.tree,
                          test$pscore.forest)
  
  MSElist <- lapply(pred.violations, getMSE, test$DV) 
  
  ##### Combine MSEs into a table for export -------------------------------------------
  
  MSE.table <- do.call(rbind, MSElist)
  rnames <- c("Logit", "Lasso", "Elastic Net", "Single Tree",
              "Regression Forest")
  rownames(MSE.table) = rnames
  MSE <- xtable(MSE.table, align = "lc", digits = c(0,5))
  label(MSE) <- "tab:MSE"
  caption(MSE) <- "MSE Comparison by Method"
  print(MSE, file = paste("./cwa/output/MSE_", sample, ".tex", sep = ""), caption.placement = "top", 
        sanitize.text.function = function(x){x}, include.colnames = F, include.rownames = T, 
        booktabs = T, label = "tab:MSE")
  
  
  ##### Sensitivity and specificity for all methods ---------------------------------------------------
  
  # First generate 0/1 values based on predictions. 
  test <- test %>%
    mutate(logit.pred = as.logical(ifelse(pscore.logit > threshold, 1, 0)),
           lasso.pred = as.logical(ifelse(pscore.lasso > threshold, 1, 0)),
           en.pred = as.logical(ifelse(pscore.en > threshold, 1, 0)),
           tree.pred = as.logical(ifelse(pscore.tree > threshold, 1, 0)),
           forest.pred = as.logical(ifelse(pscore.forest > threshold, 1, 0)))
  
  binary.predictions <- list("logit" = test$logit.pred,
                             "lasso" = test$lasso.pred,
                             "elastic net" = test$en.pred,
                             "single tree" = test$tree.pred,
                             "regression forest" = test$forest.pred)
  
  # Compute Confusion Matrix and Key Statistics 
  
  # Plot all ROC curves
  gg_color_hue <- function(n) {
    hues = seq(15, 375, length = n + 1)
    hcl(h = hues, l = 65, c = 100)[1:n]
  }
  
  n = 6
  cols = gg_color_hue(n)
  
  pdf(paste("./cwa/output/figs/roc_", sample, ".pdf", sep = ""))
  roc.curve <- plot(roc(test$DV, test$pscore.logit), print.auc = F, 
                    col = cols[1], print.auc.y = .4)
  roc.curve <- plot(roc(test$DV, test$pscore.lasso), print.auc = F, 
                    col = cols[2], print.auc.y = .4, add = TRUE)
  roc.curve <- plot(roc(test$DV, test$pscore.en), print.auc = F, 
                    col = cols[3], print.auc.y = .4, add = TRUE)
  roc.curve <- plot(roc(test$DV, test$pscore.tree), print.auc = F, 
                    col = cols[4], print.auc.y = .4, add = TRUE)
  roc.curve <- plot(roc(test$DV, test$pscore.forest), print.auc = F, 
                    col = cols[5], print.auc.y = .4, add = TRUE)
  legend(0.25,0.45, legend = c("Logit", "LASSO", "Elastic Net", "Single Tree", "Regression Forest"), 
         col = cols[1:5], lty = 1, cex = 0.75, bty = "n", lwd= c(2.5, 2.5))
  
  dev.off()
  
  
  ##### Table of Sensitivity, Specificity, AUC values -------------------------
  
  # Write function to get all of the statistics from the confusion matrix
  
  confusionMatrix <- function(pred, actual) {
    con.mat <- as.matrix(table(actual, pred))
    TP <- con.mat[2,2]
    TN <- con.mat[1,1]
    FP <- con.mat[1,2]
    FN <- con.mat[2,1]
    TPR <- TP/(TP + FN)
    TNR <- TN/(TN + FP)
    PPV <- TP/(TP + FP)
    NPV <- TN/(TN + FN)
    FNR <- 1 - TPR
    FPR <- 1 - TNR
    FDR <- 1 - PPV
    FOR <- 1 - NPV
    ACC <- (TP + TN)/(TP + TN + FP + FN)
    results <- list("Confusion Matrix" = con.mat,
                    TP = TP,
                    TN = TN,
                    FP = FP,
                    FN = FN,
                    TPR = TPR,
                    TNR = TNR,
                    PPV = PPV,
                    NPV = NPV,
                    FNR = FNR,
                    FDR = FDR,
                    FOR = FOR,
                    ACC = ACC)
    return(results)
  }
  
  
  confusion <- lapply(binary.predictions, confusionMatrix, 
                      actual = test$DV) 
  
  
  # get AUC
  getAUC <- function(pred, actual) {
    auc(response = actual, predictor = pred)
  }
  
  auc <- sapply(pred.violations, function(predictions) auc(test$DV, predictions))
  
  # extract sensitivity, specificity, and accuracy from confusion matrix output
  sensitivity <- paste0(round(100*unlist(sapply(confusion, "[", "TPR")), 1), "\\%")
  specificity <- paste0(round(100*unlist(sapply(confusion, "[", "TNR")), 1), "\\%")
  accuracy <- paste0(round(100*unlist(sapply(confusion, "[", "ACC")), 1), "\\%")
  PPV <- paste0(round(100*unlist(sapply(confusion, "[", "PPV")), 1), "\\%")
  FDR <- paste0(round(100*unlist(sapply(confusion, "[", "NPV")), 1), "\\%")      
  AUC <- round(unlist(auc), 3)
  
  # combine into single table and output
  comparison <- cbind(sensitivity, specificity, accuracy, PPV, FDR, AUC)
  rnames <- c("Logit", "LASSO", "Elastic Net", "Single Tree", "Regression Forest")
  colnames(comparison) <- c("Sensitivity (TPR)", "Specificity (TNR)", "Accuracy", 
                            "Precision (PPV)", "Negative Predictive Value (NPV)", "AUC")
  rownames(comparison) <- rnames
  comparison <- xtable(comparison, align = "lcccccc", digits = c(0,0,0,0,0,0,3))
  
  
  label(comparison) <- "tab:sensitivity"
  caption(comparison) <- "Model Comparison"
  print(comparison, file =paste("./cwa/output/sensitivity_", sample, threshold, ".tex", sep = ""), caption.placement = "top", 
        sanitize.text.function = function(x){x},
        include.colnames = T, include.rownames = T, booktabs = T)
  
  # Identify and return the lowest MSE method & highest AUC method
  low.MSE <- paste("The lowest MSE method for sample", sample, "is", rnames[which.min(MSElist)])
  high.AUC <- paste("the highest AUC method for sample", sample, "is", rnames[which.max(auc)])
  return(paste(low.MSE, ", and", high.AUC))
  
}



##############################  RUN #1 ##################################################
#NOTE: this is the run that generates the model comparisons in Supplementary Table 3 (sensitivity, specificity, etc.)
#This process is repeated five times

df.marker <- 1 #this is added to the output label

#both of these need to be split into inspected/notinspected
test <- as.data.frame(test.random[test.random$cat == "INS",])
train <- as.data.frame(train.random[train.random$cat == "INS",])

#full predictions: inspected facilities in test and ALL uninspected
full.predictions <- rbind.data.frame(test.random, train.random[train.random$cat != "INS",])

# adjust output label as needed
output.label <- paste0("081218.",df.marker, ".Rdata")


# select covariates for propensity equation 

covariate.names <- c("FAC_STATE", "FAC_CHESAPEAKE_BAY_FLG", 
                     "FAC_INDIAN_CNTRY_FLG", "FAC_US_MEX_BORDER_FLG", 
                     "FAC_FEDERAL_FLG", "FAC_PERCENT_MINORITY", "FAC_POP_DEN", 
                     "AIR_FLAG", "SDWIS_FLAG", "RCRA_FLAG",
                     "TRI_FLAG", "GHG_FLAG", "FAC_IMP_WATER_FLG", 
                     "EJSCREEN_FLAG_US", "multiple_IDs","n_IDs", 
                     "SIC_AG_f", "SIC_MINE_f", "SIC_CONS_f", "SIC_MANU_f", "SIC_UTIL_f", 
                     "SIC_WHOL_f", "SIC_RETA_f", "SIC_FINA_f", "SIC_SERV_f", "SIC_PUBL_f", 
                     "num.facilities.cty", "num.facilities.st", "Party", 
                     "PERMIT_MAJOR", "PERMIT_MINOR", "time.since.insp", 
                     "prox.1yr", "prox.2yr", "prox.5yr")

propensity.scores <- getPropensity(data = train, covariates = covariate.names,      
                                   Wvar = "DV", numTrees = 400)  

#save output 
save(propensity.scores, file = paste0("./cwa/output/propensityscores.5a", output.label))

##### Predict violations on test data ----------------------------------------

# generate model matrix for test data for lasso/elastic net equation (with interactions)
lasso.model.matrix.test <- model.matrix(propensity.scores$lasso.eq, test)[,-1]

test$pscore.logit <- predict(propensity.scores$logit, test, 
                             type = "response") 

test$pscore.lasso <- predict(propensity.scores$lasso, 
                             newx = lasso.model.matrix.test, 
                             s = propensity.scores$lasso$lambda.1se, 
                             type = "response")[,1]

test$pscore.en <- predict(propensity.scores$elasticnet, 
                          newx = lasso.model.matrix.test, 
                          s = propensity.scores$elasticnet$lambda.1se, 
                          type = "response")[,1]


test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] 

# generate model matrix for test data for equation without interaction terms
propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]

test$pscore.forest <- predict(propensity.scores$forest, 
                              newdata = propensity.model.matrix.test,
                              estimate.variance = TRUE)$predictions[,1]

save(test, file = paste0("./cwa/output/test.5a", output.label))


##### run modelAnalysis -------------------------------------------------------
modelAnalysis(test, "A1", threshold = .5)

##### Predict propensity to violate on full test dataset using low.MSE method -------------
#we use forest here
lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]

full.predictions$prop <- predict(propensity.scores$forest, 
                                 newdata = lasso.model.matrix.full)$predictions[,1]


