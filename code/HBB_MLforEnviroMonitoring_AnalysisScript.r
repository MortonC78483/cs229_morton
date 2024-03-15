################################################################################
# Authors: Miyuki Hino, Elinor Benami, Nina Brooks
# Paper title: Machine Learning for Environmental Monitoring
# Date last revised: August 2018
# Input files: 
# 1) fac.final.hist.Rdata
# 2) dmr.full.hist.Rdata
# 3) NPDES_INSPECTIONS.csv
# Notes: This is a series of scripts merged into one file.  Compare the file structure with yours to ensure this runs correctly on your computer.


####################################################
# 1. Generate train and test datasets
####################################################
  
    rm(list = ls(all = TRUE)) 
  
  ##### load packages -----------------------------------------------------------
    library(plyr) 
    library(dplyr)
    library(zoo)
    library(lubridate)
    library(reshape2)
    library(data.table) 
    library(readr)
  
  ##### ALL FACILITIES #####

  ##### load full dataset and adjust variables ----------------------------
    load("./cwa/data/modified_data/fac.final.hist.Rdata")
  
    #define DV
      fac.final.hist$DV <- as.numeric(!is.na(fac.final.hist$X1)) #1 for a failed inspection, 0 otherwise
    
    #set time.since.insp = median for any facility's first inspection ("99999"), or if they have never been inspected (NA)
    #calculate median based on non-first-inspections
      fac.final.hist$time.since.insp <- as.numeric(fac.final.hist$time.since.insp)
      for.median.timesinceinsp <- fac.final.hist$time.since.insp[fac.final.hist$cat == "INS" & fac.final.hist$time.since.insp != "99999"]
      median.timesinceinsp <- median(for.median.timesinceinsp)
    
    #mapvalues to median
      fac.final.hist$time.since.insp <- mapvalues(fac.final.hist$time.since.insp, from = c(NA, 99999),
                                                  to = rep(median.timesinceinsp,2))
    #confirm that median has not changed
      median(fac.final.hist$time.since.insp) == median.timesinceinsp
    
    #ensure that all non-inspected facilities have an NA for ACTUAL_END_DATE 
      fac.final.hist$ACTUAL_END_DATE[fac.final.hist$cat != "INS"] <- as.Date(NA)
  
  ###### select which inspection to use (for facilities with multiple inspections) --------
  
    fac.final.hist.insp <- fac.final.hist[cat == "INS"]
    
    #random selection
      set.seed(1234)
      random <- fac.final.hist.insp %>% group_by(REGISTRY_ID) %>% sample_n(1)
    #select most recent inspection
      recent <- fac.final.hist.insp %>% group_by(REGISTRY_ID) %>% filter(row_number() == 1)
    
    #recombine each with the non-inspected
      random.full <- rbind.data.frame(random, fac.final.hist[cat != "INS"])
      recent.full <- rbind.data.frame(recent, fac.final.hist[cat != "INS"])
      save(random.full, recent.full, file = "./cwa/data/modified_data/full.data.for.models.Rdata")
      
  
  ##### split into 5 train/test groups ------------------------------------------
  
    #divide into 80-20 splits
      smplmain <- sample(nrow(random.full),
                         round(8 * nrow(random.full) / 10),
                         replace = FALSE)
      
      train.random <- random.full[smplmain, ]
      test.random <- random.full[-smplmain, ] #this is the 20% hold-out
      save(train.random, test.random, file = "./cwa/data/modified_data/full.data.random.Rdata")
      
      train.recent <- recent.full[smplmain, ]
      test.recent <- recent.full[-smplmain, ] #this is the 20% hold-out
      save(train.recent, test.recent, file = "./cwa/data/modified_data/full.data.recent.Rdata")
    
    #smplmain is the 80% from which addtl 20% test sets need to be drawn
      indices <- integer(nrow(random.full))
      indices[-smplmain] <- 1
      indices[indices == 0] <- sample(rep(2:5, times = round(length(smplmain)/4)))
    
      train2.random <- random.full[indices != 2,]
      test2.random <- random.full[indices == 2,]
      
      train2.recent <- recent.full[indices != 2, ]
      test2.recent <- recent.full[indices == 2, ] 
      
      train3.random <- random.full[indices != 3,]
      test3.random <- random.full[indices == 3,]
      
      train3.recent <- recent.full[indices != 3, ]
      test3.recent <- recent.full[indices == 3, ] 
      
      train4.random <- random.full[indices != 4,]
      test4.random <- random.full[indices == 4,]
      
      train4.recent <- recent.full[indices != 4, ]
      test4.recent <- recent.full[indices == 4, ] 
      
      train5.random <- random.full[indices != 5,]
      test5.random <- random.full[indices == 5,]
      
      train5.recent <- recent.full[indices != 5, ]
      test5.recent <- recent.full[indices == 5, ] 
      
      save(train.random, test.random, train2.random, test2.random, train3.random, test3.random,
           train4.random, test4.random, train5.random, test5.random, file = "./cwa/data/modified_data/full.random.5sets.Rdata")
      
      save(train.recent, test.recent, train2.recent, test2.recent, train3.recent, test3.recent,
           train4.recent, test4.recent, train5.recent, test5.recent, file = "./cwa/data/modified_data/full.recent.5sets.Rdata")
      
    
  ##### DMR-SUBMITTING FACILITIES ONLY #####
      
  ##### import and clean DMR data -----------------------------------------------
  
    load("./cwa/data/modified_data/dmr.full.hist.Rdata")
    
    dmr.full.hist$DV <- as.numeric(!is.na(dmr.full.hist$X1))
    
    #set time.since.insp = median for any facility's first inspection ("99999"), or if they have never been inspected (NA)
    #calculate median based on non-first-inspections
      dmr.full.hist$time.since.insp <- as.numeric(dmr.full.hist$time.since.insp)
      dmr.for.median.timesinceinsp <- dmr.full.hist$time.since.insp[dmr.full.hist$cat == "INS" & dmr.full.hist$time.since.insp != "99999"]
      dmr.median.timesinceinsp <- median(dmr.for.median.timesinceinsp)
      
    #mapvalues to median
      dmr.full.hist$time.since.insp <- mapvalues(dmr.full.hist$time.since.insp, from = c(NA, 99999),
                                               to = rep(dmr.median.timesinceinsp,2))
    #confirm that median has not changed
      median(dmr.full.hist$time.since.insp) == dmr.median.timesinceinsp
    
    #ensure that all non-ins facilities have an NA for ACTUAL_END_DATE 
      dmr.full.hist$ACTUAL_END_DATE[dmr.full.hist$cat != "INS"] <- as.Date(NA)
    
  
  ##### select which inspection to use (for facilities with multiple insepctions)  -----------------------------------------
  
    dmr.full.hist.insp <- dmr.full.hist[cat == "INS"]
    
    #random selections
      set.seed(1234)
      random.dmr <- dmr.full.hist.insp %>% group_by(REGISTRY_ID) %>% sample_n(1)
    #most recent inspection
      recent.dmr <- dmr.full.hist.insp %>% group_by(REGISTRY_ID) %>% filter(row_number() == 1)
    
    #recombine each with the non-inspected
      random.dmr.full <- rbind.data.frame(random.dmr, dmr.full.hist[cat != "INS"])
      recent.dmr.full <- rbind.data.frame(recent.dmr, dmr.full.hist[cat != "INS"])
      save(random.dmr.full, recent.dmr.full, file = "./cwa/data/modified_data/full.dmr.data.for.models.Rdata")
      
    
  ##### divide into 5 train/test splits -----------------------------------------
  
    #divide into 80-20 
      smplmain <- sample(nrow(random.dmr.full),
                         round(8 * nrow(random.dmr.full) / 10),
                         replace = FALSE)
      
      train.random.dmr <- random.dmr.full[smplmain, ]
      test.random.dmr <- random.dmr.full[-smplmain, ] #this is the 20% hold-out
      save(train.random.dmr, test.random.dmr, file = "./cwa/data/modified_data/full.dmr.data.random.Rdata")
      
      train.recent.dmr <- recent.dmr.full[smplmain, ]
      test.recent.dmr <- recent.dmr.full[-smplmain, ] #this is the 20% hold-out
      save(train.recent.dmr, test.recent.dmr, file = "./cwa/data/modified_data/full.dmr.data.recent.Rdata")
      
    #smplmain is the 80% from which addtl 20% test sets need to be drawn
      indices.dmr <- integer(nrow(random.dmr.full))
      indices.dmr[-smplmain] <- 1
      indices.dmr[indices.dmr == 0] <- sample(rep(2:5, times = round(length(smplmain)/4)))
      
      train2.random.dmr <- random.dmr.full[indices.dmr != 2,]
      test2.random.dmr <- random.dmr.full[indices.dmr == 2,]
      
      train2.recent.dmr <- recent.dmr.full[indices.dmr != 2, ]
      test2.recent.dmr <- recent.dmr.full[indices.dmr == 2, ] 
      
      train3.random.dmr <- random.dmr.full[indices.dmr != 3,]
      test3.random.dmr <- random.dmr.full[indices.dmr == 3,]
      
      train3.recent.dmr <- recent.dmr.full[indices.dmr != 3, ]
      test3.recent.dmr <- recent.dmr.full[indices.dmr == 3, ] 
      
      train4.random.dmr <- random.dmr.full[indices.dmr != 4,]
      test4.random.dmr <- random.dmr.full[indices.dmr == 4,]
      
      train4.recent.dmr <- recent.dmr.full[indices.dmr != 4, ]
      test4.recent.dmr <- recent.dmr.full[indices.dmr == 4, ] 
      
      train5.random.dmr <- random.dmr.full[indices.dmr != 5,]
      test5.random.dmr <- random.dmr.full[indices.dmr == 5,]
      
      train5.recent.dmr <- recent.dmr.full[indices.dmr != 5, ]
      test5.recent.dmr <- recent.dmr.full[indices.dmr == 5, ] 
      
    save(train.random.dmr, test.random.dmr, train2.random.dmr, test2.random.dmr, train3.random.dmr, test3.random.dmr,
         train4.random.dmr, test4.random.dmr, train5.random.dmr, test5.random.dmr, file = "./cwa/data/modified_data/dmr.random.5sets.Rdata")
    
    save(train.recent.dmr, test.recent.dmr, train2.recent.dmr, test2.recent.dmr, train3.recent.dmr, test3.recent.dmr,
         train4.recent.dmr, test4.recent.dmr, train5.recent.dmr, test5.recent.dmr, file = "./cwa/data/modified_data/dmr.recent.5sets.Rdata")


####################################################
# 2. Run models on full dataset
####################################################
    
  # Note: this core script is repeated three times:
  # Once with the full dataset, once with DMR-submitting facilities (no DMR variables), once with DMR-submitting facilities (with DMR variables)
  
  print("primary dataset - model start")
  ##### Set Up -------
    rm(list = ls(all = TRUE)) #clean out workspace
    
    ## load packages
    library(reshape2)
    library(lars)
    library(ggplot2)
    library(plyr)
    library(sandwich)
    library(stargazer)
    library(data.table)
    library(maptools)
    library(dplyr)
    library(xtable)
    library(ROCR)
    library(Hmisc)
    library(texreg)
    library(devtools)
    library(RCurl)
    library(grf) 
    library(glmnet)
    library(rpart) # decision tree
    library(rpart.plot) # enhanced tree plots
    library(pROC) # for sensitivity and specificity
    
  
    # set seed for replicable results --------------
      set.seed(36)
    
    # Load data file----
      load("./cwa/data/modified_data/full.random.5sets.Rdata")
  
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
        test <- as.data.frame(test.random[cat == "INS"])
        train <- as.data.frame(train.random[cat == "INS"])
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test.random, train.random[cat != "INS"])
      
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
        
        
      # Save output 
        save(full.predictions, file = paste0("./cwa/output/full.predictions.5a", output.label))
        
        save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5a", output.label))
        
  ##############################  RUN #2 ##################################################
  
      df.marker <- 2 #this is added to the output label
      
      
      #both of these need to be split into inspected/notinspected
        test <- as.data.frame(test2.random[cat == "INS"])
        train <- as.data.frame(train2.random[cat == "INS"])
        
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test2.random, train2.random[cat != "INS"])
      
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
        
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
         modelAnalysis(test, "A2", threshold = .5)
      
      ##### Predict propensity to violate on full test dataset using low.MSE method -------------
        #we use forest here
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest, 
                                         newdata = lasso.model.matrix.full)$predictions[,1]
        
      
      # Save output 
      save(full.predictions, file = paste0("./cwa/output/full.predictions.5a", output.label))
      
      save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5a", output.label))
      
  ##############################  RUN #3 ##################################################
  
      df.marker <- 3 #this is added to the output label
      
      
      #both of these need to be split into inspected/notinspected
        test <- as.data.frame(test3.random[cat == "INS"])
        train <- as.data.frame(train3.random[cat == "INS"])
        
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test3.random, train3.random[cat != "INS"])
      
      
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
        
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
      
      modelAnalysis(test, "A3", threshold = .5)
      
      ##### Predict propensity to violate on full test dataset using low.MSE method -------------
        #we use forest here
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest, 
                                         newdata = lasso.model.matrix.full)$predictions[,1]
        
      
      # Save output -----------------------------------------------
        save(full.predictions, file = paste0("./cwa/output/full.predictions.5a", output.label))
      
        save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5a", output.label))
      
  ##############################  RUN #4 ##################################################
  
      df.marker <- 4 #this is added to the output label
      
      #both of these need to be split into inspected/notinspected
        test <- as.data.frame(test4.random[cat == "INS"])
        train <- as.data.frame(train4.random[cat == "INS"])
        
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test4.random, train4.random[cat != "INS"])
      
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
          
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
        modelAnalysis(test, "A4", threshold = .5)
      
      ##### Predict propensity to violate on full test dataset using low.MSE method -------------
        #we use forest here
          lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
          
          full.predictions$prop <- predict(propensity.scores$forest, 
                                           newdata = lasso.model.matrix.full)$predictions[,1]
          
      
      ##### Save output -----------------------------------------------
      save(full.predictions, file = paste0("./cwa/output/full.predictions.5a", output.label))

      save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5a", output.label))
      
  ##############################  RUN #5 ##################################################
  
      df.marker <- 5 #this is added to the output label
      
      #both of these need to be split into inspected/notinspected
        test <- as.data.frame(test5.random[cat == "INS"])
        train <- as.data.frame(train5.random[cat == "INS"])
        
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test5.random, train5.random[cat != "INS"])
      
      
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
        
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
        modelAnalysis(test, "A5", threshold = .5)
      
      ##### Predict propensity to violate on full test dataset using low.MSE method -------------
        #we use forest here
          lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
          
          full.predictions$prop <- predict(propensity.scores$forest, 
                                           newdata = lasso.model.matrix.full)$predictions[,1]
          
      
      ##### Save output -----------------------------------------------
      save(full.predictions, file = paste0("./cwa/output/full.predictions.5a", output.label))
      
      save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5a", output.label))
      
  ##### generate SI figure showing distribution of risk scores by method -----------
  
  
    # trimmed density plot (xlim = 0,0.4)
      prop.dist.trimmed <- ggplot(data = test, aes(x = pscore.logit, color = "Logit")) + 
        geom_density() + geom_density(aes(x = pscore.lasso, color = "Lasso")) +
        geom_density(aes(x = pscore.en, color = "Elastic Net")) + 
        geom_density(aes(x = pscore.tree, color = "Single Tree")) +
        geom_density(aes(x = pscore.forest, color = "Regression Forest")) + 
        theme(panel.background = element_blank(),
              panel.grid.minor = element_blank(),
              panel.grid.major = element_blank()
        ) +
        scale_color_discrete(name = "Method") + xlab("Risk score") +
        xlim(0,0.4)
      prop.dist.trimmed
    
      ggsave("./cwa/output/figs/prop_distributions_trimmed_081218.pdf")

####################################################
# 3. Run models on DMR dataset with no DMR variables
####################################################
        
  print("DMR model runs, part 1")
      
  # Set Up -------
    rm(list = ls(all = TRUE)) #clean out workspace
    
    # load packages
      library(reshape2)
      library(lars)
      library(ggplot2)
      library(plyr)
      library(sandwich)
      library(stargazer)
      library(data.table)
      library(maptools)
      library(dplyr)
      library(xtable)
      library(ROCR)
      library(Hmisc)
      library(texreg)
      library(devtools)
      library(RCurl)
      library(grf) 
      library(glmnet)
      library(rpart) # decision tree
      library(rpart.plot) # enhanced tree plots
      library(pROC) # for sensitivity and specificity
    
    # set seed for replicable results
     set.seed(36)
    
    # Load data file
      load("./cwa/data/modified_data/dmr.random.5sets.Rdata")
    
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
        
        
        
        ## Regression Forest (from Gradient Forest Package -- now called grf)
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
      
      
  ##### Write function for analyzing models + producing plots/tables --------------------------------------
  
    modelAnalysis <- function(test, sample = c("A1", "A2", "A3", "A4", "A5", 
                                               "B1", "B2","B3","B4","B5",
                                               "C1", "C2", "C3", "C4", "C5"), 
                              threshold = 0.5){
      # this function takes in the test DF (with predicted values) and a sample of data (A, B or C)
      # and computes the MSEs, outputs the table, produces all of the sensitivity, specificity, AUC
      # statistics and corresponding output, and then spits out the lowest MSE method at the end
      

      # Write function to compute residuals & MSEs for all 5 methods --------------------------------

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
      
      # Combine MSEs into a table for export -------------------------------------------
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
        
      # Sensitivity and specificty for all methods ---------------------------------------------------

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
        
      ##### Compute Confusion Matrix and Key Statistics -------------------------------------
      
        # Plot all ROC curves
          gg_color_hue <- function(n) {
            hues = seq(15, 375, length = n + 1)
            hcl(h = hues, l = 65, c = 100)[1:n]
          }
          
          n = 6
          cols = gg_color_hue(n)
          
          # Plot
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
          
      
      # Table of Sensitivity, Specificity, AUC values -------------------------

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
  
      df.marker <- 1 #this is added to the output label
      
      test <- as.data.frame(test.random.dmr[cat == "INS"])
      train <- as.data.frame(train.random.dmr[cat == "INS"]) 
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test.random.dmr, train.random.dmr[cat != "INS"])
      
      
      # adjust output label as needed 
        output.label <- paste0("081218.",df.marker, ".Rdata")
      
      # select covariates for propensity equation (NO DMR VARIABLES YET) 
      
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
        
        
        propensity.scores <- getPropensity(data = train, 
                                           covariates = covariate.names,      
                                           Wvar = "DV", numTrees = 400)  
        
        save(propensity.scores, file = paste0("./cwa/output/propensityscores.5b", output.label))
      
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
        
        
        test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] #same as tree.prob
        
        # generate model matrix for test data for equation without interaction terms
          propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]
          
          test$pscore.forest <- predict(propensity.scores$forest, 
                                        newdata = propensity.model.matrix.test,
                                        estimate.variance = TRUE)$predictions[,1]
          
          save(test, file = paste0("./cwa/output/test.5b", output.label))
        
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
      
        modelAnalysis(test, "B1", threshold = .5)
      
      ##### Predict propensity to violate on full test dataset using low.MSE --------
      
        #uses reg forest (lowest-MSE method)
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest,
                                         newdata = lasso.model.matrix.full)$predictions[,1]
        
      ##### Save output -----------------------------------------------
      save(full.predictions, file = paste0("./cwa/output/full.predictions.5b", output.label))
      
      save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5b", output.label))
      
  ##############################  RUN #2 ##################################################
  
      df.marker <- 2 #this is added to the output label
      
      test <- as.data.frame(test2.random.dmr[cat == "INS"])
      train <- as.data.frame(train2.random.dmr[cat == "INS"]) 
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test2.random.dmr, train2.random.dmr[cat != "INS"])
      
      
      # adjust output label as needed
        output.label <- paste0("081218.",df.marker, ".Rdata")
      
      # select covariates for propensity equation (NO DMR VARIABLES YET) 
      
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
        
        
        propensity.scores <- getPropensity(data = train, 
                                           covariates = covariate.names,      
                                           Wvar = "DV", numTrees = 400)  
        
        save(propensity.scores, file = paste0("./cwa/output/propensityscores.5b", output.label))
      
      # Predict violations on test data ----------------------------------------
      
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
        
        
          test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] #same as tree.prob
        
        # generate model matrix for test data for equation without interaction terms
          propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]
          
          test$pscore.forest <- predict(propensity.scores$forest, 
                                        newdata = propensity.model.matrix.test,
                                        estimate.variance = TRUE)$predictions[,1]
          
          save(test, file = paste0("./cwa/output/test.5b", output.label))
        
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
        modelAnalysis(test, "B2", threshold = .5)
      
      ##### Predict propensity to violate on full test dataset using low.MSE --------
      
        #uses reg forest (lowest-MSE method)
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest,
                                         newdata = lasso.model.matrix.full)$predictions[,1]
        
      ##### Save output -----------------------------------------------
      save(full.predictions, file = paste0("./cwa/output/full.predictions.5b", output.label))
      
      save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5b", output.label))
      
  
  ##############################  RUN #3 ##################################################
  
      df.marker <- 3 #this is added to the output label
      
      test <- as.data.frame(test3.random.dmr[cat == "INS"])
      train <- as.data.frame(train3.random.dmr[cat == "INS"]) 
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test3.random.dmr, train3.random.dmr[cat != "INS"])
      
      
      # adjust output label as needed 
        output.label <- paste0("081218.",df.marker, ".Rdata")
      
      # select covariates for propensity equation (NO DMR VARIABLES YET) 
      
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
        
        
        propensity.scores <- getPropensity(data = train, 
                                           covariates = covariate.names,      
                                           Wvar = "DV", numTrees = 400)  
        
        save(propensity.scores, file = paste0("./cwa/output/propensityscores.5b", output.label))
      
      # Predict violations on test data ----------------------------------------
      
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
          
          
          test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] #same as tree.prob
        
        # generate model matrix for test data for equation without interaction terms
          propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]
          
          test$pscore.forest <- predict(propensity.scores$forest, 
                                        newdata = propensity.model.matrix.test,
                                        estimate.variance = TRUE)$predictions[,1]
          
          save(test, file = paste0("./cwa/output/test.5b", output.label))
        
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
      
        modelAnalysis(test, "B3", threshold = .5)
      
      
      ##### Predict propensity to violate on full test dataset using low.MSE --------
      
        #uses reg forest (lowest-MSE method)
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest,
                                         newdata = lasso.model.matrix.full)$predictions[,1]
        
      # Save output -----------------------------------------------
      save(full.predictions, file = paste0("./cwa/output/full.predictions.5b", output.label))
      
      save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5b", output.label))
      
  ##############################  RUN #4 ##################################################
  
      df.marker <- 4 #this is added to the output label
      
      test <- as.data.frame(test4.random.dmr[cat == "INS"])
      train <- as.data.frame(train4.random.dmr[cat == "INS"]) 
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test4.random.dmr, train4.random.dmr[cat != "INS"])
      
      
      # adjust output label as needed 
        output.label <- paste0("081218.",df.marker, ".Rdata")
      
      # select covariates for propensity equation (NO DMR VARIABLES YET) ------------------------------------
      
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
        
        
        propensity.scores <- getPropensity(data = train, 
                                           covariates = covariate.names,      
                                           Wvar = "DV", numTrees = 400)  
        
        save(propensity.scores, file = paste0("./cwa/output/propensityscores.5b", output.label))
        
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
          
          
          test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] #same as tree.prob
        
        # generate model matrix for test data for equation without interaction terms
          propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]
          
          test$pscore.forest <- predict(propensity.scores$forest, 
                                        newdata = propensity.model.matrix.test,
                                        estimate.variance = TRUE)$predictions[,1]
          
          save(test, file = paste0("./cwa/output/test.5b", output.label))
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
      modelAnalysis(test, "B4", threshold = .5)
      
      ##### Predict propensity to violate on full test dataset using low.MSE --------
      
        #uses reg forest (lowest-MSE method)
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest,
                                         newdata = lasso.model.matrix.full)$predictions[,1]
        
      ##### Save output -----------------------------------------------
      save(full.predictions, file = paste0("./cwa/output/full.predictions.5b", output.label))
      
      save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5b", output.label))
      
  
  ##############################  RUN #5 ##################################################
  
      df.marker <- 5 #this is added to the output label
      
      test <- as.data.frame(test5.random.dmr[cat == "INS"])
      train <- as.data.frame(train5.random.dmr[cat == "INS"]) 
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test5.random.dmr, train5.random.dmr[cat != "INS"])
      
      
      # adjust output label as needed 
        output.label <- paste0("081218.",df.marker, ".Rdata")
      
      # select covariates for propensity equation (NO DMR VARIABLES YET) 
      
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
        
        
        propensity.scores <- getPropensity(data = train, 
                                           covariates = covariate.names,      
                                           Wvar = "DV", numTrees = 400)  
        
        save(propensity.scores, file = paste0("./cwa/output/propensityscores.5b", output.label))
        
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
          
          
          test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] #same as tree.prob
        
        # generate model matrix for test data for equation without interaction terms
          propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]
          
          test$pscore.forest <- predict(propensity.scores$forest, 
                                        newdata = propensity.model.matrix.test,
                                        estimate.variance = TRUE)$predictions[,1]
          
          save(test, file = paste0("./cwa/output/test.5b", output.label))
        
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
        modelAnalysis(test, "B5", threshold = .5)
      
      ##### Predict propensity to violate on full test dataset using low.MSE --------
      
        #uses reg forest (lowest-MSE method)
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest,
                                         newdata = lasso.model.matrix.full)$predictions[,1]
        
      ##### Save output -----------------------------------------------
      save(full.predictions, file = paste0("./cwa/output/full.predictions.5b", output.label))
      
      save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5b", output.label))

####################################################
# 4. Run models on DMR dataset with DMR variables
####################################################
  
  print("DMR model runs, part 2")
  # Set Up -------
    rm(list = ls(all = TRUE)) #clean out workspace
    
    # load packages -----------------------------------------------------------
      library(reshape2)
      library(lars)
      library(ggplot2)
      library(plyr)
      library(sandwich)
      library(stargazer)
      library(data.table)
      library(maptools)
      library(dplyr)
      library(xtable)
      library(ROCR)
      library(Hmisc)
      library(texreg)
      library(devtools)
      library(RCurl)
      library(grf) 
      library(glmnet)
      library(rpart) # decision tree
      library(rpart.plot) # enhanced tree plots
      library(pROC) # for sensitivity and specificity
      
    # set seed for replicable results 
      set.seed(36)
    
    # Load data file
      load("./cwa/data/modified_data/dmr.random.5sets.Rdata")
  
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
    
  
  ##### Write function for analyzing models + producing plots/tables --------------------------------------
  
    modelAnalysis <- function(test, sample = c("A1", "A2", "A3", "A4", "A5", 
                                               "B1", "B2","B3","B4","B5",
                                               "C1", "C2", "C3", "C4", "C5"), 
                              threshold = 0.5){
      # this function takes in the test DF (with predicted values) and a sample of data (A, B or C)
      # and computes the MSEs, outputs the table, produces all of the sensitivity, specificity, AUC
      # statistics and corresponding output, and then spits out the lowest MSE method at the end
      
  ##### Write function to compute residuals & MSEs for all 5 methods --------------------------------

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
    
  ##### Sensitivity and specificty for all methods ---------------------------------------------------

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
      
    ##### Compute Confusion Matrix and Key Statistics -------------------------------------
    
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

      # function to get all of the statistics from the confusion matrix
      
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
  
      df.marker <- 1 #this is added to the output label
      
      test <- as.data.frame(test.random.dmr[cat == "INS"])
      train <- as.data.frame(train.random.dmr[cat == "INS"]) 
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test.random.dmr, train.random.dmr[cat != "INS"])
      
      
      # adjust output label as needed 
        output.label <- paste0("081218.",df.marker, ".Rdata")
      
      # select covariates for propensity equation (ADD DMR VARIABLES) 
      
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
                             "prox.1yr", "prox.2yr", "prox.5yr",
                             "dmr.1yr", "d80.1yr", "d90.1yr", "e90.1yr")
        
          propensity.scores <- getPropensity(data = train, 
                                           covariates = covariate.names,      
                                           Wvar = "DV", numTrees = 400)  
        
          save(propensity.scores, file = paste0("./cwa/output/propensityscores.5c", output.label))
        
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
          
          
          test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] #same as tree.prob
      
        # generate model matrix for test data for equation without interaction terms
          propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]
          
          test$pscore.forest <- predict(propensity.scores$forest, 
                                        newdata = propensity.model.matrix.test,
                                        estimate.variance = TRUE)$predictions[,1]
          
          save(test, file = paste0("./cwa/output/test.5c", output.label))
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
        modelAnalysis(test, "C1", threshold = .5)
      
      ##### Predict propensity to violate on full dataset using low.MSE method -------------
        
        #uses reg forest (lowest-MSE method)
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest,
                                         newdata = lasso.model.matrix.full)$predictions[,1]
        
      # Save output -----------------------------------------------
        save(full.predictions, file = paste0("./cwa/output/full.predictions.5c", output.label))
  
        save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5c", output.label))
        
    
  ##############################  RUN #2 ##################################################
  
      df.marker <- 2 #this is added to the output label
      
      test <- as.data.frame(test2.random.dmr[cat == "INS"])
      train <- as.data.frame(train2.random.dmr[cat == "INS"]) 
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test2.random.dmr, train2.random.dmr[cat != "INS"])
      
      # adjust output label as needed
        output.label <- paste0("081218.",df.marker, ".Rdata")
      
      # select covariates for propensity equation (ADD DMR VARIABLES)
      
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
                             "prox.1yr", "prox.2yr", "prox.5yr",
                             "dmr.1yr", "d80.1yr", "d90.1yr", "e90.1yr")
        
        propensity.scores <- getPropensity(data = train, 
                                           covariates = covariate.names,      
                                           Wvar = "DV", numTrees = 400)  
        
        save(propensity.scores, file = paste0("./cwa/output/propensityscores.5c", output.label))
      
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
          
          
          test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] #same as tree.prob
        
        # generate model matrix for test data for equation without interaction terms
          propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]
          
          test$pscore.forest <- predict(propensity.scores$forest, 
                                        newdata = propensity.model.matrix.test,
                                        estimate.variance = TRUE)$predictions[,1]
          
          save(test, file = paste0("./cwa/output/test.5c", output.label))
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
        modelAnalysis(test, "C2", threshold = .5)
      
      
      ##### Predict propensity to violate on full dataset using low.MSE method -------------
      
      #uses reg forest (lowest-MSE method)
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest,
                                         newdata = lasso.model.matrix.full)$predictions[,1]
      
      # Save output -----------------------------------------------
        save(full.predictions, file = paste0("./cwa/output/full.predictions.5c", output.label))
  
        save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5c", output.label))
        
  
  ##############################  RUN #3 ##################################################
  
      df.marker <- 3 #this is added to the output label
      
      test <- as.data.frame(test3.random.dmr[cat == "INS"])
      train <- as.data.frame(train3.random.dmr[cat == "INS"]) 
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test3.random.dmr, train3.random.dmr[cat != "INS"])
      
      
      # adjust output label as needed 
        output.label <- paste0("081218.",df.marker, ".Rdata")
      
      # select covariates for propensity equation (ADD DMR VARIABLES) 
      
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
                             "prox.1yr", "prox.2yr", "prox.5yr",
                             "dmr.1yr", "d80.1yr", "d90.1yr", "e90.1yr")
      
        propensity.scores <- getPropensity(data = train, 
                                         covariates = covariate.names,      
                                         Wvar = "DV", numTrees = 400)  
      
        save(propensity.scores, file = paste0("./cwa/output/propensityscores.5c", output.label))
      
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
          
          
          test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] #same as tree.prob
        
        # generate model matrix for test data for equation without interaction terms
          propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]
          
          test$pscore.forest <- predict(propensity.scores$forest, 
                                        newdata = propensity.model.matrix.test,
                                        estimate.variance = TRUE)$predictions[,1]
          
          save(test, file = paste0("./cwa/output/test.5c", output.label))
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
      modelAnalysis(test, "C3", threshold = .5)
      
      
      ##### Predict propensity to violate on full dataset using low.MSE method -------------
      
        #uses reg forest (lowest-MSE method)
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest,
                                         newdata = lasso.model.matrix.full)$predictions[,1]
        
      # Save output -----------------------------------------------
        save(full.predictions, file = paste0("./cwa/output/full.predictions.5c", output.label))
      
        save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5c", output.label))
      
  
  ##############################  RUN #4 ##################################################
  
      df.marker <- 4 #this is added to the output label
      
      test <- as.data.frame(test4.random.dmr[cat == "INS"])
      train <- as.data.frame(train4.random.dmr[cat == "INS"]) 
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test4.random.dmr, train4.random.dmr[cat != "INS"])
      
      
      # adjust output label as needed 
        output.label <- paste0("081218.",df.marker, ".Rdata")
      
      # select covariates for propensity equation (ADD DMR VARIABLES) 
      
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
                             "prox.1yr", "prox.2yr", "prox.5yr",
                             "dmr.1yr", "d80.1yr", "d90.1yr", "e90.1yr")
        
        
        propensity.scores <- getPropensity(data = train, 
                                           covariates = covariate.names,      
                                           Wvar = "DV", numTrees = 400)  
        
        save(propensity.scores, file = paste0("./cwa/output/propensityscores.5c", output.label))
        
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
          
          
          test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] #same as tree.prob
        
        # generate model matrix for test data for equation without interaction terms
          propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]
          
          test$pscore.forest <- predict(propensity.scores$forest, 
                                        newdata = propensity.model.matrix.test,
                                        estimate.variance = TRUE)$predictions[,1]
          
          save(test, file = paste0("./cwa/output/test.5c", output.label))
        
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
        modelAnalysis(test, "C4", threshold = .5)
      
      
      ##### Predict propensity to violate on full dataset using low.MSE method -------------
      
        #uses reg forest (lowest-MSE method)
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest,
                                         newdata = lasso.model.matrix.full)$predictions[,1]
        
        
      ##### Save output -----------------------------------------------
        save(full.predictions, file = paste0("./cwa/output/full.predictions.5c", output.label))
      
        save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5c", output.label))
      
  
  ##############################  RUN #5 ##################################################
  
      df.marker <- 5 #this is added to the output label
      
      test <- as.data.frame(test5.random.dmr[cat == "INS"])
      train <- as.data.frame(train5.random.dmr[cat == "INS"]) 
      
      #full predictions: inspected facilities in test and ALL uninspected
        full.predictions <- rbind.data.frame(test5.random.dmr, train5.random.dmr[cat != "INS"])
      
      # adjust output label as needed 
        output.label <- paste0("081218.",df.marker, ".Rdata")
      
      # select covariates for propensity equation (ADD DMR VARIABLES) ------------------------------------
      
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
                             "prox.1yr", "prox.2yr", "prox.5yr",
                             "dmr.1yr", "d80.1yr", "d90.1yr", "e90.1yr")
        
        
        propensity.scores <- getPropensity(data = train, 
                                           covariates = covariate.names,      
                                           Wvar = "DV", numTrees = 400)  
        
        save(propensity.scores, file = paste0("./cwa/output/propensityscores.5c", output.label))
        
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
          
          
          test$pscore.tree <- predict(propensity.scores$tree, newdata = test)[,2] #same as tree.prob
          
        # generate model matrix for test data for equation without interaction terms
          propensity.model.matrix.test <- model.matrix(propensity.scores$propensity.eq, test)[,-1]
          
          test$pscore.forest <- predict(propensity.scores$forest, 
                                        newdata = propensity.model.matrix.test,
                                        estimate.variance = TRUE)$predictions[,1]
          
          save(test, file = paste0("./cwa/output/test.5c", output.label))
      
      ##### Compare models, generate sensitivity/specificity, MSE, AUC, etc. ----------------------------------------
        modelAnalysis(test, "C5", threshold = .5)
      
      ##### Predict propensity to violate on full dataset using low.MSE method -------------
        
        #uses reg forest (lowest-MSE method)
        lasso.model.matrix.full <- model.matrix(propensity.scores$propensity.eq, full.predictions)[,-1]
        
        full.predictions$prop <- predict(propensity.scores$forest,
                                         newdata = lasso.model.matrix.full)$predictions[,1]
      
      ##### Save output -----------------------------------------------
        save(full.predictions, file = paste0("./cwa/output/full.predictions.5c", output.label))
        
        save(test, full.predictions, propensity.scores, file = paste0("./cwa/output/full_output_5c", output.label))
      

####################################################
# 5. Generate failure rates by risk score, using best method  
####################################################
          
  ##### Set Up -------
    rm(list = ls(all = TRUE)) #clean out workspace
    
    # load packages
      library(reshape2)
      library(ggplot2)
      library(data.table)
      library(xtable)
      library(readr)
      library(lubridate) # for dates
      library(plyr)
      library(dplyr)
      library(tidyverse)
      library(RColorBrewer)
      library(ROCR)
      library(gdata)
      library(texreg)
      library(RCurl)
      library(grf)
      library(glmnet)
      library(rpart)
      library(pROC)
      library(sandwich)
      library(stargazer)
  
  ##### select model to work off of, save with "final" designation ---------------------------------------------
    
    # !!! modify this file name to match the name of your full.predictions file 
        load("./cwa/output/full.predictions.5a.5.Rdata")
    
    #save this file as the file we will use for a single-run analysis later on 
    save(full.predictions, file = "./cwa/output/full.predictions.5a.5.final.Rdata")
    
    # !!! modify this file name to match the name of your full_output file 
      load("./cwa/output/full_output_5a.5.Rdata")
      save(test, full.predictions, propensity.scores, file = "./cwa/output/full_output_5a_5_final.Rdata")
      
  ##### generate 500 train/test splits and run reg forest --------------
  
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
    
    sumx <-  paste(covariate.names, collapse = " + ") 
    
    propensity.eq <- paste("DV", paste(sumx, sep = " + "), sep = " ~ ")
    propensity.eq <- as.formula(propensity.eq)
    
    # Write function to run regression forest
      run_reg_forest <- function(insp.df, smpl, numTrees) {
        train <- insp.df[smpl,]
        test <- insp.df[-smpl,]
        
        forest.model.matrix <- model.matrix(propensity.eq, train)[,-1]
        W <- as.matrix(train[,"DV"], ncol = 1)
        reg.forest <- regression_forest(forest.model.matrix, W, 
                                        num.trees = numTrees, 
                                        ci.group.size = 4, honesty = TRUE)
        
        # generate model matrix for test data for equation without interaction terms
        propensity.model.matrix.test <- model.matrix(propensity.eq, test)[,-1]
        
        test$pscore.forest <- predict(reg.forest, 
                                      newdata = propensity.model.matrix.test,
                                      estimate.variance = TRUE)$predictions[,1]
        return(test)
      }
  
  ##### Read in data
    load("./cwa/data/modified_data/full.data.for.models.Rdata")
    random.full.ins <- random.full[random.full$cat == "INS",]
    random.full.noinsp <- random.full[random.full$cat != "INS",]
    
  ##### Create 500 training samples 
    set.seed(99)
    samplesplits <- replicate(500, sample(nrow(random.full.ins),
                                          round(8 * nrow(random.full.ins) / 10),
                                          replace = FALSE))
    #each column is a draw of numbers to be the training set, all inspected
    save(samplesplits, file = "./cwa/output/samplesplits.Rdata")
    
  ##### Initialize DF to hold only predicted values (predicts on test set of insp facilities only)
    test.predictions.all.runs <- data.frame(matrix(NA_real_, 
                                                   nrow = (nrow(random.full.ins) - nrow(samplesplits)), # = test set
                                                   ncol = 500))
  
  ##### Run regression forest 500x
    for (i in 1:500) {
      print(i)
      out <- run_reg_forest(random.full.ins, samplesplits[,i], 400)
      test.predictions.all.runs[,i] <- out$pscore.forest 
    }
    
  save(test.predictions.all.runs, file = "./cwa/output/test.predictions.all.runs.Rdata")

####################################################
# 6. Estimate failure rates from 500 runs 
####################################################

  library(Hmisc)
  library(rlang)

  ##### Define function to calculate fail rates by score cut -----
    score_table <- function(df, indicator, fail.ind){ 
      require(dplyr)
      indicator <- enquo(indicator)      # Create quosure
      fail.ind  <- enquo(fail.ind)       # Create quosure     
      score_rates <- 
        df %>%
        group_by(score.cut) %>%
        summarise(
          facs = length(unique(REGISTRY_ID)),
          insp = sum(UQ(indicator)),                    # Use UQ to unquote
          fails = sum(UQ(indicator) * UQ(fail.ind)),    # Use UQ to unquote
          fail.rate = ifelse(insp == 0, 0, fails/insp)) # Avoid infinity
      return(score_rates)
    }

  ##### load data ---------------------------------------------------------------
  
    all.test.predictions <- test.predictions.all.runs
  
    # inspections dataset
    inspections <- read_csv("./cwa/data/raw_data/npdes_downloads/NPDES_INSPECTIONS.csv")
    
  ##### get failure rates 500x --------------------------------------------------
  
    #create a container for fail rates
      fail.rates.df <- data.frame(matrix(NA_integer_, nrow = 20, ncol = 1))
      names(fail.rates.df)[1] <- "score.cut"
      fail.rates.df$score.cut <- as.character(cut2(seq(0.01,1,.05), cuts = seq(0,1,.05), digits = 2))
      
  
    start.z <- 1
    end.z <- 500
    
    for (z in start.z:end.z) { #z counts through our model runs. 
      test <- random.full.ins[-samplesplits[, z],] #the zth column of samplesplits IDs the training obs
      test$prop <- all.test.predictions[, z] #the zth column of predictions has the pscores 
      cwa.join <-  # establish relationship used in script & remove extra columns
        test %>%
        select(REGISTRY_ID, cat, ACTUAL_END_DATE, X1, FAC_STATE,
               PERMIT_MAJOR, PERMIT_MINOR, DV, prop)
      
      # 1. Re-generate the score cuts/score splits between cuts  -------------------------
        cwa.join$score.cut <- as.character(cut2(cwa.join$prop, cuts = seq(0,1,0.05), digits = 2))
        # Generate score groups (for random selection)
        cwa.join$score.group <- cwa.join %>% group_indices(score.cut) 
        
      # 2. Create a table that shows, per score cut 1) # of facilities 2) #inspections 3) # violations. this is only inspected facilities, so 1 should equal 2
        cwa.join$insp_group <- ifelse(cwa.join$cat == "INS", 1, 0)
        cwa.join$fail.insp <-  with(cwa.join,
                                    ifelse(insp_group == 0 , 0 , DV)
        ) # ensure that it's only among the inspected (2012-16) that we care about fail.insp rates
        
        score_rates <- score_table(cwa.join, insp_group, fail.insp) 
        # this produces 1) # of facilities (inspected or not), 2) # of inspected facilities, 
      
      # 3) # of inspected fac that fail, 4) rate of failure (#3/#2)
        score_rates$share.insp <- score_rates$insp / sum(score_rates$insp)
        score_rates$score.cut <- as.character(score_rates$score.cut)
        #note: share.insp is the share of inspections alloacted to that group, not the share of facilities in the group that are inspected
        
      fail.rates.df <- left_join(fail.rates.df, score_rates[, c("score.cut", "fail.rate")], 
                                 by = "score.cut", suffix = c(as.character(z-1), as.character(z)))
    }
    
    save(fail.rates.df, file = "./cwa/output/fail.rates.df.1_500.Rdata")
    
  
  fail.rate.means <- rowMeans(fail.rates.df[, 2:501], na.rm = T) #mean fail rate by score group
  fail.rate.sds <- apply(fail.rates.df[, 2:501], 1, function(x) sd(x, na.rm = T)) #SD of fail rate by score group
  
  results.by.score.cut <- cbind.data.frame(score.cut = fail.rates.df$score.cut, fail.rate.means, fail.rate.sds)
  results.by.score.cut$fail.rate.UB <- pmin(1, results.by.score.cut$fail.rate.means + qnorm(.975)*results.by.score.cut$fail.rate.sds)
  results.by.score.cut$fail.rate.LB <- pmax(0, results.by.score.cut$fail.rate.means + qnorm(.025)*results.by.score.cut$fail.rate.sds)
  #upper bound and lower bound are cut off to 0 and 1 
  
  save(results.by.score.cut, file = "./cwa/output/results.by.score.cut.Rdata")

####################################################
# 7. Calculate benefits from reallocations 
####################################################
  
  ##### Prepare data 
    # set seed for replicable results
      set.seed(36) 
    
    # Create Final Analysis Set (20% test + 20% of uninspected faciltiies)
    load("./cwa/output/full.predictions.5a.5.final.Rdata") 
    #this includes 100% of uninspected and 20% of inspected
    
    # get twenty pct random sample from uninspected
      uninsp <- full.predictions %>% filter(cat != "INS")
      test <- full.predictions %>% filter(cat == "INS")
      twenty.pct <- round(length(uninsp$REGISTRY_ID) * 0.20)
      twenty.pct.sample <- sample(x = 1:length(uninsp$REGISTRY_ID), size = twenty.pct, replace = FALSE)
      twenty.pct.uninsp <- uninsp %>% filter(row_number() %in% twenty.pct.sample) 
      test.and.20pct.uninsp <- rbind(test, twenty.pct.uninsp)
      save(test.and.20pct.uninsp, file = "./cwa/output/test.and.20pct.uninsp.Rdata")
      
      load("./cwa/data/modified_data/full.random.5sets.Rdata") 
      
      rm(test.random, test2.random, test3.random, test4.random)
      rm(train.random, train2.random, train3.random, train4.random)
      
      all_facs <- full_join(train5.random, test5.random)
      test <- test5.random %>% filter(cat == "INS") 
      uninsp <- all_facs %>% filter(cat != "INS")
      
      N_allfacs <- length(all_facs$REGISTRY_ID)
      
  
  ##### Generate One Score Table for Failures Graphic 
  
    #first join all.facs with inspections so each inspection has a state attached
      testand20pct.merge <- test.and.20pct.uninsp[, c("REGISTRY_ID", "FAC_STATE")]
      inspections.w.state <- inner_join(inspections, testand20pct.merge)
      #note: this reduces the # of inspections to just the inspections of facilities in our 
      #test + 20% uninsp data, ensuring allocations don't assign to facilities that don't exist in our data
      
    #1. how many inspections of unique facilities per year, from 2012-2016?
      insp.summary <- inspections.w.state %>% mutate(year.insp = year(mdy(ACTUAL_END_DATE))) %>%
        filter(year.insp >= 2012 & year.insp <= 2016) %>%
        distinct(REGISTRY_ID, year.insp, .keep_all = TRUE) %>% 
        group_by(year.insp) %>% summarise(annual_insp = n())
      round(mean(insp.summary$annual_insp)) 
    
    # 2. Determine distribution of inspections across states
    
      insp.by.state <- inspections.w.state %>% mutate(year.insp = year(mdy(ACTUAL_END_DATE))) %>%
        filter(year.insp >= 2012 & year.insp <= 2016) %>%
        distinct(REGISTRY_ID, year.insp, .keep_all = TRUE) %>%
        group_by(year.insp, FAC_STATE) %>% summarise(st.insp = n())
      
      avg_yr_state <- insp.by.state %>% group_by(FAC_STATE) %>%
        summarise(stavg = round(mean(st.insp)))
    
    # 3. if needed, adjust the numbers to reflect the proportion of facilities over which we are allocating
      # stavg column reflects the average # of inspections of unique facilities in the test.and.20ct.uninsp dataset
      # if needed to increase or decrease from that, scale up or down here
      facility.proportion <- 1
      
      avg_yr_state$allocation.key <- round(avg_yr_state$stavg * facility.proportion)
      
      N_insp <- sum(avg_yr_state$allocation.key)
    
    # 4. Rename vars for consistency with later script & remove extra columns
      cwa.join <-  
        test.and.20pct.uninsp %>% #This defines which facilities are reallocated over
        select(REGISTRY_ID, cat, ACTUAL_END_DATE, X1, FAC_STATE, FAC_LAT, FAC_LONG,
               PERMIT_MAJOR, PERMIT_MINOR, DV, prop)
      
      cwa.join$score.cut <- as.character(cut2(cwa.join$prop, cuts = seq(0,1,0.05), digits = 2))
      cwa.join$insp_group <- ifelse(cwa.join$cat == "INS", 1, 0)
      cwa.join$fail.insp <-  with(cwa.join,
                                  ifelse(insp_group == 0 , 0 , DV)
      ) 
    
    
    # 5. Generate Score Table (Results)
      run.results <- score_table(cwa.join, insp_group, fail.insp); run.results 
      run.results$base <- round(run.results$insp / sum(run.results$insp) * N_insp) #base allocation is by share

####################################################
# 8. Conduct Facility Inspection "Assignment" Under Different Allocations 
####################################################
      
  ##### 1. Based on the score.cut allocations in run.results, "assign" inspections to facilities in test + uninspected (set key values)
    cwa.join <- cwa.join %>% arrange(desc(prop))
    unq.cuts <- unique(cwa.join$score.cut)
    min.prob <- 0.01
  
  ##### 2. Define predicted inspection fail/pass based on 500 runs of fail rates' mean, upper bound, lower bound
  
    # initialize columns
      cwa.join$predicted.DV <- 0
      cwa.join$predicted.DV.LB <- 0
      cwa.join$predicted.DV.UB <- 0
    
    
    for (i in 1:length(unq.cuts)) {
      cut.grp <- as.character(run.results$score.cut[i])
      
      num.fails <- round(run.results$facs[run.results$score.cut == cut.grp] * 
                           results.by.score.cut$fail.rate.means[results.by.score.cut$score.cut == cut.grp])
      
      LB.num.fails <- round(run.results$facs[run.results$score.cut == cut.grp] * 
                              results.by.score.cut$fail.rate.LB[results.by.score.cut$score.cut == cut.grp])
      
      UB.num.fails <- round(run.results$facs[run.results$score.cut == cut.grp] * 
                              results.by.score.cut$fail.rate.UB[results.by.score.cut$score.cut == cut.grp])
      
      num.fac <- run.results$facs[run.results$score.cut == cut.grp]
      smpl.UB <- sample(num.fac, UB.num.fails, replace = F)
      smpl.mean <- sample(smpl.UB, num.fails, replace = F)
      smpl.LB <- sample(smpl.mean, LB.num.fails, replace = F)
      
      assignment.mean <- rep(0, num.fac)
      assignment.mean[smpl.mean] <- 1
      cwa.join$predicted.DV[cwa.join$score.cut == cut.grp] <- assignment.mean
      
      assignment.UB <- rep(0, num.fac)
      assignment.UB[smpl.UB] <- 1
      cwa.join$predicted.DV.UB[cwa.join$score.cut == cut.grp] <- assignment.UB
      
      assignment.LB <- rep(0, num.fac)
      assignment.LB[smpl.LB] <- 1
      cwa.join$predicted.DV.LB[cwa.join$score.cut == cut.grp] <- assignment.LB
    }
    
  ##### Base allocation (this mimics the existing allocation across risk scores)
    cwa.join$base.assigned <- 0
    
    for (i in 1:length(unq.cuts)) {
      cut.grp <- run.results$score.cut[i]
      num.insp <- run.results$base[i]
      num.fac <- run.results$facs[i]
      smpl <- sample(num.fac, num.insp, replace = F)
      assignment <- rep(0, num.fac)
      assignment[smpl] <- 1
      cwa.join$base.assigned[cwa.join$score.cut == cut.grp] <- assignment
    }
  
  ##### Aggressive, national ---------------------------
    
    #assigns inspections to the facilities with the highest risk score
    cwa.join <- cwa.join %>% mutate(r1.assigned = ifelse(row_number() <= N_insp, 1,0 )) 
    sum(cwa.join$r1.assigned) == N_insp #confirms that the correct # of inspections have been allocated
    
  ##### Aggressive, state-level -------------------------------------------------
    
    cwa.join$r2.assigned <- 0
    
    #assigns inspection to the facilities with the highest risk score, by state
    for (i in 1:nrow(avg_yr_state)) {
      cwa.join[cwa.join$FAC_STATE == avg_yr_state$FAC_STATE[i],] <- 
        cwa.join %>% 
        filter(FAC_STATE == avg_yr_state$FAC_STATE[i]) %>% 
        mutate(r2.assigned = ifelse(row_number() <= avg_yr_state$allocation.key[i], 1, 0)) 
    }
    sum(cwa.join$r2.assigned) == N_insp #confirms that the correct # of inspections have been allocated
  
  ##### Deterrence, national --------------------------------
    
    cwa.join$r3.assigned <- 0
    
    r3.assign.df <- cwa.join[, c("REGISTRY_ID", "prop")]
    
    #Here we assign a portion of inspections randomly
      min.prob.total.alloc <- round(min.prob * nrow(cwa.join)) # this is the number of inspections to allocate randomly
      smpl.min.prob <- sample(nrow(cwa.join), min.prob.total.alloc, replace = F)
      r3.assign.df$minprob <- 0
      r3.assign.df$minprob[smpl.min.prob] <- 1
    
    #With remaining inspections available, we assign by risk score
      insp.left.for.risk <- N_insp - min.prob.total.alloc
      
      to.be.assigned <- 
        r3.assign.df %>% 
        filter(minprob == 0) %>% 
        mutate(risk.assigned = ifelse(row_number() <= insp.left.for.risk, 1,0 ))
      
      r3.assign.df$risk <- 0
      r3.assign.df$risk[r3.assign.df$minprob == 0] <- to.be.assigned$risk.assigned
      r3.assign.df$r3.assigned <- r3.assign.df$minprob + r3.assign.df$risk
    
    cwa.join$r3.assigned <- r3.assign.df$r3.assigned
    sum(cwa.join$r3.assigned) == N_insp #confirms that the correct # of inspections have been allocated
    
  ##### Majors, deterrence, state-level -------------------------
  
    cwa.join$r4.assigned <- 0
    r4.assign.df <- cwa.join[, c("REGISTRY_ID", "prop", "PERMIT_MAJOR", "FAC_STATE")]
    r4.assign.df$major <- 0
    r4.assign.df$minor <- 0
    r4.assign.df$risk <- 0
    
    for (i in 1:nrow(avg_yr_state)) {
      st <- avg_yr_state$FAC_STATE[i]
      st.insp <- avg_yr_state$allocation.key[i]
      st.facs <- r4.assign.df %>% filter(FAC_STATE == st)
      
      #Assign inspections to half of major facilities
        st.majors <- st.facs %>% filter(PERMIT_MAJOR == TRUE)
        major.insp.total <- min(st.insp, round(nrow(st.majors)/2))
        smpl.major <- sample(nrow(st.majors), major.insp.total, replace = F)
        assignment.major <- rep(0, nrow(st.majors))
        assignment.major[smpl.major] <- 1
        r4.assign.df$major[r4.assign.df$FAC_STATE == st & r4.assign.df$PERMIT_MAJOR == TRUE] <- assignment.major
      
      #Among minor facilities, assign random inspections  
        st.minors <- st.facs %>% filter(PERMIT_MAJOR == FALSE)
        minor.insp.total <- min(st.insp-major.insp.total, round(nrow(st.minors)*min.prob))
        smpl.minor <- sample(nrow(st.minors), minor.insp.total, replace = F)
        assignment.minor <- rep(0, nrow(st.minors))
        assignment.minor[smpl.minor] <- 1
        r4.assign.df$minor[r4.assign.df$FAC_STATE == st & r4.assign.df$PERMIT_MAJOR == FALSE] <- assignment.minor
        
      #With remaining inspections, assign by risk
        st.risk <- r4.assign.df %>% filter(FAC_STATE == st & major == 0 & minor == 0) 
        risk.insp.total <- max(st.insp - major.insp.total - minor.insp.total, 0)
        assignment.risk <- c(rep(1, risk.insp.total), rep(0, nrow(st.risk) - risk.insp.total))
        r4.assign.df$risk[r4.assign.df$FAC_STATE == st & r4.assign.df$major == 0 & r4.assign.df$minor == 0] <- assignment.risk
      }
    
    
    r4.assign.df$r4.assigned <- r4.assign.df$major + r4.assign.df$minor + r4.assign.df$risk
    cwa.join$r4.assigned <- r4.assign.df$r4.assigned
    sum(cwa.join$r4.assigned) == N_insp #confirms that the correct # of inspections have been allocated
    
    save(cwa.join, file = "./cwa/output/cwa.join.Rdata")


####################################################
# 9. Produce Figure 1 
####################################################
  ##### Set Up -------
    rm(list = ls(all = TRUE)) #clean out workspace
    
    ## load packages
      library(reshape2)
      library(ggplot2)
      library(plyr)
      library(dplyr)
  
  ##### Source External Files/Functions
    ## --- R ggsave latex command ----------------------
    #'Saves a ggplot graphic to a file and creates the code to include it in a
    #'LaTeX document.
    #'
    #'Saves a ggplot graphic to a file and creates the code to include it in a
    #'LaTeX document.
    #'
    #'
    #'@param ...  arguments passed to the ggsave function
    #'(\code{\link[ggplot2]{ggsave}})
    #'@param caption The caption. Default to NULL, indicating no caption.
    #'@param label The label. Default to NULL, indicating no label.
    #'@param figure.placement The placement of the figure. Default to "hbt".
    #'@param floating Logical. Indicates if the figure should be placed in a
    #'floating environment. Default to TRUE
    #'@param caption.placement Should the caption be on top or bottom of the
    #'figure. Default to "bottom"
    #'@param latex.environments Alignment of the figure. Default to "center".
    #'@return The graphic will be saved to a plot and the relevant LaTeX code is
    #'printed.
    #'@author Thierry Onkelinx \email{Thierry.Onkelinx@@inbo.be}, Paul Quataert
    #'@seealso \code{\link[ggplot2]{ggsave}}
    #'@keywords hplot graphs
    #'@examples
    #'
    #'	require(ggplot2)
    #'  data(cars)
    #'	p <- ggplot(cars, aes(x = speed, y = dist)) + geom_point()
    #'	ggsave.latex(p, filename = "test.pdf", label = "fig:Cars", 
    #'    caption = "A demo plot", height = 5, width = 4)
    #'
    #'@export
    #'@importFrom ggplot2 ggsave
    ggsave.latex <- function(..., caption = NULL, label = NULL, figure.placement = "hbt", floating = TRUE, caption.placement="bottom", latex.environments="center"){
      ggsave(...)
      
      cat("\n\n")
      if(floating){
        cat("\\begin{figure}[", figure.placement, "]\n", sep = "")
      }
      cat("    \\begin{", latex.environments,"}\n", sep = "")
      if(!is.null(caption) && caption.placement == "top"){
        cat("        \\caption{", caption, "}\n", sep = "")
      }
      args <- list(...)
      if(is.null(args[["filename"]])){
        if(is.null(args[["plot"]])){
          names(args)[which(names(args) == "")[1]] <- "plot"
        }
        args[["filename"]] <- paste(args[["path"]], ggplot2:::digest.ggplot(args[["plot"]]), ".pdf", sep="")
      } else {
        args[["filename"]] <- paste(args[["path"]], args[["filename"]], sep="")
      }
      
      if(is.null(args[["width"]])){
        if(is.null(args[["height"]])){
          cat(
            "        \\includegraphics[height = 7in, width = 7in]{",
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        } else {
          cat(
            "        \\includegraphics[height = ", 
            args[["height"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            ", width = 7in]{", 
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        }
      } else {
        if(is.null(args[["height"]])){
          cat(
            "        \\includegraphics[height = 7in, width = 7 ", 
            args[["width"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            "]{", 
            args[["filename"]], 
            "}\n", 
            sep = "")
        } else {
          cat(
            "        \\includegraphics[height = ", 
            args[["height"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            ", width = ", 
            args[["width"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            "]{", 
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        }
      }
      if(!is.null(caption) && caption.placement == "bottom"){
        cat("        \\caption{", caption, "}\n", sep = "")
      }
      if(!is.null(label)){
        cat("        \\label{", label, "}\n", sep = "")
      }
      cat("    \\end{", latex.environments,"}\n", sep = "")
      if(floating){
        cat("\\end{figure}\n")
      }
      cat("\n\n")
    }
    
  ##### Load Data ------
  
    # Failure Rates for each of the 500 model runs (different 80-20 partition)
      load("./cwa/output/fail.rates.df.1_500.Rdata")
    
    # Aggregated  
      load("./cwa/output/results.by.score.cut.Rdata")
    
      long <- melt(fail.rates.df, id.vars = 1)
      long <- long[!is.na(long$value),] # drops the high-risk score cuts for which there were no obs
      
    # Final Analysis Set (20% test + all uninspected faciltiies)
      load("./cwa/output/full_output_5a_5_final.Rdata")
      rm(propensity.scores)
    
    # Get twenty pct number from uninspected
      test <- full.predictions %>% filter(cat == "INS")
      n.test <- length(test$REGISTRY_ID); n.test
      n.test.pretty <- prettyNum(n.test, big.mark = ","); n.test.pretty
      
  ##### Generate fail rate box plot ----
  
    long$midpt <- (as.numeric(substr(long$score.cut, 7, 10)) +  as.numeric(substr(long$score.cut, 2, 5))) / 2
    scaleFUN <- function(x) sprintf("%.2f", x)
    scaleFUN(seq(0, 1, .05))
    
    fails.by.score2 <- ggplot(long, aes(x = midpt, y = value, group = score.cut)) + 
      ylab("Rate of inspection failure") + xlab("Predicted risk score") + 
      scale_x_continuous(breaks = seq(0, 1, .05), labels = c("0", "", "0.1", "", "0.2", "", "0.3", "", "0.4", "", "0.5", "", "0.6", "", "0.7", "", "0.8", "", "0.9", "", "1")) + 
      theme(panel.grid.major = element_blank(), 
            panel.grid.minor = element_blank(),
            panel.background = element_blank(), 
            axis.line = element_line(colour = "black"),
            legend.position = "none") +
      geom_boxplot(width = 0.035)
    fails.by.score2
    
    ggsave.latex(fails.by.score2, 
                 width = 3.46457, 
                 height = 3,
                 filename = "./cwa/output/figs/boxplot_failsbyrisks_2.pdf",
                 caption = paste0("The boxplots below represent the distribution of inspection failure rates by predicted risk score.",
                                  " The center line indicates the median,",
                                  " and the edges of each box represent the 25th and 75th percentiles. Whiskers indicate" ,
                                  " 1.5x the interquartile (IQ) range, and dots indicate outliers beyond the 1.5x IQ range.", 
                                  " Distributions are generated through evaluating 500 regression forest model runs,",
                                  " where we randomly select a different 80-20\\% training and test partition", 
                                  " and present the results from inspection failures among the 20\\% test set (n = ", 
                                  n.test.pretty, ")",
                                  " from each run.",
                                  " The likelihood of inspection failure increases with", 
                                  " the predicted risk scores."))


####################################################
# 10. Produce Figs 2 and 3 ---------------------------------------
####################################################

  ##### Set up 
    rm(list = ls())

    library(reshape2)
    library(ggplot2)
    library(plyr)
    library(dplyr)
    library(RColorBrewer)
    library(tidyverse)
    
    load("./cwa/output/cwa.join.Rdata") 
    
    flipped <- cwa.join %>% 
      arrange(prop)
    
    #Generate cumulative failures
    flipped <-
      flipped %>% 
      mutate(cml.r1 = cumsum(predicted.DV * r1.assigned)) %>%
      mutate(cml.r1.LB = cumsum(predicted.DV.LB * r1.assigned)) %>%
      mutate(cml.r1.UB = cumsum(predicted.DV.UB * r1.assigned)) %>%
      #
      mutate(cml.r2 = cumsum(predicted.DV * r2.assigned)) %>%
      mutate(cml.r2.LB = cumsum(predicted.DV.LB * r2.assigned)) %>%
      mutate(cml.r2.UB = cumsum(predicted.DV.UB * r2.assigned)) %>%
      #
      mutate(cml.r3 = cumsum(predicted.DV * r3.assigned)) %>%
      mutate(cml.r3.LB = cumsum(predicted.DV.LB * r3.assigned)) %>%
      mutate(cml.r3.UB = cumsum(predicted.DV.UB * r3.assigned)) %>%
      #
      mutate(cml.r4 = cumsum(predicted.DV * r4.assigned)) %>%
      mutate(cml.r4.LB = cumsum(predicted.DV.LB * r4.assigned)) %>%
      mutate(cml.r4.UB = cumsum(predicted.DV.UB * r4.assigned)) %>%
      #
      mutate(fac_num = seq(1, nrow(.), 1))

  ##### HORIZONTAL DOT PLOT  w/ BOUNDING BARs -------------------------------------
  
    #number of inspections:
      N_insp <- sum(flipped$r1.assigned)
    
    #base rate:
      insp.only <- cwa.join %>% filter(cat == "INS")
      base.rate <- sum(insp.only$DV)/nrow(insp.only)
      
    #grab lower and upper bounds
      lower <- which(names(flipped) %in% c("cml.r1.LB", "cml.r2.LB", "cml.r3.LB", "cml.r4.LB")) ; lower 
      upper <- which(names(flipped) %in% c("cml.r1.UB", "cml.r2.UB", "cml.r3.UB", "cml.r4.UB")) ; upper 
      mean  <-  lower-1 
      
      bar_bounds_lower <- 
        as.data.frame(apply(flipped[lower], 2, max))  %>% 
        rownames_to_column(.) %>%
        plyr::rename(replace = c("rowname" = "bound")) %>%
        plyr::rename(replace = c("apply(flipped[lower], 2, max)" = "value"))
      
      bar_bounds_upper <-
        as.data.frame(apply(flipped[upper], 2, max))  %>% 
        rownames_to_column(.) %>%
        plyr::rename(c("rowname" = "bound")) %>%
        plyr::rename(c("apply(flipped[upper], 2, max)" = "value")) %>%
        bind_rows(bar_bounds_lower); bar_bounds_upper
      
      bar_bounds_mean <-
        as.data.frame(apply(flipped[mean], 2, max))  %>% 
        rownames_to_column(.) %>%
        plyr::rename(c("rowname" = "bound")) %>%
        plyr::rename(c("apply(flipped[mean], 2, max)" = "value")) %>%
        bind_rows(bar_bounds_upper); bar_bounds_mean
      
      bar_bounds <-
        bar_bounds_mean[1:4,] %>%
        plyr::rename(replace = c("value" = "mean")) %>%
        cbind(bar_bounds_lower[1:4,2]) %>%
        plyr::rename(replace = c("bar_bounds_lower[1:4, 2]" = "lower")) %>%
        cbind(bar_bounds_upper[1:4,2]) %>%
        plyr::rename(replace = c("bar_bounds_upper[1:4, 2]" = "upper")); bar_bounds
      
      bar_bounds_share <- cbind(bar_bounds[1], bar_bounds[2:4]/N_insp); bar_bounds_share
      save(bar_bounds_share, file = "./cwa/output/bar_bounds_share.Rdata")
      
    # Set up for plot
      
      #labels
      f_labs <- c(`cml.r1` = "Aggressive, national",
                  `cml.r2` = "Aggressive, state",
                  `cml.r3` = "Deterrence, national",
                  `cml.r4` = "Majors, deterrence, state"
      )
      
      # colors
        cols <- rev(brewer.pal(4, "Set1"))
    
    # Plot
      pdf("./cwa/output/figs/results_bar_plot.pdf", width = 7.086, height = 4.5) 
      
      results_bar_plot <-
        ggplot(data = bar_bounds_share, aes(x = forcats::fct_rev(bound), 
                                                                       y = mean, 
                                                                       colour = bound)) +
        geom_point(size = 2) +
        geom_text(aes(label=round(mean,2)),hjust=0.5, vjust=-1) +
        geom_errorbar(aes(ymin = lower, ymax = upper), size = 1, width = .2) + 
        scale_colour_manual(values = cols) +
        coord_flip() +
        scale_x_discrete(labels = as_labeller(f_labs)) +
        scale_y_continuous(limits = c(0, 1)) +
        labs(y = "Share of inspections that detect failures", 
             x = "") +
        geom_segment(aes(x = 0, xend = 4.35, y = base.rate, yend = base.rate),
                     linetype = "dashed", color = "orange", size = 1.2) +
        geom_text(aes(x = 4.5, y = base.rate, label = "Base"), size = 3.5, color = "black") + 
        theme(legend.position = "none",
              axis.text.y = element_text(hjust = 1, size = 10, color = "black"),
              axis.line.x = element_line(colour = "black"),
              axis.ticks.y = element_blank(),
              panel.grid.major.y = element_blank(),
              panel.grid.major = element_blank(), 
              panel.grid.minor = element_blank(),
              panel.background = element_blank()
              
        ) 
      
      results_bar_plot
      
      dev.off()
    
  ##### GENERATE STRIP PLOTS ---------------------------------------------------------------
  
    allocation.labels <- c("Majors, deterrence, state", "Deterrence, national", 
                           "Aggressive, state", "Aggressive, national", "Base")
    
    # Set Colors for polygons (area of CI) by taking a transparency factor (alpha) of linecols
      add.alpha <- function(col, alpha = 1) {
        if (missing(col))
          stop("Please provide a vector of colours.")
        apply(sapply(col, col2rgb) / 255, 2,
              function(x)
                rgb(x[1], x[2], x[3], alpha = alpha))
      }
    
    # Set line colors
      linecols <- brewer.pal(5, "Set1")
    
    #Plot 
      pdf("./cwa/output/figs/stripplot.pdf", width = 7.086, height = 4.5)
      par(mgp = c(1, 0.5, 0), mar = c(5, .1, .1, .1))
      plot(1,type = "n", 
           ylim = c(0.8, 8), 
           xlim = c(-.4, 1), 
           las = 1, ylab = "", xlab = "", 
           axes = F)
      
      segcols <- add.alpha(linecols, .15)
      noninsp.col <- add.alpha("lightgrey", .35)
      fac.col <- add.alpha("black", .008)
      last.alloc.col.index <- which(names(cwa.join) == "r4.assigned")
      
      for (i in 1:5) { #one for each allocation
        insp.riskscores <- cwa.join$prop[cwa.join[, last.alloc.col.index + 1 - i] == 1]
        segments(insp.riskscores, i, insp.riskscores, i + 0.5, col = segcols[i], lwd = 0.1)
        text(x = 0, y = i + 0.25, label = allocation.labels[i], cex = .8, pos = 2)
      }  
      
      #one for the base distribution
      all.riskscores <- cwa.join$prop
      segments(all.riskscores, 7, all.riskscores, 7.5, col = fac.col)
      text(x = 0, y = 7.25, label = "All facilities", cex = .8, pos = 2)
      text(x = 0.4, y = 6.2, label = "Inspected facilities under different allocations", cex = .9)
      
      axis(1, at = c(0, .2, .4, .6, .8, 1), labels = TRUE, tick = TRUE, pos = 0, cex.axis = .8)
      mtext("Risk score", side = 1, line = 2.5, at = .5)
      
      dev.off()

  ##### Export Results Table ---------------------------------------------------------------
  
  
    # Rescale Table  - Include Percents and CI's -----
    
    benefits.comb.CIs <-
      bar_bounds_share %>%
      mutate(improvement = format(round((mean - base.rate)/base.rate * 100)), digits = 2) %>%
      mutate(CI = paste0("(", round(lower,2), ", ", round(upper,2), ")")) %>%
      mutate(mean = format(round(mean, digits = 2)))  %>%
      select(mean, CI, improvement) #%>%
    
    benefits.comb.CIs
    
    rownames(benefits.comb.CIs) <- c("Aggressive, national",
                                     "Aggressive, state",
                                     "Deterrence, national",
                                     "Majors, deterrence, state")
    
    names(benefits.comb.CIs) <- c("Share of inspections that detect failures",
                                  "95\\% confidence interval",
                                  "Improvement over base case (\\% change)")
    
    benefits.comb.tbl <- xtable(benefits.comb.CIs, align = "lccc")
    
    n.test.pretty <- prettyNum(sum(cwa.join$cat == "INS"), big.mark = ",")
    n.20pct.pretty <- prettyNum(sum(cwa.join$cat != "INS"), big.mark = ",")
    N_insp.pretty <- prettyNum(N_insp, big.mark = ","); N_insp.pretty
    
    label(benefits.comb.tbl) <- "tab:benefits.comb"
    caption(benefits.comb.tbl) <- paste0("Expected share of inspections that detect failures and the 95\\% confidence interval",
                                         " per reallocation proposal. All four reallocations substantially increase the share of ",
                                         " inspections that detect failures.",
                                         " Rates calculated by reallocating 20\\% of average annual inspections (", N_insp.pretty, ")",
                                         " among the 20\\% test case dataset (n = ", n.test.pretty, "), and" ,
                                         " a 20\\% random sample of the uninspected facilities (n = ", n.20pct.pretty,
                                         ") per the rules in each proposal")
    
    today <- "081218"
    
    print(benefits.comb.tbl,
          file = paste0("./cwa/output/benefits.comb.CIs.", today, ".tex"),
          caption.placement = "top",
          digits = c(0, 0, 0 , 0),
          sanitize.text.function = function(x){x},
          include.colnames = T,
          include.rownames = T,
          booktabs = T,
          label = "tab:benefits.comb.CIs")
  
####################################################
# 11. BC comparison + Produce Figure 4 
####################################################
      
  ##### Set up ------------------------
    rm(list = ls(all = TRUE)) #clean out workspace
  
    ## --- R ggsave latex command ----------------------
    #'Saves a ggplot graphic to a file and creates the code to include it in a
    #'LaTeX document.
    #'
    #'Saves a ggplot graphic to a file and creates the code to include it in a
    #'LaTeX document.
    #'
    #'
    #'@param ...  arguments passed to the ggsave function
    #'(\code{\link[ggplot2]{ggsave}})
    #'@param caption The caption. Default to NULL, indicating no caption.
    #'@param label The label. Default to NULL, indicating no label.
    #'@param figure.placement The placement of the figure. Default to "hbt".
    #'@param floating Logical. Indicates if the figure should be placed in a
    #'floating environment. Default to TRUE
    #'@param caption.placement Should the caption be on top or bottom of the
    #'figure. Default to "bottom"
    #'@param latex.environments Alignment of the figure. Default to "center".
    #'@return The graphic will be saved to a plot and the relevant LaTeX code is
    #'printed.
    #'@author Thierry Onkelinx \email{Thierry.Onkelinx@@inbo.be}, Paul Quataert
    #'@seealso \code{\link[ggplot2]{ggsave}}
    #'@keywords hplot graphs
    #'@examples
    #'
    #'	require(ggplot2)
    #'  data(cars)
    #'	p <- ggplot(cars, aes(x = speed, y = dist)) + geom_point()
    #'	ggsave.latex(p, filename = "test.pdf", label = "fig:Cars", 
    #'    caption = "A demo plot", height = 5, width = 4)
    #'
    #'@export
    #'@importFrom ggplot2 ggsave
    ggsave.latex <- function(..., caption = NULL, label = NULL, figure.placement = "hbt", floating = TRUE, caption.placement="bottom", latex.environments="center"){
      ggsave(...)
      
      cat("\n\n")
      if(floating){
        cat("\\begin{figure}[", figure.placement, "]\n", sep = "")
      }
      cat("    \\begin{", latex.environments,"}\n", sep = "")
      if(!is.null(caption) && caption.placement == "top"){
        cat("        \\caption{", caption, "}\n", sep = "")
      }
      args <- list(...)
      if(is.null(args[["filename"]])){
        if(is.null(args[["plot"]])){
          names(args)[which(names(args) == "")[1]] <- "plot"
        }
        args[["filename"]] <- paste(args[["path"]], ggplot2:::digest.ggplot(args[["plot"]]), ".pdf", sep="")
      } else {
        args[["filename"]] <- paste(args[["path"]], args[["filename"]], sep="")
      }
      
      if(is.null(args[["width"]])){
        if(is.null(args[["height"]])){
          cat(
            "        \\includegraphics[height = 7in, width = 7in]{",
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        } else {
          cat(
            "        \\includegraphics[height = ", 
            args[["height"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            ", width = 7in]{", 
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        }
      } else {
        if(is.null(args[["height"]])){
          cat(
            "        \\includegraphics[height = 7in, width = 7 ", 
            args[["width"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            "]{", 
            args[["filename"]], 
            "}\n", 
            sep = "")
        } else {
          cat(
            "        \\includegraphics[height = ", 
            args[["height"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            ", width = ", 
            args[["width"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            "]{", 
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        }
      }
      if(!is.null(caption) && caption.placement == "bottom"){
        cat("        \\caption{", caption, "}\n", sep = "")
      }
      if(!is.null(label)){
        cat("        \\label{", label, "}\n", sep = "")
      }
      cat("    \\end{", latex.environments,"}\n", sep = "")
      if(floating){
        cat("\\end{figure}\n")
      }
      cat("\n\n")
    }
    
    # Load libraries
      library(ggplot2)
      library(RColorBrewer)
      library(plyr) #plyr should be loaded before dplyr
      library(dplyr)
      library(zoo) #for working with time and date variables
      library(lubridate)
      library(reshape2)
      library(data.table) #to work with large datasets
      library(readr)
      library(tidyr)
      library(scales)
      library(viridis)
      library(pROC) # for sensitivity and specificity
      library(xtable)
    
    # Load data
        # !!! modify this file name based on the name of your 5b output file
        load("./cwa/output/full_output_5b.1.Rdata")
        testB <- test
        full.predictionsB  <- full.predictions
        prop.scoresB <- propensity.scores
        rm(propensity.scores, full.predictions, test)
        
        # !!! modify this file name based on the name of your 5c output file
        load("./cwa/output/full_output_5c.1.Rdata")
        testC <- test
        full.predictionsC  <- full.predictions
        prop.scoresC <- propensity.scores
        rm(propensity.scores, full.predictions, test)
  

  ##### Generate Variable for Direction of Change from B to C
  
    BC.comparison <- 
      cbind.data.frame(B.predictions = testB$pscore.forest, 
                       C.predictions = testC$pscore.forest,
                       DV.numeric = testB$DV, DV.factor = as.factor(testB$DV), 
                       self.reported.viols = testC$e90.1yr,
                       D90s = testC$d90.1yr, D80s = testC$d80.1yr) %>%
      mutate(CminusB = C.predictions - B.predictions) %>%
      mutate(improvement = ifelse(CminusB > 0 & DV.numeric == 1, 1, 0)) %>%
      mutate(improvement = ifelse(CminusB < 0 & DV.numeric == 0, 1, improvement))
    
  ##### Produce fig 4 comparing B and C predictions --------------------
    
    #C vs. B by fails/no fails 
      fails <- BC.comparison[BC.comparison$DV.numeric == 1,]
      passes <- BC.comparison[BC.comparison$DV.numeric == 0,]
      
    #Plot 
      pdf("./cwa/output/figs/passfail.BCcomparison.081218.pdf", height = 7, width = 7.0867)
      
      par(mfrow = c(2,1), bty = "l", mar = c(5, 4, .1, 2))
      comparison.cols <- alpha(viridis(1000)[as.numeric(cut(fails$CminusB, breaks = 1000))], .7)
      plot(NULL, type = "p", xlab = "Predictions without self-reports", ylab = "Predictions with self-reports", xlim = c(0, .85), ylim = c(0,.85),
           axes = TRUE)
      points(x = fails$B.predictions, y = fails$C.predictions, 
             col = comparison.cols,
             pch = 16)
      legloc <- c(.5, .8)  #location of legend on x-axis
      xx <- seq(legloc[1], legloc[2], length.out = 1000)
      segments(xx, .05, xx, .15, col = viridis(1000))  #draw colors in legend
      text(.45, .01, "Reduce accuracy", cex = 0.7, pos = 4)
      text(.7, .01, "Improve accuracy", cex = 0.7, pos = 4)
      text(0, .75, "Failed Inspections", pos = 4)
      segments(x0 = -1, y0 = -1, x1 = 1, y1 = 1, col = "gray40")
      comparison.cols <- alpha(viridis(1000, direction = -1)[as.numeric(cut(passes$CminusB, breaks = 1000))], .7)
      
      plot(NULL, type = "p", xlab = "Predictions without self-reports", ylab = "Predictions with self-reports", xlim = c(0, .85), ylim = c(0,.85),
           axes = TRUE)
      points(x = passes$B.predictions, y = passes$C.predictions, 
             col = comparison.cols,
             pch = 16)
      legloc <- c(.5, .8)  #location of legend on x-axis
      xx <- seq(legloc[1], legloc[2], length.out = 1000)
      segments(xx, .05, xx, .15, col = viridis(1000))  #draw colors in legend
      text(.45, .01, "Reduce accuracy", cex = 0.7, pos = 4)
      text(.7, .01, "Improve accuracy", cex = 0.7, pos = 4)
      text(0, .75, "Passed Inspections", pos = 4)
      segments(x0 = -1, y0 = -1, x1 = 1, y1 = 1, col = "gray40")
      
      dev.off()
  
  ###### Create table with MSEs and AUCs comparing B and C -----------------------
  
    getMSE <- function(predicted, actual) {
      MSE <- mean((actual - predicted)^2)
      results <- list(residuals, MSE)
      return(MSE)
    }
    
    pred.violations.B <- list(testB$pscore.logit,
                              testB$pscore.lasso,
                              testB$pscore.en,
                              testB$pscore.tree,
                              testB$pscore.forest)
    
    MSElist.B <- lapply(pred.violations.B, getMSE, testB$DV) 
    
    pred.violations.C <- list(testC$pscore.logit,
                              testC$pscore.lasso,
                              testC$pscore.en,
                              testC$pscore.tree,
                              testC$pscore.forest)
    
    MSElist.C <- lapply(pred.violations.C, getMSE, testC$DV) 
    
    getAUC <- function(pred, actual) {
      auc(response = actual, predictor = pred)
    }
    
    auc.B <- sapply(pred.violations.B, function(predictions) auc(testB$DV, predictions))
    AUC.B <- round(unlist(auc.B), 3)
    
    auc.C <- sapply(pred.violations.C, function(predictions) auc(testC$DV, predictions))
    AUC.C <- round(unlist(auc.C), 3)
    
  ##### Combine MSEs and AUCs into a table for export -------------------------------------------

    MSE.B <- round(unlist(MSElist.B), 4)
    MSE.C <- round(unlist(MSElist.C), 4)
    BC.MSE.AUC.table <- cbind(MSE.B, MSE.C, AUC.B, AUC.C) 
    rnames <- c("Logit", "Lasso", "Elastic Net", "Single Tree",
                "Regression Forest")
    rownames(BC.MSE.AUC.table) <- rnames
    colnames(BC.MSE.AUC.table) <- c("MSE w/o DMR data", "MSE w/DMR data", "AUC w/o DMR data", "AUC w/DMR data")
    BC.table <- xtable(BC.MSE.AUC.table, align = "lcccc", digits = c(0, 4,4,4,4))
    label(BC.table) <- "tab:BC.table"
    caption(BC.table) <- "MSE and AUC comparison with and without self-reported data"
    
    #DATE is in the output label
    print(BC.table, file = paste("./cwa/output/BC_MSE_AUC_081218.tex", sep = ""), caption.placement = "top", 
          sanitize.text.function = function(x){x}, include.colnames = T, include.rownames = T, 
          booktabs = T, label = "tab:BC.table")

####################################################
# 12. Generate summary stats table 
####################################################
  print("gen summary stats")
  
  ##### Set Up ---------------------------
    rm(list = ls(all = TRUE)) #clean out workspace
    
    # Load packages --------------------------------------------------------
      library(pastecs) #for summary stats
      library(gdata) # for arranging variable names
      library(xtable)
      library(readr) #for read_csv
      library(data.table)
      library(plyr)
      library(dplyr)
      library(tibble)
      library(lubridate)
      library(scales)
      library(tidyverse)
      library(broom) #for tidy
      library(mosaic) # for favstats
      # remove sig figs: 
      options(scipen = 999)
    
    ## --- R ggsave latex command ----------------------
    #'Saves a ggplot graphic to a file and creates the code to include it in a
    #'LaTeX document.
    #'
    #'Saves a ggplot graphic to a file and creates the code to include it in a
    #'LaTeX document.
    #'
    #'
    #'@param ...  arguments passed to the ggsave function
    #'(\code{\link[ggplot2]{ggsave}})
    #'@param caption The caption. Default to NULL, indicating no caption.
    #'@param label The label. Default to NULL, indicating no label.
    #'@param figure.placement The placement of the figure. Default to "hbt".
    #'@param floating Logical. Indicates if the figure should be placed in a
    #'floating environment. Default to TRUE
    #'@param caption.placement Should the caption be on top or bottom of the
    #'figure. Default to "bottom"
    #'@param latex.environments Alignment of the figure. Default to "center".
    #'@return The graphic will be saved to a plot and the relevant LaTeX code is
    #'printed.
    #'@author Thierry Onkelinx \email{Thierry.Onkelinx@@inbo.be}, Paul Quataert
    #'@seealso \code{\link[ggplot2]{ggsave}}
    #'@keywords hplot graphs
    #'@examples
    #'
    #'	require(ggplot2)
    #'  data(cars)
    #'	p <- ggplot(cars, aes(x = speed, y = dist)) + geom_point()
    #'	ggsave.latex(p, filename = "test.pdf", label = "fig:Cars", 
    #'    caption = "A demo plot", height = 5, width = 4)
    #'
    #'@export
    #'@importFrom ggplot2 ggsave
    ggsave.latex <- function(..., caption = NULL, label = NULL, figure.placement = "hbt", floating = TRUE, caption.placement="bottom", latex.environments="center"){
      ggsave(...)
      
      cat("\n\n")
      if(floating){
        cat("\\begin{figure}[", figure.placement, "]\n", sep = "")
      }
      cat("    \\begin{", latex.environments,"}\n", sep = "")
      if(!is.null(caption) && caption.placement == "top"){
        cat("        \\caption{", caption, "}\n", sep = "")
      }
      args <- list(...)
      if(is.null(args[["filename"]])){
        if(is.null(args[["plot"]])){
          names(args)[which(names(args) == "")[1]] <- "plot"
        }
        args[["filename"]] <- paste(args[["path"]], ggplot2:::digest.ggplot(args[["plot"]]), ".pdf", sep="")
      } else {
        args[["filename"]] <- paste(args[["path"]], args[["filename"]], sep="")
      }
      
      if(is.null(args[["width"]])){
        if(is.null(args[["height"]])){
          cat(
            "        \\includegraphics[height = 7in, width = 7in]{",
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        } else {
          cat(
            "        \\includegraphics[height = ", 
            args[["height"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            ", width = 7in]{", 
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        }
      } else {
        if(is.null(args[["height"]])){
          cat(
            "        \\includegraphics[height = 7in, width = 7 ", 
            args[["width"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            "]{", 
            args[["filename"]], 
            "}\n", 
            sep = "")
        } else {
          cat(
            "        \\includegraphics[height = ", 
            args[["height"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            ", width = ", 
            args[["width"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            "]{", 
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        }
      }
      if(!is.null(caption) && caption.placement == "bottom"){
        cat("        \\caption{", caption, "}\n", sep = "")
      }
      if(!is.null(label)){
        cat("        \\label{", label, "}\n", sep = "")
      }
      cat("    \\end{", latex.environments,"}\n", sep = "")
      if(floating){
        cat("\\end{figure}\n")
      }
      cat("\n\n")
    }
    
    
  ##### Load Data -------------------------------------------------------------------
  
    # Random training and test set data (fulloutput5)
      load("./cwa/data/modified_data/full.random.5sets.Rdata")
    
    # Only keep test5.random and train5.random
      rm(test.random, test2.random, test3.random, test4.random)
      rm(train.random, train2.random, train3.random, train4.random)
      
    # merge train5random and test5.random together
      all_facs <- full_join(train5.random, test5.random)
      test <- test5.random
      train <- train5.random
      
      all_facs$insp_group <- ifelse(all_facs$cat ==  "INS", 1, 0)
      all_facs$fail.insp <- all_facs$DV
    
    
  ##### Format the covars we want to represent
    covariate.names.add <- c("time.since.insp",
                             "FAC_STATE", #"pscore.forest", 
                             "FAC_CHESAPEAKE_BAY_FLG", 
                             "FAC_INDIAN_CNTRY_FLG", "FAC_US_MEX_BORDER_FLG", 
                             "FAC_FEDERAL_FLG", "FAC_PERCENT_MINORITY", "FAC_POP_DEN", 
                             "AIR_FLAG", "SDWIS_FLAG", "RCRA_FLAG",
                             "TRI_FLAG", "GHG_FLAG", "FAC_IMP_WATER_FLG", 
                             "EJSCREEN_FLAG_US", "multiple_IDs","n_IDs", 
                             "SIC_AG_f", "SIC_MINE_f", "SIC_CONS_f", "SIC_MANU_f", "SIC_UTIL_f", 
                             "SIC_WHOL_f", "SIC_RETA_f", "SIC_FINA_f", 
                             "SIC_SERV_f", "SIC_PUBL_f", 
                             "num.facilities.cty", "num.facilities.st", "Party", 
                             "PERMIT_MAJOR", "PERMIT_MINOR", "prox.1yr",
                             "prox.2yr", "prox.5yr", #"fail.insp",
                             "insp_group")
  
    long.names <-  c("Days Since Inspection",
                     "State",
                     #"Risk Score",
                     "Chesapeake Bay", 
                     "Native Amer. Boundary",
                     "US-Mexican Border",
                     "Federal Facility", 
                     "Percent Minority", "Population Density",
                     "CAA Permit", "SDWIS Permit","RCRA Permit",  
                     "TRI Permit", "GHG Monitoring",  #% Is this actually a GHG permit? or registry number?
                     "Impaired Watershed",  "Env. Justice Index", "Multiple NPDES Permits", "Number of NPDES Permits", 
                     "Ag, Forestry, Fishing", "Mining", "Construction",
                     "Manufacturing", "Utilities and Transport", 
                     "Wholesale Trade", "Retail Trade", "Finance, Insurance, Real Estate", 
                     "Services", "Public Administration", 
                     "Total Facilities in County", "Total Facilities in State", "Governor Party 2017", # governor party will be exactly collinear w/ state
                     "Major Permit", "Minor Permit", "Proximate Inspections (1 yr)", 
                     "Proximate Inspections (2 yrs)", "Proximate Inspections (5 yrs)",
                     #"Inspection Failure Rate", 
                     "Inspected Group")
    
    
    mydata <- all_facs[ ,c(names(all_facs) %in% covariate.names.add)]
    converter <- as.data.frame(cbind(covariate.names.add, long.names))
    
    
  ##### Convert vars into countable/numeric formats
  
    #remove factor variables
      remove.factors <- c("FAC_STATE", "Party", "score.cut")
      mydata <- mydata[ ,c(!names(mydata) %in% remove.factors)]
    
    #convert logicals to numerics
      logicals <- sapply(mydata, is.logical)
      mydata[,logicals] <- lapply(mydata[,logicals], as.numeric)
    
    #convert the relevant chars to numerics
      chartonumeric <- c("FAC_INDIAN_CNTRY_FLG", "FAC_FEDERAL_FLG", 
                         "FAC_US_MEX_BORDER_FLG", "FAC_CHESAPEAKE_BAY_FLG",
                         "AIR_FLAG", "SDWIS_FLAG", "RCRA_FLAG",
                         "TRI_FLAG", "GHG_FLAG", "FAC_IMP_WATER_FLG",
                         "EJSCREEN_FLAG_US")
      
      mydata[,chartonumeric] <- 
        mydata %>%
        select(chartonumeric) %>% 
        mutate_all(function(x) as.numeric(mapvalues(x, from = c("N", "Y"), to = c(0, 1)))) 
    
  # get the long names from the covariates that are in the final summary stats table
    covariate.names.add.labels <- as.character(converter$long.names[converter$covariate.names.add %in% names(mydata)])
  
  # get the n
    n <- formatC(length(mydata$insp_group), big.mark = ","); n
  
  # Set column labels
    colnames.sumstats <- c("Variable.Name", "Mean", "SD", "Min", "Max" , "Median")
  
  ##### Create & Use Summary Function ----------
    summary_df <- function(df, appendage){
      # Generate the summary stats table & remove insp_group
      sumstats_alltest <- 
        do.call(rbind, dfapply(df, favstats)) %>%
        rownames_to_column() 
      
      # Translate into format we want, with readable names
      sumtest <- sumstats_alltest %>%
        select(mean, sd, min, max) %>%  
        mutate(VarName = covariate.names.add.labels) %>%
        select(VarName, everything()) %>% # move the last col name to begining %>%
        `colnames<-`(c(colnames.sumstats[1], paste(colnames.sumstats[2:length(.)], appendage, sep = ".")))
      return(sumtest)
    }
    
    sumtest <- summary_df(mydata, "all")
    
  
  ##### Extract Vars with Large Numbers to left of decimal
    options(digits = 0)
    
    cont_num_vars_2 <- c("Percent Minority", "Population Density",
                         "Number of NPDES Permits", "Total Facilities in County", 
                         "Total Facilities in State", "Proximate Inspections (1 yr)", 
                         "Proximate Inspections (2 yrs)", "Proximate Inspections (5 yrs)" )      
    
    options(digits = 1)
    sum_cont <- sumtest[which(sumtest$`Variable.Name` %in% cont_num_vars_2),]; sum_cont
    
    # Extract & gen table with vars where right of decimal matters
      '%ni%' <- Negate('%in%') # define opposite of %in%
      options(digits = 2)
      sum_01scale <-  sumtest[which(sumtest$`Variable.Name` %ni% cont_num_vars_2),]; sum_01scale
      
    # Convert the tables to characters and rbind them together (preserve decimal point values). 
    
      # left of decimal
      sum_cont_char <-  
        sum_cont %>% 
        arrange(desc(Mean.all)) %>%
        mutate(Mean.all = formatC(Mean.all, format = "f", digits = 0, drop0trailing = FALSE, big.mark = ",")) %>% 
        mutate(SD.all = formatC(SD.all, format = "f", digits = 1, drop0trailing = FALSE, big.mark = ",")) %>% 
        mutate(Min.all = formatC(Min.all, format = "f", digits = 0, drop0trailing = T)) %>% 
        mutate(Max.all = formatC(Max.all, format = "f", digits = 0, drop0trailing = T, big.mark = ",")) %>%
        as.data.frame; sum_cont_char 
      
      # right of decimal
      sum_01scale_char <- 
        sum_01scale %>% 
        mutate(Mean.all = formatC(Mean.all, format = "f", digits = 2, drop0trailing = FALSE)) %>% 
        mutate(SD.all = formatC(SD.all, format = "f", digits = 2, drop0trailing = FALSE)) %>% 
        mutate(Min.all = formatC(Min.all, format = "f", digits = 2, drop0trailing = T)) %>% 
        mutate(Max.all = formatC(Max.all, format = "f", digits = 1, drop0trailing = T)) %>%
        as.data.frame; sum_01scale_char
      
    # Bind the two tables together, and remove the vars we don't need/want
    
    summary.bind <- 
      rbind(sum_cont_char, sum_01scale_char) %>%
      mutate(MinMax = paste0("(", Min.all, " - ", Max.all, ")")) %>%
      select(-Min.all, -Max.all) %>%
      filter(Variable.Name != "Risk Score", Variable.Name != "Inspection Failure Rate", Variable.Name != "Inspected Group") %>%
      `colnames<-`(c("Variable Name", "Mean", "SD", "(Min - Max)")); summary.bind
      
  ##### Export Combined Table 1: tab_summary_bind -----------------------------------------
  
    tab_summary_bind <- xtable(summary.bind,label = "tab:summary_bind",
                               caption = paste0("Descriptive Statistics for Covariates", 
                                                " of complete dataset",
                                                " (n = ", n, ")")
    )
    
    
    print.xtable(tab_summary_bind,
                 file = "./cwa/output/tab_summary_bind.tex",
                 floating = TRUE,
                 table.placement = "H",
                 include.rownames = FALSE,
                 caption.placement = "bottom"
    )

####################################################
# 13. Produce figure showing fail rate vs. num of insp -----------------------------------------------
####################################################
      
  ##### Set Up -------
    rm(list = ls(all = TRUE)) #clean out workspace
  
    # load packages
      library(reshape2)
      library(ggplot2)
      library(data.table)
      library(xtable)
      library(readr)
      library(Hmisc) #for cuts2
      library(lubridate) # for dates
      library(tidyverse)
      library(dplyr)
      library(RColorBrewer)
      library(rlang)
  
    #set seed for replicable results
      set.seed(36)
  
  # Write function for extracting failure rates -----
    score_table <- function(df, indicator, fail.ind){ 
      require(dplyr)
      indicator <- enquo(indicator)      # Create quosure
      fail.ind  <- enquo(fail.ind)       # Create quosure     
      score_rates <- 
        df %>%
        group_by(score.cut) %>%
        summarise(
          facs = length(unique(REGISTRY_ID)),
          insp = sum(UQ(indicator)),                    # Use UQ to unquote
          fails = sum(UQ(indicator) * UQ(fail.ind)),    # Use UQ to unquote
          fail.rate = ifelse(insp == 0, 0, fails/insp)) # Avoid infinity
      return(score_rates)
    }
    
  

  ##### Load Data -----

      
    # 1. Inspections
      inspections <- read_csv("./cwa/data/raw_data/npdes_downloads/NPDES_INSPECTIONS.csv")
    
    # 2. Result by Score Cut
      load("./cwa/output/results.by.score.cut.Rdata")
    
    # 3. Final Analysis Set (20% test + all uninspected faciltiies)
      load("./cwa/output/full.predictions.5a.5.final.Rdata")
    
    # load test and 20 pct uninsp
      load("./cwa/output/test.and.20pct.uninsp.Rdata")
    
    
    # 4. Load Our #5 Random Test/Train Set to get full number of facilities and shares
      load("./cwa/data/modified_data/full.random.5sets.Rdata")
      rm(test.random, test2.random, test3.random, test4.random)
      rm(train.random, train2.random, train3.random, train4.random)
      
      all_facs <- full_join(train5.random, test5.random)
      # train <- train5.random %>% filter(cat == "INS")
      test <- test5.random %>% filter(cat == "INS") 
      uninsp <- all_facs %>% filter(cat != "INS")
      
      N_allfacs <- length(all_facs$REGISTRY_ID)
    
  ##### Establish inspection levels ---------------------------------------------
    
    #first join all.facs with inspections so each inspection has a state attached
      testand20pct.merge <- test.and.20pct.uninsp[, c("REGISTRY_ID", "FAC_STATE")]
      inspections.w.state <- inner_join(inspections, testand20pct.merge)
      #note: this reduces the # of inspections to just the inspections of facilities in our 
      #test + 20% uninsp data, ensuring allocations don't assign to facilities that don't exist in our data
      
    #1. how many inspections of unique facilities per year, from 2012-2016?
      insp.summary <- inspections.w.state %>% mutate(year.insp = year(mdy(ACTUAL_END_DATE))) %>%
        filter(year.insp >= 2012 & year.insp <= 2016) %>%
        distinct(REGISTRY_ID, year.insp, .keep_all = TRUE) %>% 
        group_by(year.insp) %>% summarise(annual_insp = n())
      round(mean(insp.summary$annual_insp)) #5,481 inspections of unique fac in our reallocation set per year, on average
      
    # 2. Determine distribution of inspections across states
    
      insp.by.state <- inspections.w.state %>% mutate(year.insp = year(mdy(ACTUAL_END_DATE))) %>%
        filter(year.insp >= 2012 & year.insp <= 2016) %>%
        distinct(REGISTRY_ID, year.insp, .keep_all = TRUE) %>%
        group_by(year.insp, FAC_STATE) %>% summarise(st.insp = n())
      
      avg_yr_state <- insp.by.state %>% group_by(FAC_STATE) %>%
        summarise(stavg = round(mean(st.insp)))
      
    # create cwa.join, Rename vars for consistency with later script & remove extra columns
      cwa.join <-  
        test.and.20pct.uninsp %>%
        select(REGISTRY_ID, cat, ACTUAL_END_DATE, X1, FAC_STATE, FAC_LAT, FAC_LONG,
               PERMIT_MAJOR, PERMIT_MINOR, DV, prop)
      
      cwa.join$score.cut <- as.character(cut2(cwa.join$prop, cuts = seq(0,1,0.05), digits = 2))
      cwa.join$insp_group <- ifelse(cwa.join$cat == "INS", 1, 0)
      cwa.join$fail.insp <-  with(cwa.join,
                                  ifelse(insp_group == 0 , 0 , DV)
      ) 
      
      run.results <- score_table(cwa.join, insp_group, fail.insp); run.results 
    
    
      cwa.join <- cwa.join %>% arrange(desc(prop))
      unq.cuts <- unique(cwa.join$score.cut)
      min.prob <- 0.01
      
    # initialize columns
      cwa.join$predicted.DV <- 0
      cwa.join$predicted.DV.LB <- 0
      cwa.join$predicted.DV.UB <- 0
      
  ##### Define predicted inspection fail/pass based on 500 runs of fail rates' mean, upper bound, lower bound
      
    for (i in 1:length(unq.cuts)) {
      cut.grp <- as.character(run.results$score.cut[i])
      num.fac <- run.results$facs[run.results$score.cut == cut.grp]
      
      num.fails <- round(run.results$facs[run.results$score.cut == cut.grp] * 
                           results.by.score.cut$fail.rate.means[results.by.score.cut$score.cut == cut.grp])
      
      LB.num.fails <- max(round(run.results$facs[run.results$score.cut == cut.grp] * 
                                  results.by.score.cut$fail.rate.LB[results.by.score.cut$score.cut == cut.grp]), 0)
      
      UB.num.fails <- min(round(run.results$facs[run.results$score.cut == cut.grp] * 
                                  results.by.score.cut$fail.rate.UB[results.by.score.cut$score.cut == cut.grp]), num.fac)
      
      
      smpl.UB <- sample(num.fac, UB.num.fails, replace = F)
      smpl.mean <- sample(smpl.UB, num.fails, replace = F)
      smpl.LB <- sample(smpl.mean, LB.num.fails, replace = F)
      
      assignment.mean <- rep(0, num.fac)
      assignment.mean[smpl.mean] <- 1
      cwa.join$predicted.DV[cwa.join$score.cut == cut.grp] <- assignment.mean
      
      assignment.UB <- rep(0, num.fac)
      assignment.UB[smpl.UB] <- 1
      cwa.join$predicted.DV.UB[cwa.join$score.cut == cut.grp] <- assignment.UB
      
      assignment.LB <- rep(0, num.fac)
      assignment.LB[smpl.LB] <- 1
      cwa.join$predicted.DV.LB[cwa.join$score.cut == cut.grp] <- assignment.LB
    }
    
  
  ##### Loop through the number of inspections, assigning and calculating inspection failure rates ------------------------
  ##### This loop covers inspections up to the observed number
  
    fac.prop.seq <- seq(0.001, 1, by = .001)
    #initialize container for outputs
    all.failure.outputs <- as.data.frame(matrix(NA_real_, nrow = length(fac.prop.seq), ncol = 17))
    
    for (k in 1:length(fac.prop.seq)) {
      facility.proportion <- fac.prop.seq[k]
      print(facility.proportion)
      avg_yr_state$allocation.key <- round(avg_yr_state$stavg * facility.proportion)
      
      N_insp <- sum(avg_yr_state$allocation.key) 
      all.failure.outputs[k ,1] <- N_insp
      
      # ---- Conduct Facility Inspection "Assignment" -------
    
        
        #base allocation --------------
          run.results$base <- round(run.results$insp / sum(run.results$insp) * N_insp) 
          #base allocation is by share of observed inspections
          #Note: in the final figure, base allocation is plotted as the current base rate, 
          #not as a "mimicked" base allocation as is done here
        
          cwa.join$base.assigned <- 0
          
          for (i in 1:length(unq.cuts)) {
            cut.grp <- run.results$score.cut[i]
            num.insp <- run.results$base[i]
            num.fac <- run.results$facs[i]
            smpl <- sample(num.fac, num.insp, replace = F)
            assignment <- rep(0, num.fac)
            assignment[smpl] <- 1
            cwa.join$base.assigned[cwa.join$score.cut == cut.grp] <- assignment
          }
        
        #r1: based on risk score --------------------------
          cwa.join <- cwa.join %>% mutate(r1.assigned = ifelse(row_number() <= N_insp, 1,0 )) 
          sum(cwa.join$r1.assigned) == N_insp 
          
        #r2: based on risk score by state ________________
          cwa.join$r2.assigned <- 0
          
          for (i in 1:nrow(avg_yr_state)) {
            cwa.join[cwa.join$FAC_STATE == avg_yr_state$FAC_STATE[i],] <- 
              cwa.join %>% 
              filter(FAC_STATE == avg_yr_state$FAC_STATE[i]) %>% 
              mutate(r2.assigned = ifelse(row_number() <= avg_yr_state$allocation.key[i], 1, 0)) 
          }
          sum(cwa.join$r2.assigned) == N_insp 
          
        #r3: minimum probability, followed by risk score ------------------
          cwa.join$r3.assigned <- 0
          r3.assign.df <- cwa.join[, c("REGISTRY_ID", "prop")]
          min.prob.total.alloc <- round(min.prob * nrow(cwa.join))
          smpl.min.prob <- sample(nrow(cwa.join), min.prob.total.alloc, replace = F)
          r3.assign.df$minprob <- 0
          r3.assign.df$minprob[smpl.min.prob] <- 1
          
          insp.left.for.risk <- N_insp - min.prob.total.alloc
          
          to.be.assigned <- 
            r3.assign.df %>% 
            filter(minprob == 0) %>% 
            mutate(risk.assigned = ifelse(row_number() <= insp.left.for.risk, 1,0 ))
          
          r3.assign.df$risk <- 0
          r3.assign.df$risk[r3.assign.df$minprob == 0] <- to.be.assigned$risk.assigned
          r3.assign.df$r3.assigned <- r3.assign.df$minprob + r3.assign.df$risk
          
          cwa.join$r3.assigned <- r3.assign.df$r3.assigned
          sum(cwa.join$r3.assigned) == N_insp 
        
        #r4: state-by-state, going first with majors, then random assignment, then risk-score -------------
          cwa.join$r4.assigned <- 0
          r4.assign.df <- cwa.join[, c("REGISTRY_ID", "prop", "PERMIT_MAJOR", "FAC_STATE")]
          r4.assign.df$major <- 0
          r4.assign.df$minor <- 0
          r4.assign.df$risk <- 0
          
          for (i in 1:nrow(avg_yr_state)) {
            st <- avg_yr_state$FAC_STATE[i]
            st.insp <- avg_yr_state$allocation.key[i]
            st.facs <- r4.assign.df %>% filter(FAC_STATE == st)
            
            st.majors <- st.facs %>% filter(PERMIT_MAJOR == TRUE)
            major.insp.total <- min(st.insp, round(nrow(st.majors)/2))
            smpl.major <- sample(nrow(st.majors), major.insp.total, replace = F)
            assignment.major <- rep(0, nrow(st.majors))
            assignment.major[smpl.major] <- 1
            r4.assign.df$major[r4.assign.df$FAC_STATE == st & r4.assign.df$PERMIT_MAJOR == TRUE] <- assignment.major
            
            st.minors <- st.facs %>% filter(PERMIT_MAJOR == FALSE)
            minor.insp.total <- min(st.insp-major.insp.total, round(nrow(st.minors)*min.prob))
            smpl.minor <- sample(nrow(st.minors), minor.insp.total, replace = F)
            assignment.minor <- rep(0, nrow(st.minors))
            assignment.minor[smpl.minor] <- 1
            r4.assign.df$minor[r4.assign.df$FAC_STATE == st & r4.assign.df$PERMIT_MAJOR == FALSE] <- assignment.minor
            
            st.risk <- r4.assign.df %>% filter(FAC_STATE == st & major == 0 & minor == 0) 
            risk.insp.total <- max(st.insp - major.insp.total - minor.insp.total, 0)
            assignment.risk <- c(rep(1, risk.insp.total), rep(0, max(0, nrow(st.risk) - risk.insp.total)))
            r4.assign.df$risk[r4.assign.df$FAC_STATE == st & r4.assign.df$major == 0 & r4.assign.df$minor == 0] <- assignment.risk
          }
          
          
          r4.assign.df$r4.assigned <- r4.assign.df$major + r4.assign.df$minor + r4.assign.df$risk
          cwa.join$r4.assigned <- r4.assign.df$r4.assigned
          sum(cwa.join$r4.assigned) == N_insp
        
      #Aggregate outputs for this particular inspection number
        outputs <- c(facility.proportion, cwa.join$predicted.DV %*% cwa.join$base.assigned,
                     cwa.join$predicted.DV.LB %*% cwa.join$base.assigned, 
                     cwa.join$predicted.DV.UB %*% cwa.join$base.assigned, 
                     cwa.join$predicted.DV %*% cwa.join$r1.assigned, 
                     cwa.join$predicted.DV.LB %*% cwa.join$r1.assigned, 
                     cwa.join$predicted.DV.UB %*% cwa.join$r1.assigned,
                     cwa.join$predicted.DV %*% cwa.join$r2.assigned,
                     cwa.join$predicted.DV.LB %*% cwa.join$r2.assigned,
                     cwa.join$predicted.DV.UB %*% cwa.join$r2.assigned, 
                     cwa.join$predicted.DV %*% cwa.join$r3.assigned,
                     cwa.join$predicted.DV.LB %*% cwa.join$r3.assigned,
                     cwa.join$predicted.DV.UB %*% cwa.join$r3.assigned,
                     cwa.join$predicted.DV %*% cwa.join$r4.assigned,
                     cwa.join$predicted.DV.LB %*% cwa.join$r4.assigned,
                     cwa.join$predicted.DV.UB %*% cwa.join$r4.assigned)
        
      #Store results
        all.failure.outputs[k,2:17] <- outputs
    }
    
    
    
  ##### Loop through the number of inspections, assigning and calculating inspection failure rates ------------------------
  ##### This loop covers inspections greater than observed
    
    test.20pct.by.state <- test.and.20pct.uninsp %>% group_by(FAC_STATE) %>% summarise(stfac = n())
    avg_yr_state <- left_join(avg_yr_state, test.20pct.by.state)
    avg_yr_state$leftover.fac <- avg_yr_state$stfac - avg_yr_state$stavg
    
    run.results$leftover.fac <- run.results$facs - run.results$insp
    
    fac.prop.seq <- seq(0.001, 1, by = .001)
    
    #initialize container for outputs
    all.failure.outputs.pt2 <- as.data.frame(matrix(NA_real_, nrow = length(fac.prop.seq), ncol = 17))
    
    for (k in 1:length(fac.prop.seq)) {
      facility.proportion <- fac.prop.seq[k]
      print(facility.proportion)
      avg_yr_state$allocation.key <-avg_yr_state$stavg + 
        round(avg_yr_state$leftover.fac * facility.proportion)
      
      N_insp <- sum(avg_yr_state$allocation.key) #5491
      all.failure.outputs.pt2[k ,1] <- N_insp
      
      # ---- Conduct Facility Inspection "Assignment" -------
      
        #base allocation --------------
        #Note: in the final figure, base allocation is plotted as the current base rate, 
        #not as a "mimicked" base allocation as is done here
        
        run.results$base <- round(run.results$insp / sum(run.results$insp) * N_insp)
        #base allocation is by share of observed inspections
        
        cwa.join$base.assigned <- 0
        
        #the base allocation is not proportional to the distribution of facilities, so 
        #at some point, we can no longer maintain the distribution of the base allocation
        if (sum(run.results$base < run.results$facs) == nrow(run.results)) {
          for (i in 1:length(unq.cuts)) {
            cut.grp <- run.results$score.cut[i]
            num.insp <- run.results$base[i]
            num.fac <- run.results$facs[i]
            smpl <- sample(num.fac, num.insp, replace = F)
            assignment <- rep(0, num.fac)
            assignment[smpl] <- 1
            cwa.join$base.assigned[cwa.join$score.cut == cut.grp] <- assignment
          }
        }
        
        
        #r1: based on risk score --------------------------
          cwa.join <- cwa.join %>% mutate(r1.assigned = ifelse(row_number() <= N_insp, 1,0 )) 
          sum(cwa.join$r1.assigned) == N_insp 
        
        #r2: based on risk score by state
          cwa.join$r2.assigned <- 0
          
          for (i in 1:nrow(avg_yr_state)) {
            cwa.join[cwa.join$FAC_STATE == avg_yr_state$FAC_STATE[i],] <- 
              cwa.join %>% 
              filter(FAC_STATE == avg_yr_state$FAC_STATE[i]) %>% 
              mutate(r2.assigned = ifelse(row_number() <= avg_yr_state$allocation.key[i], 1, 0)) 
          }
          sum(cwa.join$r2.assigned) == N_insp 
          
        #r3: minimum probability, followed by risk ------------------
          cwa.join$r3.assigned <- 0
          r3.assign.df <- cwa.join[, c("REGISTRY_ID", "prop")]
          min.prob.total.alloc <- round(min.prob * nrow(cwa.join))
          smpl.min.prob <- sample(nrow(cwa.join), min.prob.total.alloc, replace = F)
          r3.assign.df$minprob <- 0
          r3.assign.df$minprob[smpl.min.prob] <- 1
          
          insp.left.for.risk <- N_insp - min.prob.total.alloc
          
          to.be.assigned <- 
            r3.assign.df %>% 
            filter(minprob == 0) %>% 
            mutate(risk.assigned = ifelse(row_number() <= insp.left.for.risk, 1,0 ))
          
          r3.assign.df$risk <- 0
          r3.assign.df$risk[r3.assign.df$minprob == 0] <- to.be.assigned$risk.assigned
          r3.assign.df$r3.assigned <- r3.assign.df$minprob + r3.assign.df$risk
          
          cwa.join$r3.assigned <- r3.assign.df$r3.assigned
          sum(cwa.join$r3.assigned) == N_insp 
        
        # r4: state-by-state, going first with majors, then random assignment, then risk-score  -------------
          cwa.join$r4.assigned <- 0
          r4.assign.df <- cwa.join[, c("REGISTRY_ID", "prop", "PERMIT_MAJOR", "FAC_STATE")]
          r4.assign.df$major <- 0
          r4.assign.df$minor <- 0
          r4.assign.df$risk <- 0
          
          for (i in 1:nrow(avg_yr_state)) {
            st <- avg_yr_state$FAC_STATE[i]
            st.insp <- avg_yr_state$allocation.key[i]
            st.facs <- r4.assign.df %>% filter(FAC_STATE == st)
            
            st.majors <- st.facs %>% filter(PERMIT_MAJOR == TRUE)
            major.insp.total <- min(st.insp, round(nrow(st.majors)/2))
            smpl.major <- sample(nrow(st.majors), major.insp.total, replace = F)
            assignment.major <- rep(0, nrow(st.majors))
            assignment.major[smpl.major] <- 1
            r4.assign.df$major[r4.assign.df$FAC_STATE == st & r4.assign.df$PERMIT_MAJOR == TRUE] <- assignment.major
            
            st.minors <- st.facs %>% filter(PERMIT_MAJOR == FALSE)
            minor.insp.total <- min(st.insp-major.insp.total, round(nrow(st.minors)*min.prob))
            smpl.minor <- sample(nrow(st.minors), minor.insp.total, replace = F)
            assignment.minor <- rep(0, nrow(st.minors))
            assignment.minor[smpl.minor] <- 1
            r4.assign.df$minor[r4.assign.df$FAC_STATE == st & r4.assign.df$PERMIT_MAJOR == FALSE] <- assignment.minor
            
            st.risk <- r4.assign.df %>% filter(FAC_STATE == st & major == 0 & minor == 0) 
            risk.insp.total <- max(st.insp - major.insp.total - minor.insp.total, 0)
            assignment.risk <- c(rep(1, risk.insp.total), rep(0, max(0, nrow(st.risk) - risk.insp.total)))
            r4.assign.df$risk[r4.assign.df$FAC_STATE == st & r4.assign.df$major == 0 & r4.assign.df$minor == 0] <- assignment.risk
          }
          
          
          r4.assign.df$r4.assigned <- r4.assign.df$major + r4.assign.df$minor + r4.assign.df$risk
          cwa.join$r4.assigned <- r4.assign.df$r4.assigned
          sum(cwa.join$r4.assigned) == N_insp 
      
      #compile outputs for this number of inspections 
        outputs <- c(facility.proportion, cwa.join$predicted.DV %*% cwa.join$base.assigned,
                     cwa.join$predicted.DV.LB %*% cwa.join$base.assigned, 
                     cwa.join$predicted.DV.UB %*% cwa.join$base.assigned, 
                     cwa.join$predicted.DV %*% cwa.join$r1.assigned, 
                     cwa.join$predicted.DV.LB %*% cwa.join$r1.assigned, 
                     cwa.join$predicted.DV.UB %*% cwa.join$r1.assigned,
                     cwa.join$predicted.DV %*% cwa.join$r2.assigned,
                     cwa.join$predicted.DV.LB %*% cwa.join$r2.assigned,
                     cwa.join$predicted.DV.UB %*% cwa.join$r2.assigned, 
                     cwa.join$predicted.DV %*% cwa.join$r3.assigned,
                     cwa.join$predicted.DV.LB %*% cwa.join$r3.assigned,
                     cwa.join$predicted.DV.UB %*% cwa.join$r3.assigned,
                     cwa.join$predicted.DV %*% cwa.join$r4.assigned,
                     cwa.join$predicted.DV.LB %*% cwa.join$r4.assigned,
                     cwa.join$predicted.DV.UB %*% cwa.join$r4.assigned)
        
      all.failure.outputs.pt2[k,2:17] <- outputs
    }
  
  to.plot <- rbind(all.failure.outputs, all.failure.outputs.pt2)
  save(to.plot, file = "./cwa/output/to.plot.all.failure.outputs_081218.Rdata")
  
  ##### prepare for plot
    rates <- as.data.frame(sapply(to.plot[, 3:17], function(x) x/to.plot[,1]))
    rates$pct.insp <- to.plot$V1/nrow(cwa.join)
    names(rates)[1:15] <- c("base_mean", "base_lb", "base_ub", "r1_mean", "r1_lb", "r1_ub",
                            "r2_mean", "r2_lb", "r2_ub", "r3_mean", "r3_lb", "r3_ub", "r4_mean",
                            "r4_lb", "r4_ub")
    
    linecols <- brewer.pal(5, "Set1")
    polycols <- alpha(linecols, .2)
    
    #NOTE: to plot, use the observed failure rate for the base allocation (ignore the base.assigned cols)
    insp.only <- cwa.join %>% filter(cat == "INS")
    base.rate <- sum(insp.only$DV)/nrow(insp.only) * 100
    
  ##### produce plot ------------------------------------------------------------
  
    pdf("./cwa/output/figs/fails.vs.num.insp.pdf", width = 11, height = 8)
    plot(1, type="n", xlim = c(0,100), ylim = c(0,100), xlab = "% of facilities inspected", 
         ylab = "% of inspections that detect violations")
    # Add Lines + CI's
    for (i in 1:4) { #one for each allocation, which are cols 1-15 (groups of 3).
      #i = 1 corresponds to r4, counting down to r1 
      xx <- 100 * rates$pct.insp # add in x values
      yy <- 100 * rates[, 13 - 3 * (i - 1)] #this is the mean
      ub <- 100 * rates[, 15 - 3 * (i - 1)]
      lb <- 100 * rates[, 14 - 3 * (i - 1)]
      lines(xx, yy, col = linecols[i])
      polygon(c(xx, rev(xx)), c(lb, rev(ub)), border = NA, col = polycols[i])
    }
    
    segments(-1,base.rate, 101, base.rate, col = linecols[5], lty = "dashed")
    
    legloc <- c(70, 100)  #location of legend on y-axis
    leg.y <- seq(legloc[1],legloc[2],length.out= 5)
    segments(85, leg.y, 95, leg.y, col=linecols, lty = c("solid", "solid", "solid", "solid", "dashed"))  #draw colors in legend
    text(x = 84, y = leg.y, labels = c("Majors, deterrence, state",
                                       "Deterrence, national", 
                                       "Aggressive, state",
                                       "Aggressive, national",
                                       "Base"),
         cex=0.8, adj = c(1,0.5))
    dev.off()

####################################################
# 14. Produce quintile-covariate plot -----------------------------------------
####################################################
  
  ##### Set up 
    rm(list = ls(all = TRUE)) 
  
    # load packages 
      library(reshape2)
      library(ggplot2)
      library(plyr)
      library(dplyr)
      library(gdata) 
      library(ggridges) 
      library(forcats) 
      library(gtools) 
    
    ## --- R ggsave latex command ----------------------
    #'Saves a ggplot graphic to a file and creates the code to include it in a
    #'LaTeX document.
    #'
    #'Saves a ggplot graphic to a file and creates the code to include it in a
    #'LaTeX document.
    #'
    #'
    #'@param ...  arguments passed to the ggsave function
    #'(\code{\link[ggplot2]{ggsave}})
    #'@param caption The caption. Default to NULL, indicating no caption.
    #'@param label The label. Default to NULL, indicating no label.
    #'@param figure.placement The placement of the figure. Default to "hbt".
    #'@param floating Logical. Indicates if the figure should be placed in a
    #'floating environment. Default to TRUE
    #'@param caption.placement Should the caption be on top or bottom of the
    #'figure. Default to "bottom"
    #'@param latex.environments Alignment of the figure. Default to "center".
    #'@return The graphic will be saved to a plot and the relevant LaTeX code is
    #'printed.
    #'@author Thierry Onkelinx \email{Thierry.Onkelinx@@inbo.be}, Paul Quataert
    #'@seealso \code{\link[ggplot2]{ggsave}}
    #'@keywords hplot graphs
    #'@examples
    #'
    #'	require(ggplot2)
    #'  data(cars)
    #'	p <- ggplot(cars, aes(x = speed, y = dist)) + geom_point()
    #'	ggsave.latex(p, filename = "test.pdf", label = "fig:Cars", 
    #'    caption = "A demo plot", height = 5, width = 4)
    #'
    #'@export
    #'@importFrom ggplot2 ggsave
    ggsave.latex <- function(..., caption = NULL, label = NULL, figure.placement = "hbt", floating = TRUE, caption.placement="bottom", latex.environments="center"){
      ggsave(...)
      
      cat("\n\n")
      if(floating){
        cat("\\begin{figure}[", figure.placement, "]\n", sep = "")
      }
      cat("    \\begin{", latex.environments,"}\n", sep = "")
      if(!is.null(caption) && caption.placement == "top"){
        cat("        \\caption{", caption, "}\n", sep = "")
      }
      args <- list(...)
      if(is.null(args[["filename"]])){
        if(is.null(args[["plot"]])){
          names(args)[which(names(args) == "")[1]] <- "plot"
        }
        args[["filename"]] <- paste(args[["path"]], ggplot2:::digest.ggplot(args[["plot"]]), ".pdf", sep="")
      } else {
        args[["filename"]] <- paste(args[["path"]], args[["filename"]], sep="")
      }
      
      if(is.null(args[["width"]])){
        if(is.null(args[["height"]])){
          cat(
            "        \\includegraphics[height = 7in, width = 7in]{",
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        } else {
          cat(
            "        \\includegraphics[height = ", 
            args[["height"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            ", width = 7in]{", 
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        }
      } else {
        if(is.null(args[["height"]])){
          cat(
            "        \\includegraphics[height = 7in, width = 7 ", 
            args[["width"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            "]{", 
            args[["filename"]], 
            "}\n", 
            sep = "")
        } else {
          cat(
            "        \\includegraphics[height = ", 
            args[["height"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            ", width = ", 
            args[["width"]], 
            ifelse(
              is.null(args[["units"]]), 
              "in", 
              args[["units"]]
            ), 
            "]{", 
            args[["filename"]], 
            "}\n", 
            sep = ""
          )
        }
      }
      if(!is.null(caption) && caption.placement == "bottom"){
        cat("        \\caption{", caption, "}\n", sep = "")
      }
      if(!is.null(label)){
        cat("        \\label{", label, "}\n", sep = "")
      }
      cat("    \\end{", latex.environments,"}\n", sep = "")
      if(floating){
        cat("\\end{figure}\n")
      }
      cat("\n\n")
    }
    
    
  ##### Load Data -------------------------------------------------------------------
    
    # Final Analysis Set 
      load("./cwa/output/test.and.20pct.uninsp.Rdata")
    
    #for test + 20pct of uninsp
      quintileplot.df <- test.and.20pct.uninsp
      quintileplot.df <- testand20.new
    
    #rename the DV to fail.insp
      quintileplot.df$fail.insp <- quintileplot.df$DV
  
  ##### Summary statistics by propensity score quintile -----------------------------------------
    covariate.names.add <- c("FAC_STATE", "prop", "FAC_CHESAPEAKE_BAY_FLG", 
                             "FAC_INDIAN_CNTRY_FLG", "FAC_US_MEX_BORDER_FLG", 
                             "FAC_FEDERAL_FLG", "FAC_PERCENT_MINORITY", "FAC_POP_DEN", 
                             "AIR_FLAG", "SDWIS_FLAG", "RCRA_FLAG",
                             "TRI_FLAG", "GHG_FLAG", "FAC_IMP_WATER_FLG", 
                             "EJSCREEN_FLAG_US", "multiple_IDs","n_IDs", 
                             "SIC_AG_f", "SIC_MINE_f", "SIC_CONS_f", "SIC_MANU_f", "SIC_UTIL_f", 
                             "SIC_WHOL_f", "SIC_RETA_f", "SIC_FINA_f", 
                             "SIC_SERV_f", "SIC_PUBL_f", 
                             "num.facilities.cty", "num.facilities.st", "Party", 
                             "PERMIT_MAJOR", "PERMIT_MINOR", "prox.1yr",
                             "prox.2yr", "prox.5yr", "time.since.insp")
    
  
    # process by quintile, fix names 
      mydata <- quintileplot.df[ ,c(names(quintileplot.df) %in% covariate.names.add)]
  
    #remove factor variables
      remove.factors <- c("FAC_STATE", "Party") 
      mydata <- mydata[ ,c(!names(mydata) %in% remove.factors)]
      
    #convert logicals to numerics
      logicals <- sapply(mydata, is.logical)
      mydata[,logicals] <- lapply(mydata[,logicals], as.numeric)
      
    #convert the relevant chars to numerics
      chartonumeric <- c("FAC_INDIAN_CNTRY_FLG", "FAC_FEDERAL_FLG", 
                         "FAC_US_MEX_BORDER_FLG", "FAC_CHESAPEAKE_BAY_FLG",
                         "AIR_FLAG", "SDWIS_FLAG", "RCRA_FLAG",
                         "TRI_FLAG", "GHG_FLAG", "FAC_IMP_WATER_FLG",
                         "EJSCREEN_FLAG_US")
      
      mydata[,chartonumeric] <- 
        mydata %>%
        select(chartonumeric) %>% 
        mutate_all(function(x) as.numeric(mapvalues(x, from = c("N", "Y"), to = c(0, 1)))) 
      
      summary(mydata$prop)
      
    # develop quant intervals
      pquant <- quantcut(as.numeric(mydata$prop), q = seq(0, 1, by = 0.2))
      mydata$pquant <- quantcut(mydata$prop, q = seq(0, 1, by = 0.2))
      levels(pquant) <- c("1st", "2nd", "3rd", "4th", "5th")
      
      to.plot <- mydata %>%
        group_by(pquant) %>%
        summarise_all(mean); to.plot
      
      unique(quantcut(mydata$prop, q = seq(0, 1, by = 0.2))) 
    
  ##### Plot and save ---------------------------
    to.plot.m <-
      to.plot %>%
      melt(id.vars = 'pquant') 
    
    to.plot.m$group.quant = to.plot.m %>% group_indices(pquant)
    
    # new ordering
    levels(to.plot.m$variable) <- c("Days Since Inspection", "Native Amer. Boundary",
                                    "Federal Facility", 
                                    "US-Mexican Border","Chesapeake Bay", 
                                    "Percent Minority", "Population Density",
                                    "CAA Permit", "SDWIS Permit","RCRA Permit",  
                                    "TRI Permit", "GHG Permit", "Impaired Watershed", 
                                    "Env. Justice Index", "Multiple NPDES Permits", "Number of NPDES Permits", 
                                    "Ag, Forestry, Fishing", "Mining", "Construction",
                                    "Manufacturing", "Utilities and Transport", 
                                    "Wholesale Trade", "Retail Trade", "Finance, Insurance, Real Estate", 
                                    "Services", "Public Administration", 
                                    "Total Facilities in County", "Total Facilities in State", 
                                    "Major Permit", "Minor Permit", "Prox Inspections (1 yr)", 
                                    "Prox Inspections (2 yrs)", "Prox Inspections (5 yrs)",
                                    "Risk Score") 
    
    ggplot(to.plot.m, aes(x = group.quant, y = value)) + 
      geom_line(aes(group = variable)) +
      facet_wrap( ~ variable, nrow = 10, 
                  scales = "free_y") + 
      labs(x = '', y = '') +
      theme_bw() +
      theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
    
    ggsave("./cwa/output/figs/quintile_stats_final.pdf", width = 11, height = 8.5)
  

####################################################
#  Produce fig showing overlap between insp/uninsp prop scores 
####################################################
      
  ##### Set up 
    overlap.df <- test.and.20pct.uninsp #proportionate represntation from insp/uninsp groups
    
    overlap.df$insp_group <- ifelse(overlap.df$cat ==  "INS", 1, 0)
    
    #these numbers go into the caption 
      n.ins.overlap <- sum(overlap.df$cat == "INS")
      n.no.ins.overlap <- sum(overlap.df$cat != "INS")
    
  ##### Prep plot, split by quartile of risk score ----------
    quartile.cuts <- quantile(overlap.df$prop, seq(0, 1, 0.25)); quartile.cuts
    quartile.cuts <- round(quartile.cuts, digits = 4)
    
    overlap.df$quart <- cut(overlap.df$prop, quartile.cuts, 
                            labels = c(paste0("First Quartile \n [", quartile.cuts[1], " - ", quartile.cuts[2], ")"), 
                                       paste0("Second Quartile \n [", quartile.cuts[2], " - ", quartile.cuts[3], ")") , 
                                       paste0("Third Quartile \n [", quartile.cuts[3], " - ", quartile.cuts[4], ")"), 
                                       paste0("Fourth Quartile \n [", quartile.cuts[4], " - ", quartile.cuts[5], ")")),
                            include.lowest = T)
    
  ##### Plot and save -----------
    ggplot(overlap.df, 
           aes(x = prop, y = fct_rev(as_factor(quart)),
               fill = factor(insp_group, labels = c("Not inspected", "Inspected")))) + 
      geom_density_ridges(alpha = 0.45, scale = 1) + xlim(0,0.5) + #the xlim cuts off the highest-end values (hence warning)
      theme(legend.position = "bottom", 
            legend.box = "horizontal", 
            legend.title = element_blank(), 
            panel.background = element_blank()) +
      labs(x = "Risk score",
           y = "",
           title = "",
           subtitle = "")
    
    date <- Sys.Date()
    
    ggsave(paste0("./cwa/output/figs/insp_by_quartile_", date, ".pdf", sep = ""))


####################################################
# 15. Generate numbers for data flow graphic 
####################################################
      
  ##### Set Up -------
    rm(list = ls(all = TRUE)) #clean out workspace
    
    ## load packages
      library(reshape2)
      library(ggplot2)
      library(dplyr)
      
      library(data.table)
      library(lubridate) # for dates
      library(tidyverse)
    
    # Load the Relevant Datatset (fulloutput5)
      load("./cwa/data/modified_data/full.random.5sets.Rdata")

      # only keep test5.random and train5.random
      rm(test.random, test2.random, test3.random, test4.random)
      rm(train.random, train2.random, train3.random, train4.random)
      
      rm(propensity.scores) #large file, not needed
      
    # merge train5random and test5.random together
      all_facs <- full_join(train5.random, test5.random)
      test <- test5.random
      train <- train5.random
  
  ###### Calculate how many facilities we have in each category (for the data flow chart)
  
    # Overall numbers -----
      original <- length(test$REGISTRY_ID) + length(train$REGISTRY_ID);  original #316, 030
    
    # A. INSPECTED ---------------
    
      # DV's should be the same as X1 being NA's. 
        compare <- test %>% select(DV, X1) %>% mutate(xna = ifelse(is.na(X1), 0, 1))
        table(compare$DV, compare$xna)
      
      # Number of Overall Facilities ----
        num.facs <-  test %>%  distinct(REGISTRY_ID) %>% summarise(count = n()); num.facs #63206 ID's test set
        num.facs/original
      
      # Inspected (Overall) ----
        insp.all <- all_facs %>% filter(cat == "INS") %>%  summarise(count = n()); insp.all #80,126
        insp.all.pct <- insp.all/original; insp.all.pct #0.2535
      
    # # TRAINING SET -------------------------- 
      full.training.num <- train %>% filter(cat == "INS") %>%  summarise(count = n()); full.training.num # 64,325
      full.training.pct <- full.training.num/insp.all; full.training.pct #80.27
      
      insp.train <-  train %>% filter(cat == "INS") %>%  summarise(count = n()); insp.train # 64 325 inspectd 
      
      train.num.pass <- train %>% filter(cat == "INS") %>% summarise(pass = sum(is.na(X1))); train.num.pass  #60, 015
      train.pass.share <- train.num.pass/insp.train; train.pass.share # 0.932
      
      # # Fail
      train.num.fail <-  train  %>% filter(cat == "INS") %>% summarise(fail = sum(!is.na(X1))); train.num.fail # 4310
      share.fail <-  train.num.fail/insp.train; share.fail #0.0670
      
    
    # # TEST SET -------------------------- -
      insp.test <-  test %>% filter(cat == "INS") %>%  summarise(count = n()); insp.test # 15801 inspectd
      insp.test.pct <- insp.test/insp.all; insp.test.pct #0.197
      
    # B. UNINSPECTED ---------------
      all.num.uninsp <-
        all_facs %>% filter(cat != "INS") %>%  summarise(count = n()); all.num.uninsp  #235,904
      
      uninsp.all.pct <- all.num.uninsp/original; uninsp.all.pct #0.746
      
      twenty.pct <- all.num.uninsp*0.20; round(twenty.pct) #47181
      
      all.num.uninsp - round(twenty.pct) #188723
      
      #test only
      test.num.uninsp <-
        test %>% filter(cat != "INS") %>%  summarise(count = n()); test.num.uninsp  #47405
      
      uninsp.test.pct <- test.num.uninsp/num.facs; uninsp.test.pct #0.75000451
      

####################################################
# 16. Reallocate inspections over only inspected facilities and compare
####################################################

  #Note: this is the same code as in the reallocations above, but this reallocates over inspected facilities only
  
  ##### Set up
      
    ## load packages
      library(reshape2)
      library(ggplot2)
      library(data.table)
      library(xtable)
      library(readr)
      library(Hmisc) #for cuts2
      library(lubridate) # for dates
      library(tidyverse)
      library(dplyr)
      library(RColorBrewer)
      library(rlang)
      library(ROCR)
      library(gdata)
      library(corrplot)
      library(texreg)
      library(RCurl)
      library(grf)
      library(glmnet)
      library(rpart)
      library(pROC)
      library(sandwich)
      library(stargazer)
    
  
  # Write function to extract failure rate -----
    score_table <- function(df, indicator, fail.ind){ 
      require(dplyr)
      indicator <- enquo(indicator)      # Create quosure
      fail.ind  <- enquo(fail.ind)       # Create quosure     
      score_rates <- 
        df %>%
        group_by(score.cut) %>%
        summarise(
          facs = length(unique(REGISTRY_ID)),
          insp = sum(UQ(indicator)),                    # Use UQ to unquote
          fails = sum(UQ(indicator) * UQ(fail.ind)),    # Use UQ to unquote
          fail.rate = ifelse(insp == 0, 0, fails/insp)) # Avoid infinity
      return(score_rates)
    }
  
  
  ##### Load data ---------------------------------------------------------------
    
    inspections <- read_csv("../data/raw_data/npdes_downloads/NPDES_INSPECTIONS.csv")
    
    # Result by Score Cut -- this gives us our fail rates from the 500, and calculates UB and Lower Bounds
      load("./cwa/output/results.by.score.cut.Rdata")
    
    # load full data sets to figure out how many facilities overall there are
      load("./cwa/data/modified_data/full.random.5sets.Rdata") 
      rm(test.random, test2.random, test3.random, test4.random)
      rm(train.random, train2.random, train3.random, train4.random)
      
      all_facs <- full_join(train5.random, test5.random)
      N_allfacs <- length(all_facs$REGISTRY_ID)
      
    # Final Analysis Set (20% test + all uninspected faciltiies -- only extract inspected)
      load("./cwa/output/full.predictions.5a.5.final.Rdata") 
      
      uninsp <- full.predictions %>% filter(cat != "INS")
      test <- full.predictions %>% filter(cat == "INS")
      twenty.pct <- round(length(uninsp$REGISTRY_ID) * 0.20)
      twenty.pct.sample <- sample(x = 1:length(uninsp$REGISTRY_ID), size = twenty.pct, replace = FALSE)
      twenty.pct.uninsp <- uninsp %>% filter(row_number() %in% twenty.pct.sample) # 47180
      test.and.20pct.uninsp <- rbind(test, twenty.pct.uninsp)
      
    # Clean
    rm(full.predictions) 
    
    
  ##### Calculate inspection levels ----------------------------------------------
  
    #first join all.facs with inspections so each inspection has a state attached
    test.merge <- test[, c("REGISTRY_ID", "FAC_STATE")]
    inspections.w.state <- inner_join(inspections, test.merge)
    #note: this reduces the # of inspections to just the inspections of facilities in our 
    #test + 20% uninsp data, ensuring allocations don't assign to facilities that don't exist in our data
    
    #1. how many inspections of unique facilities per year, from 2012-2016?
      insp.summary <- inspections.w.state %>% mutate(year.insp = year(mdy(ACTUAL_END_DATE))) %>%
        filter(year.insp >= 2012 & year.insp <= 2016) %>%
        distinct(REGISTRY_ID, year.insp, .keep_all = TRUE) %>% 
        group_by(year.insp) %>% summarise(annual_insp = n())
      round(mean(insp.summary$annual_insp)) 
      
    # 2. Determine distribution of inspections across states
    
      insp.by.state <- inspections.w.state %>% mutate(year.insp = year(mdy(ACTUAL_END_DATE))) %>%
        filter(year.insp >= 2012 & year.insp <= 2016) %>%
        distinct(REGISTRY_ID, year.insp, .keep_all = TRUE) %>%
        group_by(year.insp, FAC_STATE) %>% summarise(st.insp = n())
      
      avg_yr_state <- insp.by.state %>% group_by(FAC_STATE) %>%
        summarise(stavg = round(mean(st.insp)))
      
    # 3. if needed, adjust the numbers to reflect the proportion of facilities over which we are allocating
    #here we scale the # of inspections based on the percentage of facilities in 
    # test.and.20pct.uninsp that are in the test set, b/c stavg is based on test.and.20pct.uninsp
    
      facility.proportion <- nrow(test)/nrow(test.and.20pct.uninsp)
      
      avg_yr_state$allocation.key <- round(avg_yr_state$stavg * facility.proportion)
      
      N_insp <- sum(avg_yr_state$allocation.key)
    
    # 3. Rename vars for consistency with later script & remove extra columns
      cwa.join <-  
        test %>%  #this marks that we are allocating over only the test set (includes only inspected facilities)
        select(REGISTRY_ID, cat, ACTUAL_END_DATE, X1, FAC_STATE, FAC_LAT, FAC_LONG,
               PERMIT_MAJOR, PERMIT_MINOR, DV, prop)
      
      cwa.join$score.cut <- as.character(cut2(cwa.join$prop, cuts = seq(0,1,0.05), digits = 2))
      cwa.join$insp_group <- ifelse(cwa.join$cat == "INS", 1, 0)
      cwa.join$fail.insp <-  with(cwa.join, ifelse(insp_group == 0 , 0 , DV)) 
    
    # 4. Generate Score Table (Results) and scale to number of inspections we have
    
      run.results <- score_table(cwa.join, insp_group, fail.insp); run.results 
      run.results$base <- round(run.results$insp / sum(run.results$insp) * N_insp); run.results #base allocation is by share
      
  
  ##### Conduct Facility Inspection "Assignment" -------
  
    observed.fail.rate.df <- data.frame(matrix(NA_real_, nrow = 500, ncol = 5))
      
    for (z in 1:500) { #because reallocations involve randomness, run this 500x and take means 
      # 1. Based on the score.cut allocations in run.results, "assign" inspections to facilities in test + uninspected (set key values)
        print(z)
        cwa.join <- cwa.join %>% arrange(desc(prop))
        unq.cuts <- unique(cwa.join$score.cut); unq.cuts
        min.prob <- 0.01
      
      
      #base allocation
        cwa.join$base.assigned <- 0
        smpl <- sample(nrow(cwa.join), N_insp, replace = F)
        cwa.join$base.assigned[smpl] <- 1
      
      #r1: based on risk score
        cwa.join <- cwa.join %>% mutate(r1.assigned = ifelse(row_number() <= N_insp, 1,0 )) 
        sum(cwa.join$r1.assigned) == N_insp 
      
      #r2: based on risk score by state
        cwa.join$r2.assigned <- 0
      
        for (i in 1:nrow(avg_yr_state)) {
          cwa.join[cwa.join$FAC_STATE == avg_yr_state$FAC_STATE[i],] <- 
            cwa.join %>% 
            filter(FAC_STATE == avg_yr_state$FAC_STATE[i]) %>% 
            mutate(r2.assigned = ifelse(row_number() <= avg_yr_state$allocation.key[i], 1, 0))
        }
        sum(cwa.join$r2.assigned) == N_insp 
      
      # r3: minimum probability, then risk-based
        r3.assign.df <- 
          cwa.join %>%  
          mutate(r3.assigned = 0) %>%  
          select(REGISTRY_ID, prop)
        
        min.prob.total.alloc <- round(min.prob * nrow(cwa.join))
        smpl.min.prob <- sample(nrow(cwa.join), min.prob.total.alloc, replace = F)
        r3.assign.df$minprob <- 0
        r3.assign.df$minprob[smpl.min.prob] <- 1
        
        insp.left.for.risk <- N_insp - min.prob.total.alloc
        
        to.be.assigned <- 
          r3.assign.df %>% 
          filter(minprob == 0) %>% 
          mutate(risk.assigned = ifelse(row_number() <= insp.left.for.risk, 1,0 ))
        
        r3.assign.df$risk <- 0
        r3.assign.df$risk[r3.assign.df$minprob == 0] <- to.be.assigned$risk.assigned
        r3.assign.df <- r3.assign.df %>%  mutate(r3.assigned = minprob + risk) 
        
        cwa.join$r3.assigned <- r3.assign.df$r3.assigned
        
        sum(cwa.join$r3.assigned) == N_insp 
      
      # r4: state-by-state: first majors, then random allocation, then risk-based
        cwa.join$r4.assigned <- 0
        r4.assign.df <- cwa.join[, c("REGISTRY_ID", "prop", "PERMIT_MAJOR", "FAC_STATE")]
        r4.assign.df$major <- 0
        r4.assign.df$minor <- 0
        r4.assign.df$risk <- 0
        
        for (i in 1:nrow(avg_yr_state)) {
          st <- avg_yr_state$FAC_STATE[i]
          st.insp <- avg_yr_state$allocation.key[i]
          st.facs <- r4.assign.df %>% filter(FAC_STATE == st)
          
          st.majors <- st.facs %>% filter(PERMIT_MAJOR == TRUE)
          major.insp.total <- min(st.insp, round(nrow(st.majors)/2))
          smpl.major <- sample(nrow(st.majors), major.insp.total, replace = F)
          assignment.major <- rep(0, nrow(st.majors))
          assignment.major[smpl.major] <- 1
          r4.assign.df$major[r4.assign.df$FAC_STATE == st & r4.assign.df$PERMIT_MAJOR == TRUE] <- assignment.major
          
          st.minors <- st.facs %>% filter(PERMIT_MAJOR == FALSE)
          minor.insp.total <- min(st.insp-major.insp.total, round(nrow(st.minors)*min.prob))
          smpl.minor <- sample(nrow(st.minors), minor.insp.total, replace = F)
          assignment.minor <- rep(0, nrow(st.minors))
          assignment.minor[smpl.minor] <- 1
          r4.assign.df$minor[r4.assign.df$FAC_STATE == st & r4.assign.df$PERMIT_MAJOR == FALSE] <- assignment.minor
          
          st.risk <- r4.assign.df %>% filter(FAC_STATE == st & major == 0 & minor == 0) 
          risk.insp.total <- max(st.insp - major.insp.total - minor.insp.total, 0)
          assignment.risk <- c(rep(1, risk.insp.total), rep(0, nrow(st.risk) - risk.insp.total))
          r4.assign.df$risk[r4.assign.df$FAC_STATE == st & r4.assign.df$major == 0 & r4.assign.df$minor == 0] <- assignment.risk
        }
        
        
        r4.assign.df$r4.assigned <- r4.assign.df$major + r4.assign.df$minor + r4.assign.df$risk
        cwa.join$r4.assigned <- r4.assign.df$r4.assigned
      
      #output number of failures based on observed number of failures
        observed.fail.rate.df[z, 1] <- sum(cwa.join$fail.insp[cwa.join$base.assigned == 1]) / N_insp
        observed.fail.rate.df[z, 2] <- sum(cwa.join$fail.insp[cwa.join$r1.assigned == 1]) / N_insp
        observed.fail.rate.df[z, 3] <- sum(cwa.join$fail.insp[cwa.join$r2.assigned == 1]) / N_insp
        observed.fail.rate.df[z, 4] <- sum(cwa.join$fail.insp[cwa.join$r3.assigned == 1]) / N_insp
        observed.fail.rate.df[z, 5] <- sum(cwa.join$fail.insp[cwa.join$r4.assigned == 1]) / N_insp
        
    }
    save(observed.fail.rate.df, file = "./cwa/output/observed.fail.rate.test.only_081218.Rdata")
    
    
  ##### Produce table comparing results -----------------------------------------
    
    load("./cwa/output/bar_bounds_share.Rdata")
    
    insp.and.uninsp <- paste0(round(bar_bounds_share$mean, 2), " (", 
                              round(bar_bounds_share$lower, 2), " - ", 
                              round(bar_bounds_share$upper, 2), ")")
    insp.only <- round(colMeans(observed.fail.rate.df[, 2:5]), 2)
                          
    compare.benefits <- cbind(insp.and.uninsp, insp.only)
    rownames(compare.benefits) <- c("Aggressive, national",
                                     "Aggressive, state",
                                     "Deterrence, national",
                                     "Majors, deterrence, state")
    
    colnames(compare.benefits) <- c("Inspected and uninspected",
                                  "Inspected only")
    
    benefits.comp.tbl <- xtable(compare.benefits, align = "lcc")
    
    label(benefits.comp.tbl) <- "tab:benefits.comparison"
    caption(benefits.comp.tbl) <- paste0("Results of reallocation over inspected facilities only. ",
                                         "Using a proportionate number of inspections on only the inspected facilities, ",
                                         "the improvements in the number of violations detected are comparable to the ",
                                         "improvements when reallocating over inspected and uninspected facilities. ",
                                         "Column 1 is identical to the results shown in Supplementary Table 5 ",
                                         "and are reported again here only for direct comparison to the sample of inspected only facilities (Column 2).")
    
    today <- "081218"
    
    print(benefits.comp.tbl,
          file = paste0("./cwa/output/benefits.comparison.", today, ".tex"),
          caption.placement = "top",
          digits = c(0, 0, 0 , 0),
          sanitize.text.function = function(x){x},
          include.colnames = T,
          include.rownames = T,
          booktabs = T,
          label = "tab:benefits.comparison")
