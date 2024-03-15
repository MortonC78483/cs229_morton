# Load NPDES inspection data from: https://purl.stanford.edu/hr919hp5420
library(readr)
library(tidyverse)
npdes <- read_csv("/Users/clairemorton/Documents/_CS229/final_proj/NPDES_INSPECTIONS.csv")

head(npdes)
wi <- npdes %>%
  filter(startsWith(NPDES_ID, "WI"))

ca <- npdes %>%
  filter(startsWith(NPDES_ID, "CA"))
data <- load("/Users/clairemorton/Documents/_CS229/final_proj/fac.final.hist.Rdata")
data <- fac.final.hist
data$DV <- as.numeric(!is.na(data$X1)) #1 for a failed inspection, 0 otherwise
data_wi <- data %>%
  filter(as.character(FAC_STATE) == "WI")

# list of CAFO-related violations
cafo_viol <- c("B0A19", "B0038", "D0A11", "B0A12", "B0032", "B0033", "B0A41", 
  "B0043", "C0A11", "D0A12", "C0019", "B0A40", "B0A23", "B0039", 
  "B0037", "B0036", "E0A13", "B0034", "B0035", "A0A22", "E0A16",
  "C0020", "E0A14", "A0A12", "A0019", "B0A42")

data_cafo <- data %>%
  filter(X1 %in% cafo_viol |
           X2 %in% cafo_viol |
           X3 %in% cafo_viol|
           X4 %in% cafo_viol|
           X5 %in% cafo_viol)



