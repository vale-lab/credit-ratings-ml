###Predicting Credit Ratings with Machine Learning Algorithms
###Author: Valentina Guerra (valentina_grr)
###Date: 23/06/2021

###########################################################
#------------------------------------------Prep work for R#
###########################################################


install.packages("here")
library(here)
here()
here("construct","a","path")

#Setting the work directory to the folder of my choice
work_dir <- "C:/Users/valentina_guerra/OneDrive - S&P Global/Documents/RTraining/Professional Certificate Data Science/Capstone/CYO Project"
setwd(work_dir)
load("ML_credit_ratings_workspace.RData") #to be kicked out once done

#Downloading the dataset from my GitHub repository
url <- "https://raw.githubusercontent.com/vale-lab/credit-ratings-ml/main/corporate_rating.csv"
dest_file <- "corporate_rating.csv"
download.file(url, destfile = dest_file)

#Reading it into R as "data"
data <- read.csv("corporate_rating.csv")
as.data.frame(data)

#Download an image from my GitHub repository (for the Rmd report)
url2 <- "https://raw.githubusercontent.com/vale-lab/credit-ratings-ml/main/Rating-scale-for-primary-rating-agencies.jpg"
dest_file2 <- "Image.jpg"
download.file(url2, destfile = dest_file2, mode = 'wb')

#Checking if all packages necessary to run the code are installed and load them
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
#if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
#if(!require(vtable)) install.packages("vtable", repos = "http://cran.us.r-project.org")
if(!require(pastecs)) install.packages("pastecs", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")


library(plyr)
library(dplyr)
library(tidyverse)
library(caret)
library(data.table)
#library(lubridate)
library(ggthemes)
#library(vtable)
library(pastecs)
library(corrplot)
library(randomForest)

library(ggplot2)
library(here)

df <- read.delim(here("data", "raw_foofy_data.csv"))

###########################################################
#--------------------------------Exploratory Data Analysis#
###########################################################

#Check if there are missing values in the dataset
any(is.na(data))
#Double-check performed column by column
apply(data, 2, function(x) any(is.na(x)))

#From str(data) we learn that Date is a character variable. 
#We transform it into a date variable, using the base package.
str(data)
class(data$Date)
data <- data %>% mutate(Date = as.Date(Date, format = "%m/%d/%Y"))

##################
#General features#
##################

#Count the numbers of ratings, companies rated and number or rating agencies in the database
prelim_structure <- data %>% 
  summarise(n_ratings = n(), n_companies = n_distinct(Name), n_agencies = n_distinct(Rating.Agency.Name))
prelim_structure %>% knitr::kable()

#Studying the timeframe of ratings in the dataset
date_dist <- data %>% mutate(year = format(Date, "%Y")) %>%
  group_by(year) %>% summarise(n_ratings_by_date = n())
date_dist %>% knitr::kable()

#Counting the numbers of ratings per rating agency.
#Very concentrated market, with S&P beign the most active CRA (Credit Rating Agency) in the database
agency_distr <- data %>% group_by(Rating.Agency.Name) %>% 
  summarise(n_ratings_by_agency = n(), proportion_percentage = n_ratings_by_agency/2029*100) 
agency_distr %>% knitr::kable()

#Visualizing the numbers of ratings per rating agency
data %>% ggplot(aes(Rating.Agency.Name, fill = Rating.Agency.Name)) + 
  geom_bar() +
  labs(y = "Number of ratings provided", title = "Ratings distribution per rating agency") +
  guides(fill=guide_legend("Rating Agency")) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) 

#Count the number of ratings by company and the number of different agencies that rated them.
data %>% group_by(Name) %>% summarise(count_rating_agency = length(unique(Rating.Agency.Name)),
                                        count_ratings = n()) %>% arrange(desc(count_ratings))


#Count the number of ratings by sector
sector_distr <- data %>% group_by(Sector) %>% summarise(count_ratings = n(), 
                                                        proportion_percentage = count_ratings/2029*100)
sector_distr %>% knitr::kable()

#Visualizing the numbers of ratings per sector
data %>% ggplot(aes(Sector, fill = Sector)) + 
  geom_bar() +
  labs(y = "Number of ratings", title = "Ratings distribution per sector") +
  guides(fill=guide_legend("Sector")) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) 


#################################################
#Study the predicted variable: the credit rating#
#################################################

#Study the distribution of ratings, ordered from the highest to the lowest rating.
#Most of ratings are concentrated around the median level of risk (BBB/BB) 
cat_distr <- data %>% group_by(Rating) %>% 
  mutate(Rating = factor(Rating, levels = c( "AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"))) %>% 
  summarise(n_ratings = n(), proportion_percenatage = n_ratings/2029*100) 
cat_distr %>% knitr::kable()

#Visualizing the numbers of ratings per category
data %>% mutate(Rating = factor(Rating, levels = c( "AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"))) %>%
  ggplot(aes(Rating, fill = Rating)) +
  geom_bar() +
  labs(y = "Number of ratings", title = "Ratings distribution per rating category") +
  guides(fill=guide_legend("Rating Category (ordered)")) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) 

#Best rated companies in the dataset (up until 2016)
best_companies <- unique(data$Name[which(data$Rating == "AAA")])
best_companies

#Worst rated companies in the dataset (up until 2016)
worst_companies <- unique(data$Name[which(data$Rating == "CC" | data$Rating == "C" | data$Rating == "D")])
worst_companies

#Mapping investment and speculative grades using a binary variable: 
#IG for investment grades and Non-IG (high yield) for speculative grades
data_bin <- data %>% mutate(Category = case_when(
  Rating == "AAA" ~"IG",
  Rating == "AA" ~"IG",
  Rating == "A" ~"IG",
  Rating == "BBB" ~"IG",
  Rating == "BB" ~"Non-IG",
  Rating == "B" ~"Non-IG",
  Rating == "CCC" ~"Non-IG",
  Rating == "CC" ~"Non-IG",
  Rating == "C" ~"Non-IG",
  Rating == "D" ~"Non-IG",
))

#Counting the numbers of IG and Non_IG 
data_bin %>% group_by(Category) %>% summarise(n_ratings_binary_cat = n(), 
                                              proportion_percentage = n_ratings_binary_cat/2029*100) %>%
  knitr::kable()

#Visualizing the distribution of IG and Non-IG
data_bin %>% mutate(Category = as.factor(Category)) %>%
  ggplot(aes(Category, fill = Category)) +
  geom_bar() +
  labs(y = "Number of ratings", title = "Number of investment grades (1) and speculative grades (0) ratings") +
  guides(fill=guide_legend("Category")) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

#Studying the numbers of Non-IG and IG by Rating Agency
#Percentages refer to the total number of ratings in the dataset
df <- data_bin %>% select(Rating.Agency.Name, Category) %>% group_by(Rating.Agency.Name) %>%
  summarise(Count_IG = sum(Category == "IG"), Percentage_IG = Count_IG/2029*100, 
            Count_Non_IG = sum(Category == "Non-IG"), Percentage_Non_IG = Count_Non_IG/2029*100)
df %>% knitr::kable()

#Plotting the number of Non-IG and IG by Rating Agency. Require plyr
#Percentages refer to the total number of ratings given by the specified rating agency
b <- data_bin %>% select(Category, Rating.Agency.Name)
b

c = ddply(data.frame(table(b)), .(Rating.Agency.Name), mutate, pct = round(Freq/sum(Freq) * 100, 1))

ggplot(c, aes(x = Category, y = Freq, fill = Rating.Agency.Name)) + 
  geom_bar(stat = "identity", width = .7) +
  geom_text(aes(label = paste(c$pct, "%", sep = "")), vjust = -1, size = 3) +
  facet_wrap(~ Rating.Agency.Name, ncol = 2) + 
  scale_y_continuous(name = "Count", limits = c(0, 1.1*max(c$Freq))) +
  theme(legend.position = "none")

#Plotting the number of Non-IG and IG by Sector. Require plyr
#Percentages refer to the total number of ratings of the sector.

d <- data_bin %>% select(Category, Sector)

e = ddply(data.frame(table(d)), .(Sector), mutate, perct = round(Freq/sum(Freq) * 100, 1))
e

ggplot(e, aes(x = Category, y = Freq, fill = Sector)) + 
  geom_bar(stat = "identity", width = .7) +
  geom_text(aes(label = paste(e$perct, "%", sep = "")), vjust = -1, size = 3) +
  facet_wrap(~ Sector, ncol = 2) + 
  scale_y_continuous(name = "Count", limits = c(0, 1.6*max(e$Freq))) +
  theme(legend.position = "none")

############################################
#Study the predictors: the financial ratios#
############################################

#--------------------------------------Descriptive statistics for Liquidity Ratios

#Studying summary statistics for financial ratios, using pastecs package
#Liquidity ratios
liquidity_stats <- data %>% select(currentRatio, quickRatio, cashRatio, daysOfSalesOutstanding, payablesTurnover) %>% 
  stat.desc(basic = TRUE)
liquidity_stats <- round(liquidity_stats, 2)
liquidity_stats %>% knitr::kable()

#Plotting the boxplot to understand the distribution
#Liquidity ratios
data %>% select(currentRatio, quickRatio, cashRatio) %>%
  pivot_longer(., cols = c(currentRatio, quickRatio, cashRatio), names_to = "Var", values_to ="Val") %>%
  ggplot(aes(x = Var, y = Val)) +
  geom_boxplot(outlier.colour = "red")

#Defining and counting outliers
#Liquidity ratios
liquidity_stats2 <- data %>% select(currentRatio, quickRatio, cashRatio, daysOfSalesOutstanding, payablesTurnover) %>%
  pivot_longer(., cols = c(currentRatio, quickRatio, cashRatio, daysOfSalesOutstanding, payablesTurnover), 
               names_to = "Var", values_to ="Val") %>%
  group_by(Var) %>%
  summarise(mean = mean(Val), St.Dev = sd(Val), q1 = quantile(Val, probs = 0.25), 
            Median = quantile(Val, probs = 0.50), 
            q3 = quantile(Val, probs = 0.75), 
            Count_Outliers = sum(Val > (q3 + 1.5*(q3 - q1)) | Val < (q1 - 1.5*(q3-q1))))
liquidity_stats2 %>% knitr::kable()

#--------------------------------------Descriptive statistics for Profitability Ratios

#Studying summary statistics for financial ratios, using pastecs package
#Profitability ratios
profitability_stats <- data %>% select(netProfitMargin, pretaxProfitMargin, grossProfitMargin, operatingProfitMargin,
                                       ebitPerRevenue) %>% stat.desc(basic = TRUE)
profitability_stats <- round(profitability_stats, 2)
profitability_stats %>% knitr::kable()

#Plotting the boxplot to understand the distribution
#Profitability ratios
data %>% select(netProfitMargin, pretaxProfitMargin, grossProfitMargin, operatingProfitMargin,
                ebitPerRevenue) %>%
  pivot_longer(., cols = c(netProfitMargin, pretaxProfitMargin, grossProfitMargin, operatingProfitMargin,
                           ebitPerRevenue), names_to = "Var", values_to ="Val") %>%
  ggplot(aes(x = Var, y = Val)) +
  geom_boxplot(outlier.colour = "red")

#Defining and counting outliers
#Profitability ratios
profitability_stats2 <- data %>% select(netProfitMargin, pretaxProfitMargin, grossProfitMargin, operatingProfitMargin,
                                        ebitPerRevenue) %>%
  pivot_longer(., cols = c(netProfitMargin, pretaxProfitMargin, grossProfitMargin, operatingProfitMargin,
                           ebitPerRevenue), 
               names_to = "Var", values_to ="Val") %>%
  group_by(Var) %>%
  summarise(mean = mean(Val), St.Dev = sd(Val), q1 = quantile(Val, probs = 0.25), 
            Median = quantile(Val, probs = 0.50), 
            q3 = quantile(Val, probs = 0.75), 
            Count_Outliers = sum(Val > (q3 + 1.5*(q3 - q1)) | Val < (q1 - 1.5*(q3-q1))))
profitability_stats2 %>% knitr::kable()

#--------------------------------------Descriptive statistics for Operating Performance Ratios

#Studying summary statistics for financial ratios, using pastecs package
#Operating performance ratios
operating_stats <- data %>% select(returnOnAssets, returnOnCapitalEmployed, returnOnEquity,
                                   assetTurnover, fixedAssetTurnover) %>% stat.desc(basic = TRUE)
operating_stats <- round(operating_stats, 2)
operating_stats %>% knitr::kable()

#Plotting the boxplot to understand the distribution
#Operating performance ratios
data %>% select(returnOnAssets, returnOnCapitalEmployed, returnOnEquity,
                assetTurnover, fixedAssetTurnover) %>%
  pivot_longer(., cols = c(returnOnAssets, returnOnCapitalEmployed, returnOnEquity,
                           assetTurnover, fixedAssetTurnover), names_to = "Var", values_to ="Val") %>%
  ggplot(aes(x = Var, y = Val)) +
  geom_boxplot(outlier.colour = "red")

#Defining and counting outliers
#Operating performance ratios
operating_stats2 <- data %>% select(returnOnAssets, returnOnCapitalEmployed, returnOnEquity,
                                    assetTurnover, fixedAssetTurnover) %>%
  pivot_longer(., cols = c(returnOnAssets, returnOnCapitalEmployed, returnOnEquity,
                           assetTurnover, fixedAssetTurnover), 
               names_to = "Var", values_to ="Val") %>%
  group_by(Var) %>%
  summarise(mean = mean(Val), St.Dev = sd(Val), 
            q1 = quantile(Val, probs = 0.25), Median = quantile(Val, probs = 0.50), 
            q3 = quantile(Val, probs = 0.75), 
            Count_Outliers = sum(Val > (q3 + 1.5*(q3 - q1)) | Val < (q1 - 1.5*(q3-q1))))
operating_stats2 %>% knitr::kable()

#--------------------------------------Descriptive statistics for Leverage Ratios

#Studying summary statistics for financial ratios, using pastecs package
#Leverage ratios
leverage_stats <- data %>% select(debtEquityRatio, debtRatio, companyEquityMultiplier) %>% stat.desc(basic = TRUE)
leverage_stats <- round(leverage_stats, 2)
leverage_stats %>% knitr::kable()

#Plotting the boxplot to understand the distribution
#Leverage ratios
data %>% select(debtEquityRatio, debtRatio, companyEquityMultiplier) %>%
  pivot_longer(., cols = c(debtEquityRatio, debtRatio, companyEquityMultiplier), names_to = "Var", values_to ="Val") %>%
  ggplot(aes(x = Var, y = Val)) +
  geom_boxplot(outlier.colour = "red")

#Defining and counting outliers
#Leverage ratios
leverage_stats2 <- data %>% select(debtEquityRatio, debtRatio, companyEquityMultiplier) %>%
  pivot_longer(., cols = c(debtEquityRatio, debtRatio, companyEquityMultiplier), 
               names_to = "Var", values_to ="Val") %>%
  group_by(Var) %>%
  summarise(mean = mean(Val), St.Dev = sd(Val), 
            q1 = quantile(Val, probs = 0.25), Median = quantile(Val, probs = 0.50), 
            q3 = quantile(Val, probs = 0.75), 
            Count_Outliers = sum(Val > (q3 + 1.5*(q3 - q1)) | Val < (q1 - 1.5*(q3-q1))))
leverage_stats2 %>% knitr::kable()

#--------------------------------------Descriptive statistics for Cash Flow Indicator Ratios

#Studying summary statistics for financial ratios, using pastecs package
#Cash flow indicator ratios
cashflow_stats <- data %>% select(freeCashFlowOperatingCashFlowRatio, freeCashFlowPerShare, 
                                  operatingCashFlowPerShare, operatingCashFlowSalesRatio,
                                  cashPerShare) %>% stat.desc(basic = TRUE)
cashflow_stats <- round(cashflow_stats, 2)
cashflow_stats %>% knitr::kable()

#Plotting the boxplot to understand the distribution
#Cash flow indicator ratios
data %>% select(freeCashFlowOperatingCashFlowRatio, freeCashFlowPerShare, 
                operatingCashFlowPerShare, operatingCashFlowSalesRatio,
                cashPerShare) %>%
  pivot_longer(., cols = c(freeCashFlowOperatingCashFlowRatio, freeCashFlowPerShare, 
                           operatingCashFlowPerShare, operatingCashFlowSalesRatio,
                           cashPerShare), names_to = "Var", values_to ="Val") %>%
  ggplot(aes(x = Var, y = Val)) +
  geom_boxplot(outlier.colour = "red")

#Defining and counting outliers
#Cash flow indicator ratios
cashflow_stats2 <- data %>% select(freeCashFlowOperatingCashFlowRatio, freeCashFlowPerShare, 
                                   operatingCashFlowPerShare, operatingCashFlowSalesRatio,
                                   cashPerShare) %>%
  pivot_longer(., cols = c(freeCashFlowOperatingCashFlowRatio, freeCashFlowPerShare, 
                           operatingCashFlowPerShare, operatingCashFlowSalesRatio,
                           cashPerShare), 
               names_to = "Var", values_to ="Val") %>%
  group_by(Var) %>%
  summarise(mean = mean(Val), St.Dev = sd(Val), 
            q1 = quantile(Val, probs = 0.25), Median = quantile(Val, probs = 0.50), 
            q3 = quantile(Val, probs = 0.75), 
            Count_Outliers = sum(Val > (q3 + 1.5*(q3 - q1)) | Val < (q1 - 1.5*(q3-q1))))
cashflow_stats2 %>% knitr::kable()

#--------------------------------------Descriptive statistics for Valuation Ratios

#Studying summary statistics for financial ratios, using pastecs package
#Valuation ratio
valuation_stats <- data %>% select(enterpriseValueMultiple) %>% stat.desc(basic = TRUE)
valuation_stats <- round(valuation_stats, 2)
valuation_stats %>% knitr::kable()

#Plotting the boxplot to understand the distribution
#Valuation ratio
data %>% select(enterpriseValueMultiple) %>%
  pivot_longer(., cols = c(enterpriseValueMultiple), names_to = "Var", values_to ="Val") %>%
  ggplot(aes(x = Var, y = Val)) +
  geom_boxplot(outlier.colour = "red")

#Defining and counting outliers
#Valuation ratios
valuation_stats2 <- data %>% select(enterpriseValueMultiple) %>%
  pivot_longer(., cols = c(enterpriseValueMultiple), 
               names_to = "Var", values_to ="Val") %>%
  group_by(Var) %>%
  summarise(mean = mean(Val), St.Dev = sd(Val), 
            q1 = quantile(Val, probs = 0.25), Median = quantile(Val, probs = 0.50), 
            q3 = quantile(Val, probs = 0.75), 
            Count_Outliers = sum(Val > (q3 + 1.5*(q3 - q1)) | Val < (q1 - 1.5*(q3-q1))))
valuation_stats2 %>% knitr::kable()

###########################################################
#----------------------------------Building the algorithms#
###########################################################

#####################
#Cleaning Operations#
#####################

#From a visual inspection of str(data), I noticed that pretax profit margin and ebit per revenue seems identical. 
#If they are, we could remove one of the variable to make the dataset lighter.
#They are identical in 90.83% of cases: we can't remove.
sum(data$pretaxProfitMargin == data$ebitPerRevenue)/nrow(data)*100

#As operating profit and EBIT are often used as synonyms by finance professionals, we check whether they are identical.
#They are identical in only 13.31%: we keep both margins.
sum(data$operatingProfitMargin == data$ebitPerRevenue)/nrow(data)*100

#-------------------------------------------------1) Investment or speculative grade?

#Under problem n. 1, the machine learning challenge is to predict whether the company will receive an investment
#or a non-investment grade rating.

#This problem use the dataset "data_bin_ml" as reference, in which credit ratings are mapped to a binary variable:
#IG for investment grades and Non-IG for speculative grades.

#Select predictors (please refer to the report to understand the reason of the selection).
data_bin_ml <- data_bin %>% select(Category, Rating.Agency.Name, Sector, currentRatio, quickRatio, 
                                   cashRatio, daysOfSalesOutstanding, payablesTurnover, netProfitMargin, 
                                   pretaxProfitMargin, grossProfitMargin, operatingProfitMargin, ebitPerRevenue, 
                                   debtEquityRatio, debtRatio, freeCashFlowOperatingCashFlowRatio, 
                                   operatingCashFlowSalesRatio)

#Convert character variables into factors so they can be given as predictors to algorithms

data_bin_ml <- mutate_if(data_bin_ml, is.character, as.factor)

####################
#Train and test set#
####################

#Divide the dataset "data_bin_ml" into a test set and a train set. Randomly split with CreateDataPartition
#Test set will be 10% of data_bin_ml
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = data_bin_ml$Category, times = 1, p = 0.1, list = FALSE)
train_set <- data_bin_ml[-test_index,]
temp <- data_bin_ml[test_index,]

#Make sure that Sectors in test set are also in the train set
test_set <- temp %>% 
  semi_join(train_set, by = "Sector") 

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

#####################
#Logistic regression#
#####################

#What is the probability of being investment grade rating given who you are, the rating agency rating you,
#and the value of your ratios? 

logit <- glm(Category ~ ., data = train_set, family = "binomial"(link = "logit"), maxit = 100)
#maxit parameter is there to avoid warning message "algorithm did not converge"
summary(logit)

p_hat_logit <- predict(logit, test_set, type= "response") #response give predicted probabilities
#Decision rule: we predict IG if the probability is bigger than 0.5 (0.5 would be guessing without knowing)
rating_hat_logit <- factor(ifelse(p_hat_logit > 0.5, "IG", "Non-IG"))

#Accuracy: the overall proportion that is predicted correctly
confusionMatrix(rating_hat_logit, test_set$Category)$overall["Accuracy"]

#One assumption of logistic regression is no or little multicollinearity among the dependent variables.
#We check this assumption, by computing the pairwise correlation among the predictors (of class numeric only)
x <- data_bin_ml %>% select(currentRatio, quickRatio, cashRatio, 
                            daysOfSalesOutstanding, payablesTurnover, netProfitMargin, pretaxProfitMargin, 
                            grossProfitMargin, operatingProfitMargin, ebitPerRevenue, debtEquityRatio, debtRatio, 
                            freeCashFlowOperatingCashFlowRatio, operatingCashFlowSalesRatio)
x <- cor(x, use = "pairwise.complete")

#We make a plot of the correlation matrix and we can see that there are many variables with a high correlation 
#between each other: Net Profit Margin and Pretax Profit Margin, EBIT per Revenue and Operating Profit Margin, etc.
corrplot(x)


logit2 <- glm(Category ~ Sector + currentRatio + cashRatio + pretaxProfitMargin +
                ebitPerRevenue + debtRatio + operatingCashFlowSalesRatio,
              data = train_set, family = "binomial"(link = "logit"), maxit = 100)
#maxit parameter is there to avoid warning message "algorithm did not converge"
summary(logit2)

p_hat_logit2 <- predict(logit2, test_set, type= "response") #response gives predicted probabilities

#Decision rule: we predict IG if the probability is bigger than 0.5 (0.5 would be guessing without knowing)
rating_hat_logit2 <- factor(ifelse(p_hat_logit2 > 0.5, "IG", "Non-IG"))

#Accuracy: the overall proportion that is predicted correctly
confusionMatrix(rating_hat_logit2, test_set$Category)$overall["Accuracy"]

#################################
#Quadratic Discriminant Analysis#
#################################

#Logistic regression does not perform very well. It may be due to the fact that it assumes that the boundary 
#between the two category is linear, which may not be the case.

#We try quadratic discriminant analysis (qda), which assumes the boundary to be a quadratic function.

#QDA does not perform well with too many predictors: the model below won't work
#qda <- train(Category ~., data = train_set, method = "qda")

#We choose the predictors that were defined as statistical significance in logistic regression 
qda <- train(Category ~ Sector + currentRatio + cashRatio + pretaxProfitMargin +
                   ebitPerRevenue + debtRatio + operatingCashFlowSalesRatio, data = train_set, method = "qda")

rating_hat_qda <- predict(qda, test_set) #default option gives predicted outcome
confusionMatrix(rating_hat_qda, test_set$Category)$overall["Accuracy"]

#####################
#k-nearest neighbors#
#####################

#What is the probability of being investment grade rating given who you are, the rating agency rating you,
#and the value of your ratios? 

knn <- knn3(Category ~ ., data = train_set, k = 5) #5 is the default

rating_hat_knn <- predict(knn, test_set, type = "class") #class gives the predicted outcome

confusionMatrix(rating_hat_knn, test_set$Category)$overall["Accuracy"]

#What is the optimum value of value of k, the one with which we get maximum accuracy?
#We repeat the training on the model on train_set and the testing on test_set with a sequence of k going from 1
#to 45, where 45 represents the square root of 2029 (tot. n. of observations in dataset)

i = 1
k.optm = 1
for(i in 1:45){
  knn.opt <- knn3(Category ~ ., data = train_set, k = i)
  rating_hat_knn_opt <- predict(knn.opt, test_set, type = "class")
  k.optm[i] = confusionMatrix(rating_hat_knn_opt, test_set$Category)$overall["Accuracy"]
  k = i
  cat(k, '=', k.optm[i], '\n')
}

data_optk <- data_frame(k = (1:45), Accuracy = k.optm)
data_optk %>% ggplot(aes(k, Accuracy)) +
  geom_point()

#Remove test index, train set and test set, these variables names will be recycled in the next ML challenge
rm(test_index, train_set, test_set, temp, removed)

#-------------------------------------------------2) Predicting credit ratings

#Under problem n. 2, the machine learning challenge is to predict the credit rating, using a numerical variable
#mapped to the actual credit ratings. We use the numerical variable to replicate the rank order of credit ratings.

#This problem use the dataset "data_map_ml" as reference, in which credit ratings are mapped to a numerical variable
#called "Credit" going from 1 to 10, where 1 represents the best credit quality possible (AAA rating) 
#and 10 the default of a company (D rating).

#Mapping ratings to a numerical variable (useful to get the equivalent of the rank-order of ratings through a 
#numerical equivalent that the machine can understand)
data_map_ml <- data %>% mutate(Credit = case_when(
  Rating == "AAA" ~1,
  Rating == "AA" ~2,
  Rating == "A" ~3,
  Rating == "BBB" ~4,
  Rating == "BB" ~5,
  Rating == "B" ~6,
  Rating == "CCC" ~7,
  Rating == "CC" ~8,
  Rating == "C" ~9,
  Rating == "D" ~10,
))

#Select predictors (please refer to the report to understand the reason of the selection).
#Same predictors as in the previous challenge
data_map_ml <- data_map_ml %>% select(Credit, Rating.Agency.Name, Sector, currentRatio, quickRatio, cashRatio, 
                           daysOfSalesOutstanding, payablesTurnover, netProfitMargin, pretaxProfitMargin, 
                           grossProfitMargin, operatingProfitMargin, ebitPerRevenue, debtEquityRatio, debtRatio, 
                           freeCashFlowOperatingCashFlowRatio, operatingCashFlowSalesRatio)

#Drop AAA, CC, C, D ratings as the number of observation is too low and they won't be represented in the test set
data_map_ml %>% group_by(Credit) %>% summarise(n_ratings = n()) #15 observations to be removed
#Get the index of rows to be removed
drop <- with(data_map_ml, which(Credit == 1 | Credit == 8 | Credit ==9 | Credit == 10, arr.ind = TRUE))
data_map_ml <- data_map_ml[-drop, ]

#Convert character variables into factors so they can be given as predictors to algorithms
data_map_ml <- mutate_if(data_map_ml, is.character, as.factor)
data_map_ml$Credit <- as.factor(data_map_ml$Credit) 
#Even if numeric, Credit is categorical and should be converted in factors

####################
#Train and test set#
####################

#Divide the dataset "data_map_ml" into a test set and a train set. Randomly split with CreateDataPartition
#Test set will be 10% of data_bin_ml
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = data_map_ml$Credit, times = 1, p = 0.1, list = FALSE)
train_set <- data_map_ml[-test_index,]
temp <- data_map_ml[test_index,]

#Make sure that Credits in test set are also in the train set
test_set <- temp %>% 
  semi_join(train_set, by = "Credit") 

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

#####################
#k-nearest neighbors#
#####################

#What is the probability of receiving a determined rating given who you are, the rating agency rating you,
#and the value of your ratios? 

knn_rating <- train(Credit ~ ., 
                    method = "knn",
                    data = train_set,
                    tuneGrid = data.frame(k = seq(1, 45, 1))) #We test this sequence to determine the optimum k.

ggplot(knn_rating, highlight = TRUE) #1 is the optimum k
#knn_rating$bestTune
#knn_rating$finalModel

rating_hat_knn2 <- predict(knn_rating, test_set, type = "raw") #raw gives predicted outcomes
confusionMatrix(rating_hat_knn2, test_set$Credit)$overall["Accuracy"]

#Study sensitivity and specificity for each credit rating
cm_knn <- confusionMatrix(rating_hat_knn2, test_set$Credit)$byClass[, 1:2]
cm_knn

#####################
#Classification tree#
#####################

#cp (complexity parameter): the parameter that controls the size of the tree. To add another partition
#the RMSE must improve by a factor of cp for the new partition to be added.
#minsplit = minimum number of observations to be partitioned (default 20)
#minbucket = minimum number of observations in each partition

class_tree <- train(Credit ~ ., 
                    method = "rpart", 
                    data = train_set,
                    tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25))) #we test this sequence to determine the optimum cp

#The optimal value of the complexity parameter is 0.0125
ggplot(class_tree, highlight = TRUE)

rating_hat_class_tree <- predict(class_tree, test_set)
confusionMatrix(rating_hat_class_tree, test_set$Credit)$overall["Accuracy"]


#Study sensitivity and specificity for each credit rating
cm_class_tree <- confusionMatrix(rating_hat_class_tree, test_set$Credit)$byClass[, 1:2]
cm_class_tree

###############
#Random Forest#
###############

#Takes around 15 minutes to run
forest <- train(Credit ~ ., 
                 method = "rf", 
                 data = train_set,
                 tuneGrid = data.frame(mtry= seq(2, 16, 1))) #we test this sequence to determine the optimum mtry.

#The optimal value of randomly selected predictors is 3
ggplot(forest, highlight = TRUE)

rating_hat_forest <- predict(forest, test_set)
confusionMatrix(rating_hat_forest, test_set$Credit)$overall["Accuracy"]

#Study sensitivity and specificity for each credit rating
cm_forest <- confusionMatrix(rating_hat_forest, test_set$Credit)$byClass[, 1:2]
cm_forest

#Study variable importance for the fitted random forest "forest" and visualizing it
var_imp <- varImp(forest)
ggplot(var_imp)

###########################################################
#------------------------Save RData to knit the Rmd Report#
###########################################################

#It is important to save the RData file in the work directory as this will be the basis to knit the Rmd report
save.image(file = "ML_credit_ratings_workspace.RData")












#-----------------------------------------------------------------------------------------Old Stuff
#Trying to replace the usage of dplyr here using data.table
  
DT <- data.table(b)
DT[, count(Category)/sum(Category) * 100, by=.(Category, Rating.Agency.Name)]
DT[, Freq/sum(Freq) * 100, by=.(Category, Rating.Agency.Name)]


bla <- b %>% group_by(Rating.Agency.Name, Category) %>% 
  tally() 

%>% mutate(pct = n/2029*100)
data.frame(bla)

ggplot(data = bla, aes(x = Category, y = n)) + 
  geom_bar(stat = "identity", width = .7) +
  geom_text(aes(label = paste(bla$pct, "%", sep = "")), vjust = -1, size = 3) +
  facet_wrap(~ Rating.Agency.Name, ncol = 2) + theme_bw()


+
  scale_y_continuous(limits = c(0, 1.2*max(c$Freq)))


#Studying summary statistics for financial ratios, use qwraps
#Liquidity ratios

liquidity_summary <- list("Current Ratio" =
                            list("min" = ~ min(data$currentRatio),
                                 "max" = ~ max(data$currentRatio),
                                 "mean (sd)" = ~ qwraps2::mean_sd(data$currentRatio)
                            )
)

bla <- summary_table(data, liquidity_summary)

#Studying summary statistics for financial ratios, use dplyr
#Liquidity ratios
data %>% select(currentRatio, quickRatio, cashRatio, daysOfSalesOutstanding, payablesTurnover) %>% 
  summarise(across(.fns = list(min = min, max = max, mean = mean, median = median, sd = sd)))
  
  summarise(across(.cols = c("currentRatio", "quickRatio", "cashRatio"), 
                          .fns = list(min = min, max = max, mean = mean, median = median, sd = sd)))

  
#Studying summary statistics for financial ratios, using vtable package (include kableExtra, wich conflicts with knitr)
#Liquidity ratios
  
data %>% 
  st(summ = c('mean(x)','sd(x)','min(x)','pctile(x)[25]', 'pctile(x)[50]', 'pctile(x)[75]','max(x)'), 
     title = "Summary Statistics", 
     summ.names = c('Mean', 'Std. Dev', 'Min', 'Pctl. 25', 'Median', 'Pctl. 75', 'Max'), 
     digits = 2)
liquidity_stats %>% knitr::kable()


#Liquidity ratios
liquidity_stats <- data %>% select(currentRatio, quickRatio, cashRatio, daysOfSalesOutstanding, payablesTurnover) %>% 
  stat.desc()
liquidity_stats <- round(liquidity_stats, 2)
liquidity_stats %>% knitr::kable()



install.packages("corrplot")
library(corrplot)
corrplot(as.matrix(x), is.corr=FALSE, tl.cex = 0.5)

install.packages("pls")
library(pls)

pcr <- pcr(Category ~ ., data = train_set, scale = T, validation = "CV") #pcr requires only numeric

i = 1
k.optm = 1
for(i in 1:45){
  knn.opt <- knn3(Category ~ ., data = train_set, k = i)
  rating_hat_knn_opt <- predict(knn.opt, test_set, type = "class")
  k.optm[i] = confusionMatrix(rating_hat_knn_opt, test_set$Category)$overall["Accuracy"]
  k = i
  cat(k, '=', k.optm[i], '\n')
}

optimum_k <- plot(k.optm, type="p", xlab="k",ylab="Accuracy")
optimum_k <- recordPlot(optimum_k)


linear <- lm(Credit ~ ., data = train_set)
summary(linear)

rating_hat_linear <- round(predict(linear, test_set, type= "response"), 0) #response gives outcomes
rating_hat_linear[rating_hat_linear > 10] <- 10

#Accuracy: the overall proportion that is predicted correctly
confusionMatrix(rating_hat_linear,test_set$Credit)$overall["Accuracy"] #won't work because CF doesn't work on continuous data

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

linear_rmse <- RMSE(test_set$Credit, rating_hat_linear)
linear_rmse