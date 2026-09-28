# Auburn-Regions starter
train <- read.csv('../data/application_train.csv')
eval <- read.csv('../data/application_evaluation.csv')
fit <- glm(DefaultFlag ~ ApplicantAge + AnnualIncome + RequestedAmount + VendorScoreBeacon + DebtToIncomeRatio, data=train, family=binomial())
p <- predict(fit, newdata=eval, type='response')
write.csv(data.frame(CustomerID=eval$CustomerID, PredictedDefaultProbability=p), '../submission/R_starter_predictions.csv', row.names=FALSE)
