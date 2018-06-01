# time series forecasting

# getting wikipedia trend data
library(pageviews)
data <- article_pageviews(project = "en.wikipedia",
                  article = "Tom_Brady", platform = "all",
                  user_type = "all", start = "2018010100", end = "2018053100", 
                  reformat = TRUE)

# reorder the columns
data <- data[,c(7,8,1:6)]
# renaming a column
colnames(data)[colnames(data)=="views"] <- "count"

# plot
library(ggplot2)
qplot(date,count, data = data)
summary(data)

# if we convert count to log() it'll be better to investigate
ds <- data$date
y <- log(data$count)
df <- data.frame(ds,y)
qplot(ds,y, data = df)

# forecast package
library(prophet)
# model
model <- prophet(df)

# prediction
futureDf <- make_future_dataframe(model,periods = 365)
tail(futureDf)
pred <- predict(model,futureDf)
tail(pred[c("ds","yhat","yhat_lower","yhat_upper")])
# yhat values are in logs take exp of it to get forecasted views or count

# plot forecast
plot(model,pred)
prophet_plot_components(model,pred)
