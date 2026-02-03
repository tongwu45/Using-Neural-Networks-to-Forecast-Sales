# Using-Neural-Networks-to-Forecast-Sales

# Before We Start: Preparations and Set Up
Before we start, let us do some preparation work to get ready!

We need to:

  1. Set working directory to a folder on your device,
  
  2. Download dataset(s) needed for this tutorial onto your working directory, and
  
  3. Load data and other files for analysis later as we proceed

The dataset of this tutorial is made available to you. The file is named Neural_network and is stored in an xlsx format.

To save the dataset and for the convenience of analysis, let us download all the relevant files in this tutorial to the default Download folder of your computer. Of course, feel free to create your own folder and use it as your working directory.

To load the dataset to R, we need to run only part of the following chunk of codes, depending on the operating system of your device. Please add a “#” before each line of non-applicable codes in the chunk below to avoid executing unnecessary or redundant codes and receiving error messages. For example, if you are using a Mac, you need to put “#” before code setwd(“C:/Users/wangwanxin/Dropbox/PC/Downloads”).

NOTE The specific working directory should be modified based on the location where you create your folder and save your data. In the chunk below, the working directory address is mine, not yours. You need to make a change for it to work, especially if you are using a PC. For Mac users, your working directory should be “~/Downloads/forecasting” as long as you created your folder and saved your data in “Downloads”, i.e., you don’t need to change anything.

```r
# Please run the following codes depending on the operating system on your laptop/PC. 

# if you are using iOS (e.g., a Mac),  you will need to set your working directory as the Downloads folder as follows:

#setwd("~/Downloads/forecasting")

# if you are using Windows (e.g., a PC), you will need to first set your working directory to Downloads folder on one of your disks:

setwd("C:/Users/wangwanxin/Dropbox/PC/Downloads")

# load the dataset, and name the dataset "NeuralPriceAdsData" in our R environment.

NeuralPriceAdsData <-read_xlsx("Neural_network.xlsx")
```



WE ARE READY TO GO!




# Introduction


Forecasting seems not a cutting-edge technique to most of us since you have already learned to perform basic forecasting using some standard tools like the moving average analysis, ARMA/ARIMA time series analysis, and multiple linear regression. However, in a variety of business contexts, forecasting is still one of the (if not the only) most essential tasks that an analyst should Ace at before any further challenging analytics can be performed.



# 1. Multiple Linear Regression and Neural Networks for Forecasting


## 1.1 Multiple Linear Regression as a Vintage Technique
A common need in business analytics is forecasting the future sales of a product. In forecasting, you try and predict a dependent variable (usually called y) from one or more independent variables (usually referred to as 𝑥1, 𝑥2, …, 𝑥𝑛).

To gain better and more accurate insights about the often complex relationships between a variable of interest and its predictors, as well as to better forecast, one needs to move from simple time series analytics towards multiple linear regression in which more than one independent variable is used to forecast y. Utilizing multiple regression may lead to improved forecasting accuracy along with a better understanding of the variables that actually cause y.



Some of you might argue that such a model makes too strict an assumption that the relationship between all the predictors and predicted value is restricted to be linear, or additive, which is not necessarily the case. For example, could advertising spending have a non-linear (e.g., multiplicative) impact on sales? Definitely yes.

Having said so, it is actually still hard for us to speculate on the exact functional form of the forecasting equation since, after all, there can be so many types of possibilities. However, thanks to several exciting advancements in analytical technologies, we can rely on artificial intelligence to figure this myth out for us.



## 1.2 Neural Networks as an Advanced Tool
Neural nets are a fantastic form of artificial intelligence that can capture complex relationships between dependent and independent variables. Essentially a neural network is a “black box” that searches many models (including nonlinear models involving interactions) to find a relationship involving the independent variables that best predict the dependent variable. In a neural network, the independent variables are called input cells, and the dependent variable is called an output cell (more than one output is OK).

As in regression, neural nets have a certain number of observations (say, N). Each observation contains a value for each independent variable and dependent variable. Also, similar to a regression, the goal of the neural network is to make accurate predictions for the output cell or dependent variable.

As you will see, the usage of neural networks is increasing rapidly because neural networks are great at finding patterns. In regression, you only find a pattern if you know what to look for.

For example, if 𝑦=𝑙𝑛(𝑥) and you simply use x as an independent variable, you cannot predict y very well. A neural network does not need to be “told” the nature of the relationship between the independent variables and the dependent variable. If a relationship or pattern exists and you provide the neural network enough data, it can find the pattern on its own by “learning” it from the data.

A major advantage of neural networks over regression is that this method requires no statistical assumptions about your data. For example, unlike regression, you do not assume that your errors are independent and normally distributed.

Question: After knowing the basic concepts, now do you think neural networks will be able to outperform or will underperform multiple linear model in forecasting? Why do you think so?



# 2.Using Neural Networks to Predict Sales


To demonstrate how neural networks can find patterns in data, let us use the data in the data file. We are going to fit a multiple linear model and use neural network to do forecasting and then compare the model performance.

Here in the data, you are given the weekly price of the product, advertising spending (in hundreds of dollars), and weekly sales of the product (in thousands of units). Note that here we are not talking about revenues, which is calculated as Price X Quantity sold; here sales refer to the Quantity, or Volume of units sold.

What’s the plan? We start by first running a multiple linear regression to predict Sales from Price and Advertising. Then we apply neural network to repeat the same task and compare model performances under the two methods.

Before we do so, let us take a preview of the first few rows of our dataset by running the following one-liner:

```r
head(NeuralPriceAdsData)
```
```text
## # A tibble: 6 x 3
##   Sales Price Advertising
##   <dbl> <dbl>       <dbl>
## 1  400.     7          53
## 2  365      9          68
## 3  388.     8          76
## 4  432      5          70
## 5  401.     7          64
## 6  388.     8          78
```


Feel free to do some descriptive analysis or visualizations of the data to get a better idea of the situation.

## 2.1 Multiple Linear Regression


At this point we use 80 percent of the data set to estimate the linear model. There are 334 data points (i.e., weeks) in the NeuralPriceAds data. Hence we can use 267 data points (334 * 80%) for model estimation, and 67 data points (334 * 20%) for forecasting and testing model accuracy.

## Dividing Data into Training and Testing Sets


There are many ways to divide the data into training and testing set. Here the observations used for the testing and training sets are randomly chosen. That is, each observation has an 80 percent chance of being in the training set and a 20 percent chance of being in the testing set. We are going to use the test dataset to obtain the error later.

```r
set.seed(123)

# Dividing training and testing set following the 80/20 rule.

# This is to get the row index of 267 random rows of data 
train_ind_mlr <- sample(seq_len(nrow(NeuralPriceAdsData)), size = floor(nrow(NeuralPriceAdsData) * 0.8))

# Here, using the row index that we just identified above, we divide the data into two: training and testing set. "mlr" stands for Multiple Linear Regression.

train_mlr <- NeuralPriceAdsData[train_ind_mlr,]
test_mlr <- NeuralPriceAdsData[-train_ind_mlr,]
```

## Model Estimation


Now we can estimate the multiple linear regression using the training set. Here we use the “lm” function, which refers to linear regression–pretty self-explanatory, right?

```r
# Estimating MLR model, and name the model "lm_estimation:

lm_estimation <- lm(Sales ~ Price + Advertising, data = train_mlr)

# Summarizing model results using the "summary" function:

summary(lm_estimation)
```
```text
## 
## Call:
## lm(formula = Sales ~ Price + Advertising, data = train_mlr)
## 
## Residuals:
##     Min      1Q  Median      3Q     Max 
## -3.9585 -1.6549 -0.1645  1.4275  4.0738 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept) 513.195739   0.913976  561.50  < 2e-16 ***
## Price       -16.732609   0.074881 -223.46  < 2e-16 ***
## Advertising   0.064859   0.008557    7.58 5.88e-13 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 2.027 on 264 degrees of freedom
## Multiple R-squared:  0.9949, Adjusted R-squared:  0.9949 
## F-statistic: 2.57e+04 on 2 and 264 DF,  p-value: < 2.2e-16
```
Reading from the results, the regression has a high 𝑅2 and a residual standard error of 2.03 units.

What is residual standard error? The residual standard error (or residual standard deviation) is a measure used to assess how well a linear regression model fits the data. (The other measure to assess this goodness of fit is the more well-known R-squared). Simply put, the residual standard deviation is the average amount that the real values of Y (here sales) differ from the predictions provided by the regression line.

## Results Interpretations


It is not enough to merely commenting on the goodness of fit (e.g., residual standard error or R-squared) when looking at regression results. We need to interpret the model coefficients, which can be quite informative for generating business insights. For example, in this simple model we get that:

  1. Weekly sales is approximately 513000 on average, in the absence of price and advertising spending. [Question: is this informative in the real practice?]
  
  2. A 1 unit increase in price is associated with 16730 units decrease in weekly sales.
  
  3. A 1 unit (here, 100 dollars) increase in advertising spending is associated with 65 units increase in weekly sales.

## An Alternative Log-log model


Many of the times, to further guide practice, one can estimate a log-log model instead, so that the coefficients are elasticities instead of in absolute terms. You only need to log-transform both the dependent and the independent variables and estimate a linear model using the lm function again.

```r
# Estimating MLR model, and name the model "lm_estimation:

lm_estimation2 <- lm(log(Sales) ~ log(Price) + log(Advertising+1), data = train_mlr)

# Summarizing model results using the "summary" function:

summary(lm_estimation2)
```
```text
## 
## Call:
## lm(formula = log(Sales) ~ log(Price) + log(Advertising + 1), 
##     data = train_mlr)
## 
## Residuals:
##       Min        1Q    Median        3Q       Max 
## -0.018954 -0.014245  0.001153  0.013727  0.020792 
## 
## Coefficients:
##                       Estimate Std. Error t value Pr(>|t|)    
## (Intercept)           6.544176   0.020239 323.339   <2e-16 ***
## log(Price)           -0.306300   0.003559 -86.058   <2e-16 ***
## log(Advertising + 1)  0.008144   0.004184   1.946   0.0527 .  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.01334 on 264 degrees of freedom
## Multiple R-squared:  0.9664, Adjusted R-squared:  0.9661 
## F-statistic:  3795 on 2 and 264 DF,  p-value: < 2.2e-16
```


QUESTION: Why did we add 1 to the variable “Advertising” here?



Now we should interpret the model results as follows:

  1. A 1 percent increase in price is associated with 0.31 percent decrease in weekly sales.
  
  2. A 1 percent increase in advertising spending is associated with 0.008% increase in weekly sales.
  


QUESTION: Is our sales elastic or inelastic (i.e., how sensitive is our sales to changes in price)?



Now let us switch back to the first model, lm_estimation1 and proceed with our analysis.

## Model Predictions
After estimating an MLR model using the training set, we forecast sales in both the training and the testing set using predict function.

```r
# Generate predictions for both training and testing set, naming them lm_predict_train and lm_predict_test, respectively. Here the difference is that for the testing set, the dataset used is a new dataset, i.e., the testing set, which is different from the training set that was used for model estimation. 

lm_predict_train <- predict(lm_estimation , data = train_mlr)
lm_predict_test <- predict(lm_estimation , newdata = test_mlr)

# We name the predicted sales using MLR method "Predict_MLR", and add that column into our dataset:

NeuralPriceAdsData[train_ind_mlr,'Predict_MLR'] <- lm_predict_train
NeuralPriceAdsData[-train_ind_mlr,'Predict_MLR'] <- lm_predict_test

#Now take a look at the first few rows of the updated data. Now the data should have 1 more column named Predict_MLR that we just constructed. Since we are mostly interested in predicting the testing set, we only check the data in the testing set: 

head(NeuralPriceAdsData[-train_ind_mlr,])
```
```text
## # A tibble: 6 x 4
##   Sales Price Advertising Predict_MLR
##   <dbl> <dbl>       <dbl>       <dbl>
## 1  388.     8          76        384.
## 2  415.     6          52        416.
## 3  434.     5          91        435.
## 4  365      9          92        369.
## 5  404.     7          85        402.
## 6  350     10          71        350.
```
The first and last column in the table above lists original and predicted sales using MLR method, respectively. For example, in the first row, the actual sales was 387600 units, while MLR gives a prediction of 384264 units–there is obviously a gap but it’s not bad.

Now we proceed to evaluate prediction accuracy by calculating the Root Mean Squared Error, or RMSE. If you are not familiar with the formula of RMSE, please take a look yourself, or refresh your memory using the codes below:

```r
# Use the row index to get the original value of sales in the testing set. 
test_mlr.r <- NeuralPriceAdsData$Sales[-train_ind_mlr]

# Use the row index to get the predicted value of sales (using MLR method) in the testing dataset. 
test_mlr_predict.r <- NeuralPriceAdsData$Predict_MLR[-train_ind_mlr]

# Calculate the Root Mean Squared Error of testing dataset. Note that we divide the sum of squared difference between the actual and the predicted values by 67 because we have 67 observations (rows) in the testing set.
rmse.test_mlr <- (sum((test_mlr.r - test_mlr_predict.r )^2)/67)^0.5

# Display RMSE.
rmse.test_mlr
```
```text
## [1] 2.039959
```
We get that the RMSE of sales forecasting using MLR method is 2.04.

## Visualizing Prediction Performance
To further visualize our forecasting performance, we can plot both the actual and the predicted sales using MLR on the same graph. You can use ggplot2 package to plot graphs, yet here we are using another set of commands, just to present you more options when it comes to plotting.


```r
week <- c(1:67)

## MLR method: extract actual and predicted sales

actual_sales <- NeuralPriceAdsData[-train_ind_mlr,'Sales']
predicted_sales_mlr <-NeuralPriceAdsData[-train_ind_mlr,'Predict_MLR']

# Construct a data frame for this plotting task
plot_data_mlr <-data.frame(week, actual_sales, predicted_sales_mlr)

# Plot dots and lines: actual sales are in red while predicted sales are in blue.

plot(plot_data_mlr$week, plot_data_mlr$Sales, col="red")
lines(plot_data_mlr$week, plot_data_mlr$Sales, col="red")
points(plot_data_mlr$week, plot_data_mlr$Predict_MLR, col="blue")
lines(plot_data_mlr$week, plot_data_mlr$Predict_MLR,col="blue")
```
<img width="726" height="445" alt="截屏2026-02-03 上午11 40 10" src="https://github.com/user-attachments/assets/fae68e59-4bea-4cc7-b118-2e8476c3f6bc" />



Question: The plot above is definitely quite rough. What’s missing? Can you improve the layout of the plot? (Hint: take a look at the ggplot2 package, or read the materials on data visualization that is provided by me.)



## 2.2 Neural Networks


Now it’s time for us to apply neural networkS to repeat the analysis above.



## 2.2.1 Data Normalization


As a first step, we are going to address data pre-processing.

It is good practice to normalize your data before training a neural network. This step is critical as, depending on your dataset, avoiding normalization may lead to useless results or to a very difficult training process (most of the time, the algorithm will not converge before the number of maximum iterations allowed).

You can choose different methods to scale the data (z-normalization, min-max scale, etc.). Here we chose to use the min-max method and scale the data in the interval [0,1]. Usually, scaling in the intervals [0,1] or [-1,1] tends to give better results.

<img width="289" height="74" alt="截屏2026-02-03 上午11 40 50" src="https://github.com/user-attachments/assets/92774916-c11c-448c-a25c-466f48750e50" />



```r
# Normalise data
maxs <- apply(NeuralPriceAdsData, 2, max) 
mins <- apply(NeuralPriceAdsData, 2, min)
# Now the scaled version of the data are saved as "scaled".
scaled <- as.data.frame(scale(NeuralPriceAdsData, center = mins, scale = maxs - mins))
```

## 2.2.2 Split the data into training and testing set


At this point we still use 80 percent of the data set to train the network. Same as in the MLR section, the observations used for the testing and training sets are randomly chosen. That is, each observation has an 80 percent chance of being in the training set and a 20 percent chance of being in the testing set.

Note that for clarity of the flow, we use *“_mlr”* and *“_nn”* to distinguish two methods examined in this section. If you are more familiar and proficient with R, you can modify the codes yourself to avoid redundancy.

```r
# Set the seed to make your partition reproducible
set.seed(123)

# Divide the data into training and testing set following the same 80/20 rule:

train_ind_nn <- sample(seq_len(nrow(NeuralPriceAdsData)), size = floor(nrow(NeuralPriceAdsData) * 0.8))

train_nn <- scaled[train_ind_nn,]
test_nn <- scaled[-train_ind_nn,]
```


## 2.2.3 Train the model


The package we use is the the neuralnet library, and the function is named neuralnet.

```r
nn <- neuralnet(formula = Sales ~ Price + Advertising, 
               data = train_nn,
               hidden = 2, 
               linear.output=TRUE, 
               err.fct = 'sse')
```

It requires the following input into the function:

  1. formula: Sales ~ Price + Advertising, Sales is the output and Price and Advertising are features. This is the same formula used in linear regression.
  
  2. data: the data frame containing the variables specified in the formula.
  
  3. hidden: a vector of integers specifying the number of hidden neurons (vertices) in each layer. One could specify the number of hidden layers and the number of neurons in each layer. It is beyond the scope of this book to discuss how neural networks create predictions. It is sufficient at this point to know that (i) one hidden layer gets decent performance for most problems, including this case of sales prediction; (ii) the number of neurons in that layer is the mean of the number of features in the independent variables and dependent variables, which is 1 or 2 in this case. We could test it out to compare the performance of 1 neuron vs 2 neurons and decide.
  
  4. linear.output: we are solving a linear regression problem so it’s True.

  5. err.fct is the error function. sse stands for Sum of Squared Error.



Essentially, we are building a NN like this:

<img width="306" height="275" alt="截屏2026-02-03 上午11 42 53" src="https://github.com/user-attachments/assets/82a34c7c-fbcf-4ba3-ab3e-1dd5ed10dc51" />


Figure 1: NN visualisation

After model fitting, we can visualize our neural network using the plot function:

```r
plot(nn)
```
<img width="435" height="424" alt="截屏2026-02-03 上午11 43 14" src="https://github.com/user-attachments/assets/0831a095-460b-4628-8ba0-c49969daebae" />

Figure 2: NN Results

The black lines show the connections between each layer and the weights on each connection while the blue lines show the bias term added in each step. The bias can be thought as the intercept of a linear model.

The net is essentially a black box so we cannot say that much about the fitting, the weights and the model. For example, we see that Price has a negative weight of -1.104 in hidden neuron 1 while 1.822 in hidden neuron 2. What does this mean? We do not know, or at least, it requires quite heavy efforts for us to speculate. Yet looking at the plot it is suffice to say that the training algorithm has converged and therefore the model is ready to be used.

You may also consider take a look at the results summary of our nn model, however, the outputs are not very informative or straight forward for us to interpret (hence you might want to skiip it in the real practice).

```r
summary(nn)
```
```text
##                     Length Class      Mode    
## call                  6    -none-     call    
## response            267    -none-     numeric 
## covariate           534    -none-     numeric 
## model.list            2    -none-     list    
## err.fct               1    -none-     function
## act.fct               1    -none-     function
## linear.output         1    -none-     logical 
## data                  4    data.frame list    
## exclude               0    -none-     NULL    
## net.result            1    -none-     list    
## weights               1    -none-     list    
## generalized.weights   1    -none-     list    
## startweights          1    -none-     list    
## result.matrix        12    -none-     numeric
```


The above outputs might not be very informative. It is only the net.result and result.matrix that we are most interested in. net.result contains the fitted value of Sales, and result.matrix contains the error value of the trained model.

```r
nn$result.matrix['error',]
```
```text
##      error 
## 0.03679449
```
This error metric is also displayed in the neural network plot above (“Error: 0.03679”).

Although we print this out, the value above is not informative enough for us to compare model estimation performance, because the RMSE is still calculated from the scaled data. In fact, at this point both the fitted sales value and error value are computed on the normalised scale. We need to revert the values back to the original scales.



For example for sales:

<img width="549" height="59" alt="截屏2026-02-03 上午11 44 02" src="https://github.com/user-attachments/assets/20205e4c-d3fb-45f8-8494-3670f08ebbbe" />


```r
#revert the fitted value back to original scale following the equation above

fitted.train_nn <- nn$net.result[[1]] * (max(NeuralPriceAdsData$Sales)-min(NeuralPriceAdsData$Sales))+min(NeuralPriceAdsData$Sales)
```
We will calculate the Root Mean Squared Error (RMSE) so that it’s comparable to the error generated by linear regression model.

```r
#use the row index to get the original value of sales in train dataset. 
train_nn.r <- NeuralPriceAdsData$Sales[train_ind_nn]

#calculate the Root Mean Squared Error of train dataset
rmse.train_nn <- (sum((train_nn.r - fitted.train_nn )^2)/nrow(fitted.train_nn))^0.5

rmse.train_nn
```
```text
## [1] 1.411139
```

This time things are comparable. The RMSE is 1.41 which is lower than that of the linear regression of 2.02. Note that, again, both RMSE’s are calculated from the training set for us to compare model performances. What we really care about (or care more about) is the prediction performance using the testing data.

Now we will predict the sales in test data.



## 2.2.4 Compute Prediction Error and Compare Model Performance Using MLR vs. NN


Now let us generate the predicted sales using neural network.

```r
#fit model using test dataset
Predict.nn <- compute(nn,test_nn)

#get the predicted sales in original scale
Predict.nn_ <- Predict.nn$net.result*(max(NeuralPriceAdsData$Sales)-min(NeuralPriceAdsData$Sales))+min(NeuralPriceAdsData$Sales)

#gather test data
test.r_nn <- NeuralPriceAdsData$Sales[-train_ind_nn]

rmse.test_nn <- (sum((test.r_nn - Predict.nn_)^2)/67)^0.5
#print(paste(MSE.lm,MSE.nn))

rmse.test_nn
```
```text
## [1] 1.412666
```


The RMSE for test data is is 1.41 which is lower than that of the linear regression of 2.03 and pretty impressive.

Below we print the predicted sales (Predict) from the testing set, together with actual sales data (Sales). You can observe that the two columns are nearly identical, indicating that the neural net figured out the pattern in the data.


```r
NeuralPriceAdsData[train_ind_nn,'Predict_NN'] <- fitted.train_nn
NeuralPriceAdsData[-train_ind_nn,'Predict_NN'] <- Predict.nn_

head(NeuralPriceAdsData[-train_ind_nn,])
```
```text
## # A tibble: 6 x 5
##   Sales Price Advertising Predict_MLR Predict_NN
##   <dbl> <dbl>       <dbl>       <dbl>      <dbl>
## 1  388.     8          76        384.       385.
## 2  415.     6          52        416.       416.
## 3  434.     5          91        435.       434.
## 4  365      9          92        369.       368.
## 5  404.     7          85        402.       404.
## 6  350     10          71        350.       349.
```


Here in the table above, we can directly inspect the difference in predicted sales using MLR vs. NN method. It seems that predictions generated under the NN method follows the variations of the actual sales more closely.

To further visualize and assess our prediction accuracy, we plot the testing data again. You can see that the plotted actual sales (curve in red) and the predicted sales (curve in green) co-move nicely with each other. This means that our predicted sales are rather accurate.


```r
week <- c(1:67)

## NN method
actual_sales <- NeuralPriceAdsData[-train_ind_nn,'Sales']
predicted_sales_nn <-NeuralPriceAdsData[-train_ind_nn,'Predict_NN']

plot_data_nn <-data.frame(week, actual_sales, predicted_sales_nn)

# The actual sales are in red while the predicted sales are in green

plot(plot_data_nn$week, plot_data_nn$Sales, col="red")
lines(plot_data_nn$week, plot_data_nn$Sales, col="red")
points(plot_data_nn$week, plot_data_nn$Predict_NN, col="green")
lines(plot_data_nn$week, plot_data_nn$Predict_NN,col="green")
```
<img width="710" height="423" alt="截屏2026-02-03 上午11 45 38" src="https://github.com/user-attachments/assets/d070f31c-b698-4d19-a46a-60389cf0e999" />



```r
NeuralPriceAdsData$Week <- c(1:334)

# Start plotting using ggplot2 package

ggplot(data = NeuralPriceAdsData) + 
  geom_line(mapping = aes(x= Week, y= Sales,color="Actual_Sales"), size=0.5)+
  geom_point(aes(x= Week, y= Sales), color = "Dark gray", size=0.5) +
  geom_line(mapping = aes(x= Week, y= Predict_MLR, color="Predict_MLR"), size=0.8)+
  geom_point(aes(x= Week, y= Predict_MLR), color = "Dark Green",size=0.3) + 
  geom_line(mapping = aes(x= Week, y= Predict_NN, color="Predict_NN"), size=0.8)+
  geom_point(aes(x= Week, y= Predict_NN), color = "Dark Blue",size=0.3) + 
  labs(x = "week Index",
         y = "Sales Volume",
         color = "Legend",
       title = "Sales Forecasting",
       subtitle = "Comparing Actual vs. Predicted Sales using MLR and NN Method",
       caption = "Data: Confidential Sales Records",
       tag = "Figure X") +
  theme(panel.grid = element_blank(),
        panel.background = element_rect(fill = "White"),
        plot.title = element_text(size = 14, face = "bold"),
        plot.background = element_rect(colour = "black", fill=NA, size=1),
        legend.key = element_rect(fill = "White"))+
  scale_color_manual(values = c("Actual_Sales" = "Dark Gray", "Predict_MLR" = "Dark Green", "Predict_NN"="Dark Blue"))
```
<img width="683" height="483" alt="截屏2026-02-03 上午11 46 08" src="https://github.com/user-attachments/assets/c2eba9b8-dda7-47ae-acfa-28274a06fc02" />




This is somewhat too dense isn’t it. To solve such an issue, you might consider adding one line of codes at the top of the above chunk: “png(filename=”Model_Comparison.png”, width=1000, height=1000)“. Here you are saving the plot into a png file with a width of 1000 pixels and height of 1000 pixels. You can adjust the size of the images as you like.

Once you run the chunk of codes, you won’t see anything here in this window. Instead, you can find an image named “Model_Comparison.png” on your device, which you can open and inspect.

What else can we do? This is a question left for you :).



Question: It seems that the NN method outperforms the MLR. Why do you think people would still prefer to use the MLR in the real business practice?



# 3. Cross Validation 


We have briefly covered the nature and techniques of multiple types of cross validation (CV) checks. Here we are going to explore the k-fold CV only. You are welcomed to try out other methods yourself.

We put this section the last because it is optional for you. However, in the real analytics practice, you do cross validation immediately after you fit the models using the training sets, i.e., before you evaluate the prediction performance of the models.

Note that for CV, we need to split the training and the testing sub-sets WITHIN the training set that we defined above (i.e., train_mlr, test_mlr, train_nn, and test_nn). The idea is to play with the training set and leave the testing set untouched.



# 3.1 Cross Validation of MLR model

```r
set.seed(200)

mlr.fit <- lm(Sales~Price+Advertising,data=train_mlr,y = TRUE,x=TRUE)

# See the results, which is the average of 10 validation tests
cv.lm(mlr.fit,K=10)
```
```text
## Mean absolute error        :  1.720513 
## Sample standard deviation  :  0.2203067 
## 
## Mean squared error         :  4.17727 
## Sample standard deviation  :  0.7710931 
## 
## Root mean squared error    :  2.03576 
## Sample standard deviation  :  0.1913398
```
Here we have a summary of MAE, MSE, and RMSE of the 10-fold cross validation of our linear model.



## 3.2 Cross Validation of NN


We are going to implement a fast cross validation using a for loop for the neural network and the cv.lm() function for the linear model.

As far as I know, there is no built-in function in R to perform cross-validation on this kind of neural network, if you do know such a function, please let me know. Here is the 10 fold cross-validated MSE for the NN:

```r
set.seed(567)

cv.error <- NULL
# We want 10-fold Cross Validation so k=10 here
k <- 10

# pbar is just for the sake of visualizing our validation progress; it's a progress bar that helps us understand where are we with the process. In our case where the data is quite small with a simple NN, pbar reaches 100% almost instantly. If you have a very complicated model that takes a lot of time to fit, it might be good to use pbar to help you track things.

pbar <- create_progress_bar('text')
pbar$init(k)
```
```text
## 
  |                                                                            
  |                                                                      |   0%
```
```r
# Now we start the k-fold CV process by writing a loop in R
for(i in 1:k){
  # Again, splitting training and testing data WITHIN the training set.
    index <- sample(1:nrow(train_nn),round(0.8*nrow(train_nn))) 
    train.cv <- train_nn[index,]
    test.cv <- train_nn[-index,]
  # Calculating the predicted outcomes in the testing set
    pr.nn <- compute(nn,test.cv[,1:3])
  # Reverting scaled values to actual values in both predicted and actual data
    pr.nn <- pr.nn$net.result*(max(NeuralPriceAdsData$Sales)-min(NeuralPriceAdsData$Sales))+min(NeuralPriceAdsData$Sales)   
    test.cv.r <- (test.cv$Sales)*(max(NeuralPriceAdsData$Sales)-min(NeuralPriceAdsData$Sales))+min(NeuralPriceAdsData$Sales)   
    # Calculating MSE (you can choose to calculate other metrics). All MSEs are saved in cv.error, a 10x1 vector
    cv.error[i] <- sum((test.cv.r - pr.nn)^2)/nrow(test.cv)    
    pbar$step()
}
```
```text
## 
  |                                                                            
  |=======                                                               |  10%
  |                                                                            
  |==============                                                        |  20%
  |                                                                            
  |=====================                                                 |  30%
  |                                                                            
  |============================                                          |  40%
  |                                                                            
  |===================================                                   |  50%
  |                                                                            
  |==========================================                            |  60%
  |                                                                            
  |=================================================                     |  70%
  |                                                                            
  |========================================================              |  80%
  |                                                                            
  |===============================================================       |  90%
  |                                                                            
  |======================================================================| 100%
```
```r
# Get the mean MSE
mean(cv.error)
```
```text
## [1] 2.035783
```
```r
# Print out all MSE
cv.error
```
```text
##  [1] 1.224336 2.802383 1.762445 1.879523 2.192062 1.932155 1.949170 1.943524
##  [9] 2.526999 2.145232
```
Here from the output above, we get that the mean erro of cross validation (here, we calculated MSE as the error metric) is 2.04. We also print out each of the 10 MSEs in each of our iterations.

```r
# Create a boxplot to visualise the central tendency and the spread of the MSE of our cross validation

boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)
```
<img width="626" height="464" alt="截屏2026-02-03 上午11 48 48" src="https://github.com/user-attachments/assets/fe5fce29-3cca-4a7e-a441-7d3b18c3e496" />

 As you can see, the average MSE for the neural network (2.04) is lower than the one of the linear model (4.18) although there seems to be a certain degree of variation in the MSEs of the cross validation. This may depend on the splitting of the data or the random initialization of the weights in the net. By running the simulation different times with different seeds you can get a more precise point estimate for the average MSE.

# 4. Conclusion Marks and Extensions


In this project, I have explored forecasting in a retail context where multiple periods of historical data are available. Here is a brief review of what I did:

  1. Splitting the dataset into training and testing set via the 80/20 rule, we first estimated a MLR model using the training set, and examined its forecasting performance (here in this tutorial, we used the RMSE) using the testing set.
  
  2. I then did exactly the same things using NN method.
  
  3. Then compared the RMSE of the two models and found that the RMSE under NN method is lower than that under the MLR method, indicating a superior forecasting performance of the NN over the MLR.
  
  4. Finally, visualise the forecasting performance of respective model by plotting the actual and the predicted sales on the same graph for comparison.
  
  5. I also explored k-fold cross validation using the training set.

If you are interested, you are invited to take a look at our supplementary material, which introduces forecasting techniques when your historical data (e.g., sales) are quite intermittent.

You are also invited to test your forecasting skills via the optional Assignment for you. Good luck!

