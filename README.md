# Predicting-Power-Outages-Durations
by Sarah Borsotto (sborsott@ucsd.edu) and Hector Gallo (hgallo@ucsd.edu)

---

## Problem Identification

Major power outages can have detrimental effects on public safety, infrastructure, and the economy. Therefore, predicting power outage severity is crucial for proactive planning and resource allocation. It enables utilities and emergency services to allocate resources efficiently and mitigate potential damages, ultimately reducing the impact on the community. We seek to anticipate power outage severity by predicting outage durations, since longer power outages tend to be more negatively influential. Given that a power outage just occurred, we want to know how long the outage will last depending on the cause of the power outage, as well as population metrics, cost metrics, season, geography, and time. We will be implementing a k-Nearest Neighbors (k-NN) regressor that incorporates these variables to predict outage duration. In order to evaluate our model for its accuracy, we plan to use RMSE and R-squared as our metrics. We hope to minimize RMSE while maximizing R-squared.

Here is a brief explanation of how the k-NN regressor model works: When a new data point is given for prediction, the algorithm identifies the 'k' nearest neighbors of that point in the feature space in terms of euclidean distance. That is, it will calculate the distance between the new data point and all points in the training set. Then, it will sort the distances in ascending order to identify the 'k' nearest neighbors. The predicted value for the new data point is often the weighted average (or mean) of the target values of its 'k' nearest neighbors. The weights can be based on the inverse of the distance or any other suitable weighting scheme. The predicted value for the new data point is then output as the result of the regression. In summary, KNN Regressor makes predictions based on the average of the target values of the 'k' nearest neighbors in the feature space. The simplicity and flexibility of KNN make it a popular choice for regression tasks, especially in cases where the underlying relationships in the data are non-linear or where interpretability is essential.

Based on our research, we know that the features that generally have a higher impact on the power outage durations are the cauge of the outage, the peak hours of energy consumption (which are generally between 4:00 pm and 9:00 pm), season, wheter the area is urban or rural, population density, state, and year. At the “time of prediction”, we would have access to  all of these features since all of them are knowledge that we have access to before a power outage even starts. Some of the features that we wouldn't have access to before the power outage, for example, would be customers affected, and outage restoration date.

---

## Baseline Model

What can we expect in terms of outage duration in general? Will the values be high? Will there be a lot of variation? Let's investigate the distribution of power outage durations by plotting a boxplot of outage durations.

| Column                | Data Type          |
|-----------------------|--------------------|
| YEAR                  | float64            |
| MONTH                 | float64            |
| U.S._STATE            | object             |
| CAUSE.CATEGORY        | object             |
| OUTAGE.DURATION       | float64            |
| RES.PRICE             | float64            |
| COM.PRICE             | float64            |
| IND.PRICE             | float64            |
| TOTAL.PRICE           | float64            |
| RES.SALES             | float64            |
| COM.SALES             | float64            |
| IND.SALES             | float64            |
| TOTAL.SALES           | float64            |
| RES.PERCEN            | float64            |
| COM.PERCEN            | float64            |
| IND.PERCEN            | float64            |
| RES.CUSTOMERS         | float64            |
| COM.CUSTOMERS         | float64            |
| IND.CUSTOMERS         | float64            |
| TOTAL.CUSTOMERS       | float64            |
| RES.CUST.PCT          | float64            |
| COM.CUST.PCT          | float64            |
| IND.CUST.PCT          | float64            |
| PC.REALGSP.STATE      | float64            |
| PC.REALGSP.USA        | float64            |
| PC.REALGSP.REL        | float64            |
| PC.REALGSP.CHANGE     | float64            |
| UTIL.REALGSP          | float64            |
| TOTAL.REALGSP         | float64            |
| UTIL.CONTRI           | float64            |
| PI.UTIL.OFUSA         | float64            |
| POPULATION            | float64            |
| POPPCT_URBAN          | float64            |
| POPPCT_UC             | float64            |
| POPDEN_URBAN          | float64            |
| POPDEN_UC             | float64            |
| POPDEN_RURAL          | float64            |
| AREAPCT_URBAN         | float64            |
| AREAPCT_UC            | float64            |
| PCT_LAND              | float64            |
| PCT_WATER_TOT         | float64            |
| PCT_WATER_INLAND      | float64            |
| OUTAGE.START          | datetime64[ns]     |


Wow! There seems to be a lot of outliers in our dataset. The boxplot appears small compared to the rest of our graph because the mean outage duration is very far from the max outage duration. This could negatively influence our prediction model, since outliers could skew our prediction towards higher values, even though most of our data centers around a lower threshold. In order to ensure that our prediction model follows the trend of the majority of our data, and that it doesn't become too biased from outlier datapoints, we will only look at power outages that are within the upper fence of the boxplot, which is 7020 minutes.

We still have a good amount of data to work with. As we noticed in the boxplot, only a small amount of our data is above 7020 minutes.

We also don't have any null values, so we can work with our new dataset directly without having to drop or impute null values.

Since we are using a regression model to predict outage duration times, we need to find variables that can act as good predictors for our dependent variable. We have an abundant list of quantitative variables we could use to predict outage duration times, so let's see if we can find any correlations between each variable and outage duration.  

Below we generated scatter plots of each quantitative variable with outage duration times. While there are only 3 graphs illustrated below, we went through all of the graphs in increments of 5 and looked for any possible relationships between outage durations and other quantitative variables.

Unfortunately it doesn't look like there are any clear relationships between duration times and the other quantitative variables, such as COM.CUSTOMERS. Most of the points seem to be clustered. Some graphs did look more promising than others, including RES.CUST.PCT, COM.CUST.PCT, IND.CUST.PCT, PC.REALGSP.USA, UTIL.CONTRI, POPPCT_UC, POPDEN_RURAL, PCT_LAND, and PCT_WATER_TOT. To get a closer look, let's see what the correlation coefficents are between outage duration and each quantitative variable.

The correlation coefficients seem to fit our interpretation of the graphs. Yet, these variables have relatively low correlation with outage duration, the highest being only 0.266. As such, our quantitative variables may not be good linear predictors for outage duration. We will further explore this in our final model, but for now let's focus on categorical variables, as they may tell us more about outage duration times.

Intuitively, a strong predictor for outage duration may be the cause of the power outage. In our analysis, we are assuming that the power outage has just occurred, and that the cause of the outage is known. Based on this information, we may be able to generalize outage duration times. For example, severe-weather induced power outages may last longer than intentional attacks since there is a higher sense of urgency for attacks, and because most of the time companies must wait for the harsh weather conditions to pass. Another parameter that could significantly influence power outage durations is location. Different states may have different weather patterns, population sizes, availability of resources, local regulations, and so on. Accordingly, for our baseline model we focus on these two predictors, CAUSE.CATEGORY and U.S._STATE. These variables are both nominal. CAUSE,CATEGORY depicts the cause of the power outage, like severe-weather or intentional attack, while U.S._STATE depicts the state where the power outage occurred, like California. We are one-hot encoding both variables so we can input them into our KNN regression model.

|    | train_rmse_err | test_rmse_err | train_r2_err | test_r2_err |
|---:|-----------------|---------------|--------------|-------------|
|  1 | 1642.270999     | 1875.361017   | 0.050308     | -0.511042    |
|  2 | 1373.207860     | 1484.866550   | 0.336004     | 0.052713     |
|  3 | 1332.848949     | 1467.738424   | 0.374460     | 0.074441     |
|  4 | 1372.078114     | 1400.169371   | 0.337096     | 0.157698     |
|  5 | 1349.243155     | 1379.142951   | 0.358977     | 0.182806     |
|... | ...             | ...           | ...          | ...          |
| 96 | 1370.640540     | 1293.449097   | 0.338484     | 0.281204     |
| 97 | 1371.944761     | 1295.082746   | 0.337225     | 0.279387     |
| 98 | 1372.836064     | 1293.390659   | 0.336363     | 0.281269     |
| 99 | 1373.395286     | 1294.065417   | 0.335822     | 0.280519     |
|100 | 1374.111346     | 1293.866986   | 0.335130     | 0.280740     |

The output of our knn_reg_perf function is a dataframe of training and testing rmse and r-squared values from 1 to 200. The index of the dataframe denotes the number of neighbors used in the regressor. We plot this below to get a better view of our error data.

<iframe src="assets/fig4.html" width=800 height=600 frameBorder=0></iframe>

Our rmse for both training and testing appears to be minimized around 20. What does our r-squared look like?

<iframe src="assets/fig5.html" width=800 height=600 frameBorder=0></iframe>

Luckily for us, both our training and test r-squared errors appear to be maximized at around 20. To get a more accurate number for our neighbor metric we will use gridsearchcv.

After using gridsearch we see that 23 is our optimized parameter for number of neighbors. Below is the data for our rmse and r-2 squared values for testing and training with 23 neighbors using a KNN model.

|                | Value        |
|----------------|--------------|
| train_rmse_err | 1294.576506  |
| test_rmse_err  | 1286.928656  |
| train_r2_err   | 0.409869     |
| test_r2_err    | 0.288433     |


As we can see, our rmse is about 1000 minutes, which is about 16 hours. Additionally, our r-squared values are around .30, which isn't very high. While our rmse is high, we have a lot of variation in our outage duration data that could explain this score. We believe that for a simple KNN regression model these scores are not bad, but we hope to improve them in our final model by adding more features.



---

## Final Model

---

## Fairness Analysis

---
