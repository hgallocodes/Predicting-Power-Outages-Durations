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
|   train_rmse_err |   test_rmse_err |   train_r2_err |   test_r2_err |\n|-----------------:|----------------:|---------------:|--------------:|\n|          1642.27 |         1875.36 |      0.0503079 |    -0.511042  |\n|          1373.21 |         1484.87 |      0.336004  |     0.0527128 |\n|          1332.85 |         1467.74 |      0.37446   |     0.0744409 |\n|          1372.08 |         1400.17 |      0.337096  |     0.157698  |\n|          1349.24 |         1379.14 |      0.358977  |     0.182806  |\n|          1309.5  |         1356.6  |      0.396187  |     0.209297  |\n|          1285.77 |         1335.57 |      0.417871  |     0.233623  |\n|          1293.53 |         1327.55 |      0.410826  |     0.242808  |\n|          1293.78 |         1312.59 |      0.41059   |     0.259776  |\n|          1284.13 |         1304.33 |      0.419355  |     0.269059  |\n|          1297.96 |         1304.7  |      0.406781  |     0.268649  |\n|          1295.13 |         1295.87 |      0.409361  |     0.278508  |\n|          1291.71 |         1296.82 |      0.412482  |     0.277455  |\n|          1290.29 |         1298.34 |      0.413774  |     0.275759  |\n|          1290.45 |         1295.96 |      0.413626  |     0.278408  |\n|          1294.58 |         1286.93 |      0.409869  |     0.288433  |\n|          1293.46 |         1277.11 |      0.410887  |     0.299245  |\n|          1288.22 |         1273.22 |      0.415654  |     0.303516  |\n|          1290.32 |         1291.33 |      0.413741  |     0.283555  |\n|          1288.56 |         1291.52 |      0.415341  |     0.283351  |\n|          1290.84 |         1286.49 |      0.413269  |     0.288921  |\n|          1292.9  |         1286.39 |      0.411398  |     0.289032  |\n|          1288.53 |         1285.41 |      0.415365  |     0.290107  |\n|          1290.1  |         1275.14 |      0.413941  |     0.301405  |\n|          1289.62 |         1275.5  |      0.41438   |     0.30101   |\n|          1289.62 |         1272.17 |      0.414376  |     0.304665  |\n|          1294.33 |         1278    |      0.410093  |     0.29827   |\n|          1294.45 |         1272.34 |      0.409984  |     0.304469  |\n|          1300.09 |         1277.86 |      0.404832  |     0.298426  |\n|          1300.51 |         1285.11 |      0.404446  |     0.290447  |\n|          1300.64 |         1285.22 |      0.404325  |     0.290326  |\n|          1301.73 |         1290.36 |      0.403327  |     0.284633  |\n|          1301.04 |         1284.51 |      0.403959  |     0.291106  |\n|          1303.1  |         1282.34 |      0.402068  |     0.293494  |\n|          1304.26 |         1283.29 |      0.401006  |     0.292452  |\n|          1306.87 |         1284.93 |      0.398608  |     0.290639  |\n|          1307.55 |         1284.14 |      0.397979  |     0.291518  |\n|          1308.17 |         1286.79 |      0.397413  |     0.288586  |\n|          1306.97 |         1286.55 |      0.398513  |     0.288849  |\n|          1306.77 |         1283.83 |      0.398702  |     0.29186   |\n|          1308.27 |         1278.45 |      0.397321  |     0.297773  |\n|          1310.73 |         1283.09 |      0.395052  |     0.292675  |\n|          1312.97 |         1286.46 |      0.392977  |     0.288954  |\n|          1315.72 |         1289.97 |      0.390434  |     0.285071  |\n|          1319.04 |         1291.66 |      0.387355  |     0.28319   |\n|          1320.12 |         1292.79 |      0.386355  |     0.281936  |\n|          1319.27 |         1292.67 |      0.387141  |     0.282068  |\n|          1321.99 |         1295.94 |      0.384607  |     0.278433  |\n|          1323.19 |         1292.97 |      0.383491  |     0.281738  |\n|          1323.67 |         1295.57 |      0.383049  |     0.278848  |\n|          1325.39 |         1290.82 |      0.381437  |     0.284125  |\n|          1325.74 |         1288.97 |      0.381118  |     0.286179  |\n|          1328.63 |         1290.72 |      0.378412  |     0.284233  |\n|          1328.97 |         1285.51 |      0.378099  |     0.290003  |\n|          1328.66 |         1284.4  |      0.378386  |     0.291227  |\n|          1332.89 |         1282.84 |      0.374417  |     0.292951  |\n|          1331.26 |         1277.58 |      0.375947  |     0.298734  |\n|          1330.88 |         1276.3  |      0.37631   |     0.300134  |\n|          1332.83 |         1277.92 |      0.374481  |     0.298355  |\n|          1333.66 |         1276.51 |      0.3737    |     0.299904  |\n|          1336.65 |         1276.09 |      0.370889  |     0.300365  |\n|          1338.13 |         1276    |      0.369496  |     0.300472  |\n|          1339.17 |         1273.18 |      0.368516  |     0.303551  |\n|          1340.87 |         1276.36 |      0.366911  |     0.300072  |\n|          1341.57 |         1274.25 |      0.366244  |     0.302389  |\n|          1342.34 |         1273.77 |      0.365517  |     0.302914  |\n|          1342.61 |         1272.22 |      0.365263  |     0.304609  |\n|          1342.73 |         1272.69 |      0.365154  |     0.304091  |\n|          1346.44 |         1275.76 |      0.361637  |     0.300728  |\n|          1347.08 |         1275.63 |      0.361035  |     0.300871  |\n|          1345.97 |         1277.16 |      0.362082  |     0.2992    |\n|          1346.69 |         1277.98 |      0.361403  |     0.298294  |\n|          1345.99 |         1278.33 |      0.362061  |     0.297906  |\n|          1347.7  |         1277.26 |      0.360445  |     0.299086  |\n|          1351.92 |         1279.14 |      0.356434  |     0.297021  |\n|          1353    |         1281.78 |      0.355403  |     0.294121  |\n|          1353.07 |         1281.37 |      0.355333  |     0.294566  |\n|          1355.09 |         1280.56 |      0.353413  |     0.295459  |\n|          1357.97 |         1280.18 |      0.350662  |     0.295874  |\n|          1356.15 |         1282.01 |      0.352397  |     0.293866  |\n|          1356.81 |         1280.82 |      0.351771  |     0.295177  |\n|          1355.81 |         1279.31 |      0.352726  |     0.296835  |\n|          1357.2  |         1282.66 |      0.351397  |     0.293141  |\n|          1358.7  |         1282.46 |      0.349962  |     0.293371  |\n|          1359.01 |         1281.94 |      0.34966   |     0.293936  |\n|          1361.13 |         1284.45 |      0.34763   |     0.291171  |\n|          1363.01 |         1284.1  |      0.345831  |     0.291559  |\n|          1362.33 |         1284.18 |      0.34648   |     0.291467  |\n|          1364.09 |         1285.3  |      0.344797  |     0.29023   |\n|          1365.38 |         1288.88 |      0.343555  |     0.286271  |\n|          1365.01 |         1288.57 |      0.343912  |     0.286617  |\n|          1366.73 |         1289.9  |      0.342258  |     0.285139  |\n|          1367.63 |         1292.91 |      0.341384  |     0.281798  |\n|          1368.98 |         1292.75 |      0.340085  |     0.281976  |\n|          1369.55 |         1292.18 |      0.339533  |     0.282614  |\n|          1370.64 |         1293.45 |      0.338484  |     0.281204  |\n|          1371.94 |         1295.08 |      0.337225  |     0.279387  |\n|          1372.84 |         1293.39 |      0.336363  |     0.281269  |\n|          1373.4  |         1294.07 |      0.335822  |     0.280519  |\n|          1374.11 |         1293.87 |      0.33513   |     0.28074   |
The output of our knn_reg_perf function is a dataframe of training and testing rmse and r-squared values from 1 to 200. The index of the dataframe denotes the number of neighbors used in the regressor. We plot this below to get a better view of our error data.

Our rmse for both training and testing appears to be minimized around 20. What does our r-squared look like?

Luckily for us, both our training and test r-squared errors appear to be maximized at around 20. To get a more accurate number for our neighbor metric we will use gridsearchcv.

After using gridsearch we see that 23 is our optimized parameter for number of neighbors. Below is the data for our rmse and r-2 squared values for testing and training with 23 neighbors using a KNN model.

As we can see, our rmse is about 1000 minutes, which is about 16 hours. Additionally, our r-squared values are around .30, which isn't very high. While our rmse is high, we have a lot of variation in our outage duration data that could explain this score. We believe that for a simple KNN regression model these scores are not bad, but we hope to improve them in our final model by adding more features.



---

## Final Model

---

## Fairness Analysis

---
