# Predicting-Power-Outages-Durations
by Sarah Borsotto (sborsott@ucsd.edu) and Hector Gallo (hgallo@ucsd.edu)

---

## Problem Identification

Major power outages can have detrimental effects on public safety, infrastructure, and the economy. Therefore, predicting power outage severity is crucial for proactive planning and resource allocation. It enables utilities and emergency services to allocate resources efficiently and mitigate potential damages, ultimately reducing the impact on the community. We seek to anticipate power outage severity by predicting outage durations, since longer power outages tend to be more negatively influential. Given that a power outage just occurred, we want to know how long the outage will last depending on the cause of the power outage, as well as population metrics, cost metrics, geography, and time. We will be implementing a k-Nearest Neighbors (k-NN) regressor that incorporates these variables to predict outage duration. In order to evaluate our model for its accuracy, we plan to use RMSE and R-squared as our metrics. We hope to minimize RMSE while maximizing R-squared.

Here is a brief explanation of how the k-NN regressor model works: When a new data point is given for prediction, the algorithm identifies the 'k' nearest neighbors of that point in the feature space in terms of euclidean distance. That is, it will calculate the distance between the new data point and all points in the training set. Then, it will sort the distances in ascending order to identify the 'k' nearest neighbors. The predicted value for the new data point is often the weighted average (or mean) of the target values of its 'k' nearest neighbors. The weights can be based on the inverse of the distance or any other suitable weighting scheme. The predicted value for the new data point is then output as the result of the regression. In summary, KNN Regressor makes predictions based on the average of the target values of the 'k' nearest neighbors in the feature space. The simplicity and flexibility of KNN make it a popular choice for regression tasks, especially in cases where the underlying relationships in the data are non-linear or where interpretability is essential.

Based on our research, we know that the features that generally have a higher impact on the power outage durations are the cauge of the outage, the peak hours of energy consumption (which are generally between 4:00 pm and 9:00 pm), whether the area is urban or rural, population density, state, and year. At the “time of prediction”, we would have access to  all of these features since all of them are knowledge that we have access to before a power outage even starts. Some of the features that we wouldn't have access to before the power outage, for example, would be customers affected, and outage restoration date.

---

## Baseline Model

What can we expect in terms of outage duration in general? Will the values be high? Will there be a lot of variation? Let's investigate the distribution of power outage durations by plotting a boxplot of outage durations.

***Boxplot here***

Wow! There seems to be a lot of outliers in our dataset. The boxplot appears small compared to the rest of our graph because the mean outage duration is very far from the max outage duration. This could negatively influence our prediction model, since outliers could skew our prediction towards higher values, even though most of our data centers around a lower threshold. In order to ensure that our prediction model follows the trend of the majority of our data, and that it doesn't become too biased from outlier datapoints, we will only look at power outages that are within the upper fence of the boxplot, which is 7020 minutes.

We still have a good amount of data to work with. As we noticed in the boxplot, only a small amount of our data is above 7020 minutes.

We also don't have any null values, so we can work with our new dataset directly without having to drop or impute null values.

Since we are using a regression model to predict outage duration times, we need to find variables that can act as good predictors for our dependent variable. We have an abundant list of quantitative variables we could use to predict outage duration times, so let's see if we can find any correlations between each variable and outage duration.

Below we generated scatter plots of each quantitative variable with outage duration times. While there are only 3 graphs illustrated below, we went through all of the graphs in increments of 5 and looked for any possible relationships between outage durations and other quantitative variables.

### Three plots go here ***

Unfortunately it doesn't look like there are any clear relationships between duration times and the other quantitative variables, such as COM.CUSTOMERS. Most of the points seem to be clustered. Some graphs did look more promising than others, including RES.CUST.PCT, COM.CUST.PCT, IND.CUST.PCT, PC.REALGSP.USA, UTIL.CONTRI, POPPCT_UC, POPDEN_RURAL, PCT_LAND, and PCT_WATER_TOT. To get a closer look, let's see what the correlation coefficents are between outage duration and each quantitative variable.

*** Correlation coefficients table ***
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

Luckily for us, both our training and test r-squared errors appear to be maximized at around 10. To get a more accurate number for our neighbor metric we will use gridsearchcv.

Based on these plots, we decided to use 10 as our number of neighbors. Below is the data for our rmse and r-2 squared values for testing and training with 10 neighbors using a KNN model.

|                | Value        |
|----------------|--------------|
| train_rmse_err | 1294.576506  |
| test_rmse_err  | 1286.928656  |
| train_r2_err   | 0.409869     |
| test_r2_err    | 0.288433     |


As we can see, our rmse is about 1300 minutes, which is about 16 hours. Additionally, our r-squared values are around .30, which isn't very high. While our rmse is high, we have a lot of variation in our outage duration data that could explain this score. We believe that for a simple KNN regression model these scores are not bad, but we hope to improve them in our final model by adding more features.

---

## Final Model

For our final model, we want to try to improve upon the baseline model. We will continue to one-hot encode U.S.STATE and CAUSE.CATEGORY. Additionally, we decided to feature engineer two new features, the time of the outage, as well as one of the quantitative features we explored previously. To better predict power outages based on time, we looked at peak hours for energy consumption, which tends to be from 4pm to 9pm. We expect that power outages that started during peak hours would impact the severity of the power outage. In order to include this in our KNN regression model, we transformed our OUTAGE.START column into 0 or 1 for peak vs non-peak hours. We utilized a function transformer, as well as a helper function, to do this. For our other additional feature, we need to look at possible combinations of MONTH and YEAR with the quantitative variables that had the highest correlation coefficient with outage duration. In order to identify an optimized combination, we will perform a manual iterative method that finds the average rmse score for each KNN regressor that includes said combination. For example, in our first iteration, we will employ a stdscalarbygroup transformer on MONTH and RES.CUST.PCT, where MONTH is the month that the power outage occurred in and RES.CUST.PCT is the percent of residential customers served in the U.S. state in percentage. The stdscalarbygroup transformer standardizes the quantitative variable by grouping the categorical variable. This means that the RES.CUST.PCT would be standardized based on the month. We have created a function that goes through each combination, finds the average rmse for 10 different k neighbors, and returns a dictionary with this information that we can then use to pick the best parameters for our stdscalarbygroup transformer.

We see that the lowest rmse score is YEAR and POPPCT_UC, where POPPCT_UC is percentage of the total population of the U.S. state represented by the population of the urban clusters. We will therefore incorporate this into our final baseline model.

|    | train_rmse_err | test_rmse_err | train_r2_err | test_r2_err |
|---:|-----------------|---------------|--------------|-------------|
|  1 |     105.684702  |   1840.035670 |    0.997300  |    1.125817 |
|  2 |     910.761805  |   1633.587352 |    0.708155  |    0.678476 |
|  3 |    1053.317783  |   1545.492809 |    0.557461  |    0.524874 |
|  4 |    1111.706816  |   1505.790350 |    0.526024  |    0.479102 |
|  5 |    1152.026594  |   1475.341636 |    0.483599  |    0.448388 |
|  6 |    1173.089981  |   1435.087526 |    0.467839  |    0.393032 |
|  7 |    1189.793687  |   1438.795772 |    0.442593  |    0.380050 |
|  8 |    1205.354492  |   1424.373958 |    0.413447  |    0.361813 |
|  9 |    1220.067789  |   1423.852698 |    0.397547  |    0.351167 |
| 10 |    1232.077148  |   1420.939274 |    0.382934  |    0.354234 |
| 11 |    1241.251799  |   1417.351332 |    0.374690  |    0.350008 |
| 12 |    1246.214292  |   1415.914804 |    0.370883  |    0.347326 |
| 13 |    1254.141361  |   1395.002525 |    0.360587  |    0.336891 |
| 14 |    1255.123179  |   1374.835545 |    0.353124  |    0.331480 |
| 15 |    1261.756344  |   1379.408443 |    0.349293  |    0.323860 |
| 16 |    1265.901720  |   1376.964935 |    0.345111  |    0.319492 |
| 17 |    1267.736803  |   1364.314707 |    0.341199  |    0.321177 |
| 18 |    1274.951097  |   1364.237629 |    0.335359  |    0.317671 |
| 19 |    1276.773141  |   1355.857298 |    0.331533  |    0.313840 |
| 20 |    1280.277572  |   1343.414699 |    0.328424  |    0.308299 |
| 21 |    1289.140633  |   1347.857259 |    0.325885  |    0.299305 |
| 22 |    1295.931427  |   1346.961912 |    0.322734  |    0.293356 |
| 23 |    1300.007807  |   1350.867097 |    0.318751  |    0.288839 |
| 24 |    1301.418411  |   1350.796035 |    0.318024  |    0.283763 |
| 25 |    1301.110768  |   1354.411895 |    0.314845  |    0.277796 |
| 26 |    1302.816367  |   1355.197238 |    0.310633  |    0.273237 |
| 27 |    1304.752940  |   1354.283373 |    0.306981  |    0.269502 |
| 28 |    1307.214645  |   1351.632309 |    0.303739  |    0.263166 |
| 29 |    1307.815539  |   1349.204668 |    0.298618  |    0.260607 |
| 30 |    1310.412254  |   1353.617253 |    0.297752  |    0.259972 |
| 31 |    1313.379848  |   1358.735671 |    0.293442  |    0.255846 |
| 32 |    1313.252969  |   1359.235548 |    0.292632  |    0.252869 |
| 33 |    1313.030619  |   1363.457462 |    0.290917  |    0.251648 |
| 34 |    1314.326382  |   1366.786330 |    0.287943  |    0.250035 |
| 35 |    1314.517797  |   1367.075777 |    0.284985  |    0.247926 |
| 36 |    1314.315319  |   1366.985188 |    0.283564  |    0.248222 |
| 37 |    1316.566700  |   1365.240244 |    0.281018  |    0.246938 |
| 38 |    1319.331858  |   1363.494964 |    0.277981  |    0.244985 |
| 39 |    1322.186877  |   1367.151699 |    0.277404  |    0.242297 |
| 40 |    1325.100049  |   1370.370870 |    0.275620  |    0.239591 |
| 41 |    1326.888421  |   1370.690068 |    0.273390  |    0.239710 |
| 42 |    1327.323262  |   1374.084230 |    0.272829  |    0.237424 |
| 43 |    1327.005737  |   1377.289026 |    0.269717  |    0.235372 |
| 44 |    1327.306803  |   1377.754334 |    0.268244  |    0.234932 |
| 45 |    1327.606345  |   1373.275409 |    0.266370  |    0.232430 |
| 46 |    1327.964071  |   1373.303228 |    0.264351  |    0.230166 |
| 47 |    1329.346467  |   1372.804152 |    0.262345  |    0.228614 |
| 48 |    1331.733503  |   1373.719738 |    0.260238  |    0.225789 |
| 49 |    1333.236596  |   1375.092115 |    0.258422  |    0.222594 |
| 50 |    1333.171581  |   1375.985922 |    0.256788  |    0.219934 |


We can graph the output of our rmse and r-squared scores below, like we did for our baseline.

<iframe src="assets/fig6.html" width=800 height=600 frameBorder=0></iframe>

<iframe src="assets/fig7.html" width=800 height=600 frameBorder=0></iframe>

Based on these graphs, it seems that rmse is minimized at the beginning, and r-squared is similarily maximized at the beginning. We will therefore use a parameter of 2 for the number of neighbors.

|                | Value        |
|----------------|--------------|
| train_rmse_err | 910.761805   |
| test_rmse_err  | 1633.587352  |
| train_r2_err   | 0.708155     |
| test_r2_err    | 0.678476     |

Our training rmse score is 910, our testing rmse score is 1633, our training r-squared score is 0.71, and our testing r-squared score is .68. As we can see the training rmse score decreased from our baseline and our r-squared scores for both the training and test sets improved by about 0.30. However, our rmse score for the testing set increased slightly. We may be able to improve our model to better fit unknown data with other parameters. We believe that we were able to achieve better training rmse because we have added more features to our model that can help predict outage duration time. More specifically, peak energy consumption hours can impact severity of a power outage since resources are exhausted. Similarily, year and U.S. state percentage population can inform us about possible power outage durations, as some years may have more outages and higher populations could lead to longer durations, as more people are utilizing energy. Lastly, as we saw in our baseline, the location and cause of the category is a strong indicator of outage duration. This makes sense, as most power outages in our dataset are severe-weather induced, meaning that other variables like cost or population don't have great influence over duration time, since the most important attribute is how long the weather event lasts.



---

## Fairness Analysis

---
