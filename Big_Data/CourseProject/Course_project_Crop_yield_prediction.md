# Prediction of yields and Futures price of crops in Europe using various data sources

Team MiECa:
- Mikhail Sarafanov
- Egor Turukhanov
- Juan Camilo Diosa E.

## Crop yield prediction

In order to make the most accurate forecasts, 4 different yield forecasting models were used:
* Linear regression model
* «Distribution analysis»
* Bayesian network
* Autoregressive Integrated Moving Average (ARIMA)

Then, model predictions were ensembled using Kalman filters.


### Linear regression

To forecast yields in September or October, information about the sum of active temperatures, precipitation, and average pressure for the first half of the year is used. We also used nonlinear models for forecasting, but they did not exceed the accuracy of multiple linear regression.
![Regression.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_6.png)

The source code is presented here:
#### [Prediction_regression](https://github.com/Dreamlone/ITMO_Masters_degree/blob/master/Methods_and_models_for_multivariate_data_analysis/PredictiveModels/Prediction_regression.py)



```python
country = 'France'
crop = 'Barley (tonnes per hectare)'

reals = []
preds = []
ys = []
for j in [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
    pred, real = predict('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid', j, country, crop)
    reals.append(real)
    preds.append(pred)
    ys.append(j)

ys = np.array(ys)
reals = np.array(reals)
preds = np.array(preds)
print_metrics(reals, preds)
residuals_plots(reals, preds, color = 'green')

ys = np.ravel(ys)
reals = np.ravel(reals)
preds = np.ravel(preds)
df = pd.DataFrame({'Year': ys,
                   'Prediction': preds,
                   'Real': reals})

file_name = 'Regression_' + country + '_' + crop + '.csv'
file_path = os.path.join('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/RESULTS', file_name)
df.to_csv(file_path, sep=';', encoding='utf-8', index = False)
```

The syntax for all remaining predictive models does not differ from the one shown above.

### «Distribution analysis»

Idea - Yields for years with similar weather conditions will be similar

The algorithm (approach):
Pairwise comparison of temperature, precipitation, and pressure distributions. Prediction-yield for the year that is most similar to the considered one.

Distributions used:
* Temperature for the first half of the year, temperature for the months: February, April, June;
* Precipitation for the first half of the year, precipitation for the months: February, April, June;
* Pressure for the first half of the year, pressure for the months: February, April, June;

![Distribution_1.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_7.png)

Used for comparison:
* Kruskal-Wallis test;
* To adjust p-value, a multiple testing correction is introduced – the Bonferroni correction.

Calculation for comparing weather conditions:
* The p-value is subtracted from the statistics values;
* The more different the distributions, the greater the difference between sums of values;
* Using Kullback-Leibler divergence provide almost the same result.

![Distribution_2.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_8.png)

The source code is presented here:
#### [Prediction_distribution](https://github.com/Dreamlone/ITMO_Masters_degree/blob/master/Methods_and_models_for_multivariate_data_analysis/PredictiveModels/Prediction_distribution.py)


### Bayesian network

First of all, we've implement data clustering procedure. The values of the considering parameter were grouped into 3 categories (using Gaussian Mixture):

* Crop yield (3 clusters) – hidden state
* Sum of active temperatures (3 clusters)
* Rainfall (3 clusters)
* Mean pressure (3 clusters)

![Clusters.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_9.png)

Using States of hydrometeorological parameters - selecting the yield state:
* For example, if the existing condition is «High heat supply», «Low rainfall supply»,  «High pressure»;
* Then the probability of the state «High yield» - 0.2, «Medium yield» - 0.4, «Low yield» - 0.4;
* Forecast – the average yield of the predicted cluster.


The source code is presented here:
#### [Prediction_markov](https://github.com/Dreamlone/ITMO_Masters_degree/blob/master/Methods_and_models_for_multivariate_data_analysis/PredictiveModels/Prediction_markov.py)

### Autoregressive Integrated Moving Average (ARIMA)

The ARIMA yield forecast is more accurate than «Distribution analysis» and «Bayesian network» prediction.

The source code is presented here:
#### [ARIMA](https://github.com/Dreamlone/ITMO_Masters_degree/blob/master/Methods_and_models_for_multivariate_data_analysis/PredictiveModels/Prediction_ARIMA.py)


### Algorithm ensembling
None of the presented algorithms allowed to overcome the 10% accuracy threshold (MAPE), so the Kalman filter was used to improve the quality of the forecast.

![All_predictions.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_10.png)

#### [Kalman](https://github.com/Dreamlone/ITMO_Masters_degree/blob/master/Methods_and_models_for_multivariate_data_analysis/PredictiveModels/Prediction_Kalman_ensemble.py)

### Results

The model was implemented for:
* Yield of wheat, rice, maize, and barley;
* Countries: Germany, France, Italy, Romania, Poland, Austria, the Netherlands, Switzerland, Spain and the Czech Republic;
* Data for the period from 2008 to 2018 were used for verification;


* MAPE (Linear regression) – 10.42%;
* MAPE (Distribution analysis) – 13.80%;
* MAPE (Bayesian Network) – 14.55%;
* MAPE (ARIMA) – 10.41%;
*  MAPE (Ensemble using Kalman filter) – 9.89%.

![Results.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_11.png)

