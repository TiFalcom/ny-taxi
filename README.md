# New York City Taxi Trip Prediction

This project aims to create predictive models to estimate the daily number of taxi trips in New York City. Using the public dataset available on [nyc.gov](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), various machine learning methods and time series models will be explored to capture patterns and make demand forecasts.

# Replication
To replicate the environment, ensure you are using Python 3.12 and run the code above.

``` bash
. install.sh
```

# 0.Exogenous Data Capture
Getting exogenous data that can help prediction.  

Three datasets were used to get exogenous data:  
- [Weather Data from NYC Central Park](https://www.weather.gov/wrh/Climate?wfo=okx)  
- [NYC Taxy Zone](https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/exjm-f27b)

To treat the data access the code bellow.  
[00-Exogenous-Data-Capture.ipynb](notebooks/00-Exogenous-Data-Capture.ipynb)

# 1.Data Basic Process
Cleaning data, merging travels with weather information and data point knowledge base, fixing features to the right type.

[01-Data-Basic-Process.ipynb](notebooks/01-Data-Basic-Process.ipynb)

```bash
python src/data/basic_process.py --config_file=features --dataset_name=full
```

If you computed train and test splited tables, you can union with the script bellow:

```bash
python src/data/concat_data.py --datasets_list=data/interim/train --datasets_list=data/interim/test --dataset_name_output=data/interim/full
```

# 2.Split Data
Split train, test and validation datasets to avoid leak on feature engineering. Using [20240101, 20240131) for training, [20240201, 20240207] for validation, and [20240208, 20240229] for test.

```bash
python src/data/split_train_test.py --dataset_name=full --ymd_train=20240101 --ymd_valid=20240201 --ymd_test=20240208
```

# 3.Feature Engineering
Features were created using pickupdate like 'hour', 'week', ...
To execute this part, run the code bellow.  

```bash
python src/features/create_features.py --dataset_prefix=full
```

# 4.Encoding
Categorical encoding applied to features with less than 15 categories. Used one hot encoding, because we want to aggregate this data and get trends about each categorie.

To create the encoders run this script:  
```bash
python src/features/create_encoders.py --dataset_prefix=full
```

And to encode dataset and save a checkpoint, run this:  
```bash
python src/features/create_encoded_features.py --dataset_prefix=full --encoder_type=2
```

# 5.Aggregate Features
Condense analytical features to hour features. And fix target with lag 1

Features we decided to condense for this experiment was:  
- Minimun and Maximun Temperature - Max  
- Qty. Passengers - Count  
- Day of Week - Count  
- Period of Day - Count  

```bash
python src/features/create_groupby_features.py --dataset_prefix=full
```

# 6.Scale Data
Scale 0-1 for linear models.

To fit the encoders run:
```bash
python src/features/create_scalers.py --dataset_prefix=full
```

And to scale dataset and save a checkpoint, run this:  
```bash
python src/features/create_scaled_features.py --dataset_prefix=full
```

# 7.Feature Enginnering - Lag
Create Lag Features for tabular models, created lag 3 for:
- Maximum and Minimum Temperature  
- Qty Passengers  
- Day of week
- Period of Day  

```bash
python src/features/create_lag_features.py --dataset_prefix=full
```

# 6.Feature Selection
A manually feature selection were done, to acelerate the process, on future works we can implement a robust method.

Features selected for Decision Tree and Boosting:  
['PULocationID', 'Maximum_max', 'Minimum_max','passenger_count_sum', 'day_of_week_max', 'period_of_day_dawn_sum','period_of_day_morning_sum', 'period_of_day_afternoon_sum','period_of_day_evening_sum', 'Maximum_max_l3','Minimum_max_l3', 'passenger_count_sum_l3', 'day_of_week_max_l3','period_of_day_dawn_sum_l3', 'period_of_day_morning_sum_l3','period_of_day_afternoon_sum_l3', 'period_of_day_evening_sum_l3','qty_travels_l3', 'Maximum_max_l2', 'Minimum_max_l2','passenger_count_sum_l2', 'day_of_week_max_l2','period_of_day_dawn_sum_l2', 'period_of_day_morning_sum_l2','period_of_day_afternoon_sum_l2', 'period_of_day_evening_sum_l2','qty_travels_l2', 'Maximum_max_l1', 'Minimum_max_l1','passenger_count_sum_l1', 'day_of_week_max_l1','period_of_day_dawn_sum_l1', 'period_of_day_morning_sum_l1','period_of_day_afternoon_sum_l1', 'period_of_day_evening_sum_l1','qty_travels_l1']


Trends selected for ARIMA and SARIMAX:  
['qty_travels']  

# 7.Tunning
There were no tunning implemented yet, on future works we can implement some methods.  

Params for Decision Tree:  
{random_state : 777, max_depth : 40, min_samples_leaf : 15}  

Params for Boosting:  
{boosting_type : gbdt, max_depth : 5, n_estimators : 500, learning_rate : 0.01, random_state : 777, min_child_samples : 3}

Params for ARIMA:  
{order : [3, 1, 3], freq : h}  

Params for SARIMAX:  
{order : [3, 1, 3], freq : h, seasonal_order : [3, 1, 3, 5]}


# 8.Train Model
On this experiment we decided to work with only two locations (163, 79), because ARIMA/SARIMAX are expensive to fit, after building the entire pipeline we will work with more locations.  

For training the Boosting model run the code bellow:
```bash
python src/model/train_boosting.py --config_file=features --folder_dataset_prefix=lag/full --hyperparams_file=hyperparams --model_suffix=reg_with_lag
```

For training the Tree model run the code bellow:
```bash
python src/model/train_decision_tree.py --config_file=features --folder_dataset_prefix=lag/full --hyperparams_file=hyperparams --model_suffix=reg_with_lag
```

For predicting the ARIMA model run the code bellow:
```bash
python src/model/predict_arima.py --folder_dataset_prefix=aggregated/full --model_suffix=2locations
```

For predicting the SARIMAX model, run the code bellow - Still not working
```bash
python src/model/predict_sarimax.py --folder_dataset_prefix=aggregated/full --model_suffix=2locations
```

WIP:  
- Logistic Regression  
- Linear Regression  
- Basic MLP  
- LSTM  

# 9.Register Experiment (mlflow?)
Use a framework to register experiments (maybe a folder structure)

# 10.Results
Compare results between techniques


## Project Organization

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>



```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

