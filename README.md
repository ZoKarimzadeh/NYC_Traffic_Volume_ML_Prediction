# NYC Traffic Volume Machine Learning Prediction

This repository contains the code for predicting NYC traffic volumes using Ridge Regression and Generalized Additive Models (GAM). The project involves data preprocessing, model training, evaluation, and comparison of the two methods.

## Project Overview
This project aims to predict traffic volumes in New York City using machine learning models. The dataset used for this project is "NYC Automated Traffic Volume Counts" obtained from Kaggle.

## Data Description
- **Dataset Name:** NYC Automated Traffic Volume Counts
- **Source:** [Kaggle](https://www.kaggle.com/datasets/aadimator/nyc-automated-traffic-volume-counts)
- **Size:** 27,190,511 records

## Requirements
To run the code in this repository, you need the following R packages:
- `glmnet`
- `mgcv`
- `dplyr`
- `tidyr`
- `ggplot2`

You can install the required packages using the following commands:
```r
install.packages("glmnet")
install.packages("mgcv")
install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")
