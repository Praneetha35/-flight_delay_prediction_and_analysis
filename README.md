# Flight Delay Prediction and Analysis

This project analyzes 7.2 million flight records from 2018 to explore flight performance metrics such as delays, cancellations, and diversions. The analysis utilizes R and various libraries for data manipulation, visualization, exploratory data analysis (EDA), and predictive modeling to identify trends in flight performance.

## Tech Stack

- **Programming Language:** R
- **Libraries:**
  - `dplyr` for data manipulation
  - `ggplot2` for data visualization
  - `caret` for modeling and training
  - `pROC` for ROC curve and AUC evaluation
  - `data.table` for efficient data manipulation
  - `corrplot` for correlation matrix visualization

## Features

- **Data Preprocessing:** Handled missing values, feature engineering, and outlier detection.
- **Exploratory Data Analysis (EDA):** Visualized distribution of delays, average delays by airline, and a heatmap for delays by day and month.
- **Predictive Modeling:** Built a linear regression model to predict significant arrival delays (>30 mins) with 95% accuracy.

## Results

- **Exploratory Data Analysis (EDA):** Provided insights into flight delays, including trends and performance by airlines.
- **Predictive Modeling:** A linear regression model achieved an accuracy of 95% in predicting significant delays (>30 mins).
