# Load required libraries
library(dplyr)            # For data manipulation
library(ggplot2)          # For visualizations
library(tidyr)            # For reshaping data (pivot_longer)
library(caret)            # For train-test splitting and modeling
library(corrplot)         # For correlation heatmap
library(data.table)       # For data manipulation with data.table
library(pROC)             # For Analyzing and Visualizing

# Load the dataset
data <- read.csv("/Users/prane/Downloads/2018.csv")

# Convert the dataset to data.table format
data_dt <- as.data.table(data)

# Sample the data (0.1%)
set.seed(123)  # Ensure reproducibility
sampled_data <- data %>% sample_frac(0.001)

# Inspect the sampled data
str(sampled_data)
summary(sampled_data)

# Data Preprocessing
# 1. Handle missing values
sampled_data <- sampled_data %>%
  mutate(across(c(CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY), ~ replace_na(., 0)))

# 2. Convert date to Date type
sampled_data$FL_DATE <- as.Date(sampled_data$FL_DATE)

# 3. Create new features: Departure and Arrival Hour
sampled_data <- sampled_data %>%
  mutate(
    dep_hour = floor(CRS_DEP_TIME / 100),
    arr_hour = floor(CRS_ARR_TIME / 100)
  )

# Exploratory Data Analysis (EDA)
# 1. Distribution of Departure Delays
ggplot(sampled_data, aes(x = DEP_DELAY)) +
  geom_histogram(binwidth = 10, fill = "blue", alpha = 0.7) +
  labs(title = "Distribution of Departure Delays", x = "Departure Delay (minutes)", y = "Frequency")

# 2. Average Departure Delay by Airline (using data.table)
avg_dep_delay <- data_dt[, .(avg_dep_delay = mean(DEP_DELAY, na.rm = TRUE)), by = OP_CARRIER]
ggplot(avg_dep_delay, aes(x = reorder(OP_CARRIER, avg_dep_delay), y = avg_dep_delay, fill = avg_dep_delay)) +
  geom_col() +
  coord_flip() +
  labs(title = "Average Departure Delay by Airline", x = "Airline", y = "Average Departure Delay (minutes)")

# Calculate mean arrival delay by carrier using data.table
mean_delay <- data_dt[, .(mean_arrival_delay = mean(ARR_DELAY, na.rm = TRUE)), by = OP_CARRIER]

# Bar plot: Visualize mean arrival delay by carrier
ggplot(mean_delay, aes(x = OP_CARRIER, y = mean_arrival_delay, fill = OP_CARRIER)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(
    title = "Average Arrival Delay by Carrier",
    x = "Carrier",
    y = "Mean Arrival Delay (minutes)"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels

# 3. Heatmap of Average Departure Delay by Day and Month
# Prepare heatmap data
heatmap_data <- sampled_data %>%
  mutate(day = as.numeric(format(FL_DATE, "%d")),
         month = as.numeric(format(FL_DATE, "%m"))) %>%
  group_by(month, day) %>%
  summarize(mean_dep_delay = mean(DEP_DELAY, na.rm = TRUE), .groups = "drop")

# Convert heatmap_data to data.table before using dcast
heatmap_data_dt <- as.data.table(heatmap_data)

# Reshape data for heatmap (Wide Format) using data.table's dcast
heatmap_data_wide <- data.table::dcast(heatmap_data_dt, month ~ day, value.var = "mean_dep_delay")

# Reshape to Long Format for ggplot using tidyr's pivot_longer
heatmap_data_long <- heatmap_data_wide %>%
  pivot_longer(cols = -month, names_to = "day", values_to = "mean_dep_delay") %>%
  mutate(day = as.numeric(day))  # Ensure 'day' is numeric

# Create the heatmap
ggplot(heatmap_data_long, aes(x = day, y = month, fill = mean_dep_delay)) +
  geom_tile() +
  scale_fill_gradient(low = "lightyellow", high = "red", na.value = "white") +
  labs(
    title = "Heatmap of Average Departure Delay",
    x = "Day",
    y = "Month",
    fill = "Mean Delay"
  ) +
  theme_minimal()

# 4. Correlation Matrix for Numerical Variables
numeric_data <- sampled_data %>%
  select(DEP_DELAY, ARR_DELAY, AIR_TIME, DISTANCE, TAXI_OUT, TAXI_IN)

cor_matrix <- cor(numeric_data, use = "complete.obs")
corrplot(cor_matrix, method = "circle", type = "upper", tl.cex = 0.8, title = "Correlation Matrix")

# Predictive Modeling: ARR_DELAY (Arrival Delay)
# Feature Selection and Data Preparation
model_data <- sampled_data %>%
  select(ARR_DELAY, DEP_DELAY, DISTANCE, TAXI_OUT, TAXI_IN, OP_CARRIER) %>%
  mutate(OP_CARRIER = as.factor(OP_CARRIER)) %>%
  filter(!is.na(ARR_DELAY))  # Remove rows with NA target variable

# Handle Missing Values in Predictors
model_data <- model_data %>%
  mutate(across(c(DEP_DELAY, DISTANCE, TAXI_OUT, TAXI_IN), ~ replace_na(., 0)))

# Create a binary classification for significant delay
model_data <- model_data %>%
  mutate(ARR_DELAY_CLASS = ifelse(ARR_DELAY > 30, 1, 0))  # 1 for significant delay, 0 for no delay

# One-hot encode categorical variable (OP_CARRIER)
model_data$OP_CARRIER <- as.numeric(model_data$OP_CARRIER)

# Split Data into Training and Testing Sets
set.seed(123)
train_index <- createDataPartition(model_data$ARR_DELAY_CLASS, p = 0.8, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Model 1: Linear Regression (For Classification)
linear_model <- lm(ARR_DELAY_CLASS ~ ., data = train_data)
linear_preds <- predict(linear_model, newdata = test_data)
linear_preds_class <- ifelse(linear_preds > 0.5, 1, 0)  # Convert to binary predictions

# Evaluate Linear Regression (Classification Metrics)
linear_cm <- confusionMatrix(as.factor(linear_preds_class), as.factor(test_data$ARR_DELAY_CLASS))
linear_accuracy <- linear_cm$overall['Accuracy']
linear_precision <- linear_cm$byClass['Precision']
linear_recall <- linear_cm$byClass['Recall']
linear_f1 <- linear_cm$byClass['F1']

# Compute ROC and AUC for Linear Regression
linear_roc <- roc(test_data$ARR_DELAY_CLASS, linear_preds)
linear_auc <- auc(linear_roc)

cat("Linear Regression (Classification):\n")
cat("Accuracy:", linear_accuracy, "\nPrecision:", linear_precision, "\nRecall:", linear_recall, "\nF1-score:", linear_f1, "\nAUC:", linear_auc, "\n")