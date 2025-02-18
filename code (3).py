# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("AdvancedAnalytics") \
    .getOrCreate()

# 1. Load the data (Assuming your CSV is stored locally or on a cloud bucket)
df = spark.read.option("header", "true").csv("./retaildata.csv")

# Show the first few rows of the dataset to inspect it
df.show()

# 2. Data Cleaning
# Check for null values and data types
df.printSchema()

# Remove rows with null values
df_cleaned = df.dropna()

# Convert appropriate columns to numeric (e.g., Age, Salary)
df_cleaned = df_cleaned.withColumn("Age", col("Age").cast("int")) \
    .withColumn("Salary", col("Salary").cast("float"))

# Show the cleaned dataset
df_cleaned.show()

# Display some basic statistics (e.g., mean, min, max, etc.)
df_cleaned.describe().show()

# 3. Exploratory Data Analysis (EDA)
# Grouping by Gender to see average Salary by Gender
gender_salary_df = df_cleaned.groupBy("Gender").mean("Salary").toPandas()

# Plot average salary by Gender
sns.barplot(x='Gender', y='avg(Salary)', data=gender_salary_df)
plt.title("Average Salary by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Salary")
plt.show()

# Group by Country and see the distribution of Age and Salary
country_salary_df = df_cleaned.groupBy("Country").agg({"Age": "avg", "Salary": "avg"}).toPandas()

# Plot average Age and Salary by Country
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Average Salary by Country
sns.barplot(x='Country', y='avg(Salary)', data=country_salary_df, ax=ax[0])
ax[0].set_title("Average Salary by Country")
ax[0].set_xlabel("Country")
ax[0].set_ylabel("Average Salary")

# Plot 2: Average Age by Country
sns.barplot(x='Country', y='avg(Age)', data=country_salary_df, ax=ax[1])
ax[1].set_title("Average Age by Country")
ax[1].set_xlabel("Country")
ax[1].set_ylabel("Average Age")

plt.tight_layout()
plt.show()

# Convert to Pandas for easy plotting (be careful with large datasets)
df_pandas = df_cleaned.toPandas()

# Plot Salary vs Age for different countries
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Salary', hue='Country', data=df_pandas)
plt.title("Salary vs Age by Country")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

# 4. Feature Engineering
# One-Hot Encoding for Gender and Country
indexer_gender = StringIndexer(inputCol="Gender", outputCol="GenderIndex")
encoder_gender = OneHotEncoder(inputCol="GenderIndex", outputCol="GenderVec")

indexer_country = StringIndexer(inputCol="Country", outputCol="CountryIndex")
encoder_country = OneHotEncoder(inputCol="CountryIndex", outputCol="CountryVec")

# VectorAssembler to combine features into a single vector
assembler = VectorAssembler(inputCols=["Age", "Salary", "GenderVec", "CountryVec"], outputCol="features")

# Create a pipeline for transformations
pipeline = Pipeline(stages=[indexer_gender, encoder_gender, indexer_country, encoder_country, assembler])

# Fit and transform the data
df_transformed = pipeline.fit(df_cleaned).transform(df_cleaned)

df_transformed.show()

# 5. Train Machine Learning Model (Linear Regression to Predict Salary)
lr = LinearRegression(featuresCol="features", labelCol="Salary")

# Train the model
lr_model = lr.fit(df_transformed)

# Make predictions
predictions = lr_model.transform(df_transformed)

# 6. Evaluate the model using RMSE
evaluator = RegressionEvaluator(labelCol="Salary", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) on training data = {rmse}")

# 7. Visualize the model's residuals
residuals = predictions.select("prediction", "Salary").toPandas()
residuals["residual"] = residuals["Salary"] - residuals["prediction"]

# Plot residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals["residual"], kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# 8. Hyperparameter Tuning (Grid Search)
# Build a grid of hyperparameters for tuning
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.01, 0.001])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build())

# Set up Cross-Validation
cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=RegressionEvaluator(labelCol="Salary", predictionCol="prediction", metricName="rmse"),
                    numFolds=3)

# Fit the model with cross-validation
cv_model = cv.fit(df_transformed)

# Get the best model
best_model = cv_model.bestModel
print(f"Best Model: {best_model}")

# Evaluate the best model
best_predictions = best_model.transform(df_transformed)
best_rmse = evaluator.evaluate(best_predictions)
print(f"Best RMSE after Hyperparameter Tuning: {best_rmse}")

# 9. Visualize the predicted vs actual salary
best_predictions_df = best_predictions.select("Salary", "prediction").toPandas()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Salary', y='prediction', data=best_predictions_df)
plt.title("Predicted vs Actual Salary")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.show()

# 10. Model Prediction on New Data
# Let's assume new data is available as a list of tuples
new_data = [(30, "Female", "USA"), (45, "Male", "Germany")]
columns = ["Age", "Gender", "Country"]

new_df = spark.createDataFrame(new_data, columns)

# Transform the new data
new_df_transformed = pipeline.fit(new_df).transform(new_df)

# Make predictions
new_predictions = best_model.transform(new_df_transformed)
new_predictions.show()

# 11. Save the Model (for later use)
best_model.save("path_to_save_model/salary_prediction_model")

# 12. Stop Spark session (cleanup)
spark.stop()
