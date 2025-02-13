# mapper.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def map_data(df):
    """
    Mapper function for transforming the dataset.
    - Converts Age and Salary to appropriate data types.
    - Drops rows with missing values.
    """
    # Clean the data by casting columns to appropriate types
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned.withColumn("Age", col("Age").cast("int")) \
                           .withColumn("Salary", col("Salary").cast("float"))
    
    # Return the cleaned dataframe
    return df_cleaned

def load_data(input_path):
    """
    Load the dataset from CSV into a Spark DataFrame.
    """
    spark = SparkSession.builder.appName("DataProcessing").getOrCreate()
    df = spark.read.option("header", "true").csv(input_path)
    return df
