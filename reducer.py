# reducer.py
from pyspark.sql import functions as F
import pandas as pd

def reduce_data(df_cleaned):
    """
    Reducer function for aggregating and summarizing the data.
    - Calculate average salary by Gender and Country.
    """
    # Group by Gender and Country to calculate average Salary
    gender_salary_df = df_cleaned.groupBy("Gender").agg(F.avg("Salary").alias("avg_Salary"))
    country_salary_df = df_cleaned.groupBy("Country").agg(F.avg("Age").alias("avg_Age"), F.avg("Salary").alias("avg_Salary"))
    
    # Convert to Pandas for easier plotting (be cautious with large datasets)
    gender_salary_df_pandas = gender_salary_df.toPandas()
    country_salary_df_pandas = country_salary_df.toPandas()
    
    return gender_salary_df_pandas, country_salary_df_pandas
