# main.py
import matplotlib.pyplot as plt
import seaborn as sns
from mapper import load_data, map_data
from reducer import reduce_data

def plot_gender_salary(gender_salary_df):
    """
    Plot the average salary by gender.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Gender', y='avg_Salary', data=gender_salary_df)
    plt.title("Average Salary by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Average Salary")
    plt.show()

def plot_country_salary(country_salary_df):
    """
    Plot the average salary and age by country.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Average Salary by Country
    sns.barplot(x='Country', y='avg_Salary', data=country_salary_df, ax=ax[0])
    ax[0].set_title("Average Salary by Country")
    ax[0].set_xlabel("Country")
    ax[0].set_ylabel("Average Salary")
    
    # Plot 2: Average Age by Country
    sns.barplot(x='Country', y='avg_Age', data=country_salary_df, ax=ax[1])
    ax[1].set_title("Average Age by Country")
    ax[1].set_xlabel("Country")
    ax[1].set_ylabel("Average Age")
    
    plt.tight_layout()
    plt.show()

def main(input_path):
    # Load data
    df = load_data(input_path)
    
    # Transform data
    df_cleaned = map_data(df)
    
    # Aggregate data
    gender_salary_df, country_salary_df = reduce_data(df_cleaned)
    
    # Visualize results
    plot_gender_salary(gender_salary_df)
    plot_country_salary(country_salary_df)

if __name__ == "__main__":
    # Specify the path to your data
    input_path = "./retaildata.csv"
    main(input_path)
