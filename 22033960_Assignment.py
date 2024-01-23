import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(file):
    """
    Read the World Bank dataset.
    Returns years as columns dataset and countries as columns dataset.
    """
    # Read dataset
    df = pd.read_csv(file).iloc[:, :-1]
    
    # Transform dataset with years as columns
    df_years = df

    # Transform dataset with countries as columns
    df_countries = df.set_index('Country Name').T
    df_countries.index.name = 'Indicator'
    df_countries = df_countries.reset_index()

    return df_years, df_countries

# Read data
file = "world_bank_data.csv"
df_years, df_countries = read_data(file)
print(df_years.head())
print(df_countries.head())


def explore_data(df, indicators, countries):
    
    """
    Explore statistical properties of the selected indicators
    and countries, and return summary statistics.
    """
    # Filter dataset by selected countries
    selected_data = df[df['Country Name'].isin(countries)]
    
    summaries = {}
    for ind in indicators:
        # Filter dataset by the current indicator
        selected = selected_data[selected_data["Indicator Name"] == ind]
        
        # Drop unnecessary columns
        selected.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1, inplace=True)
        
        # Set the index to 'Country Name', transpose the dataframe, and calculate summary statistics
        summary = selected.set_index("Country Name").T.describe()
        
        # Add summary statistics for the current indicator to the summaries dictionary
        summaries[ind] = summary

    return summaries


# Defining indicators
indicators_of_interest = [
    'Energy use (kg of oil equivalent per capita)',
    'Electric power consumption (kWh per capita)',
    'Access to electricity (% of population)',
    'Renewable energy consumption (% of total final energy consumption)'
]

# Defining Countries (BRICS Group Countries)
countries_of_interest = [
    'Brazil', 'Russian Federation', 'India', 'China',  'South Africa']

# Explore statistical properties of selected indicators and countries
summaries = explore_data(
    df_years, indicators_of_interest, countries_of_interest)

for ind, summary in summaries.items():
    print(f"Statistical Characteristics of Selected Countries for Indicator {str(ind).upper()} are...")
    print(summary)
    
    
def visualize_data(df, indicator, countries, title):
    """
    Visualize the data using line plots for the selected indicators and countries.
    """
    # Filtering dataset by selected countries
    selected_data = df[df['Country Name'].isin(countries)]

    # Filtering dataset by selected indicator while setting year as index
    selected_data = selected_data[selected_data['Indicator Name'] == indicator]
  
    # Removing unnecessary columns
    selected_data = selected_data.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1).set_index("Country Name").T
    
    
    # Create line plot
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=selected_data, dashes=False)
    plt.title(title, y=1.02)
    plt.ylabel(indicator)
    plt.xlabel('Year')
    plt.xticks(rotation=90)
    plt.legend(countries)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()
    
for indicator in indicators_of_interest:
        visualize_data(df_years, indicator, countries_of_interest,
                       f"{indicator} for BRICS countries")

def visualize_correlation(df, ind_x, ind_y, countries):
    """
    Visualize the relationship between two indicators for the selected countries
    using scatter plots with regression lines.
    """
    
    # Create a figure and axes
    plt.figure(figsize=(12, 5))
    for i, country in enumerate(countries):
        x = df[(df["Country Name"] == country) & (df_years["Indicator Name"] == ind_x)].iloc[:, 4:].T.values
        y = df[(df["Country Name"] == country) & (df_years["Indicator Name"] == ind_y)].iloc[:, 4:].T.values
        sns.regplot(x=x, y=y, label=country)
    
    plt.xlabel(f"{ind_x} (Scaled)")
    plt.ylabel(f"{ind_y} (Scaled)")
    plt.title(f"Correlation between {ind_x} and {ind_y}")
    plt.legend()
    plt.show()
    
    # Visualize the relationship between population growth and CO2 emissions
visualize_correlation(df_years, "Renewable energy consumption (% of total final energy consumption)", "Electric power consumption (kWh per capita)", countries_of_interest)

# Visualize the relationship between population growth and CO2 emissions
visualize_correlation(df_years, "Energy use (kg of oil equivalent per capita)", "Access to electricity (% of population)", countries_of_interest)



df = df_years[df_years['Country Name'].isin(countries_of_interest) & df_years['Indicator Name'].isin(indicators_of_interest)]

# Pivot the data to make it suitable for the heatmap
df_pivot = df.pivot(index='Country Name', columns='Indicator Name', values='2019')

# Create a heatmap of correlations between indicators
sns.heatmap(df_pivot.corr(), cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlations between the Indicators', y=1.04)
plt.show()
