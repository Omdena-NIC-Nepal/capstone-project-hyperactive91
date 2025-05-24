import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_yearly_temp(df):
    """
    plot yearly average temperature
    """
    df['Date'] = pd.to_datetime(df['Date'])  #  convert to datetime first

    # separate month as period
    df['Month'] = df['Date'].dt.to_period('M')

    monthly_temp = df.groupby('Month')['Temp_2m'].mean()
    monthly_temp.index = monthly_temp.index.to_timestamp()
    fig, ax = plt.subplots(figsize = (10, 6))
    plt.plot(monthly_temp.index, monthly_temp.values, color='red')
    plt.title('Monthly Average Temperature Trend (1981-2020)')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    return(fig)


def plot_yearly_precip(df):
    """
    plot yearly average precipitation
    """
    monthly_precp = df.groupby('Month')['Precip'].mean()
    monthly_precp.index = monthly_precp.index.to_timestamp()
    fig, ax = plt.subplots(figsize = (10, 6))
    plt.plot(monthly_precp.index, monthly_precp.values, color='blue')
    plt.title('Monthly Average Precipitation Trend (1981-2020)')
    plt.xlabel('Year')
    plt.ylabel('Precipitaion in mm')
    plt.grid(True)
    return(fig)
    


def plot_boxplot(df):
    """
    box plot
    """
    df1=df.drop('Unnamed: 0',axis=1)
    df1=df1.drop(['Date','District','Month'],axis=1)
    fig, ax = plt.subplots(figsize = (10, 6))
    sns.boxplot(df1,label=True,orient='h')
    return fig

def plot_avgtemp_bydistrict(df):
    """
    District-Wise Average Temperature
    """
    # District-Wise Average Temperature
    district_avg_temp = df.groupby('District')['Temp_2m'].mean().sort_values()

    fig, ax = plt.subplots(figsize = (14, 10))
    district_avg_temp.plot(kind='barh', color='red')
    plt.title('Average Temperature by District')
    plt.xlabel('Temperature (°C)')
    return fig

def plot_avgprecip_bydistrict(df):
    """
    District-Wise Average Temperature
    """
    # District-Wise Average Temperature
    district_avg_temp = df.groupby('District')['Precip'].mean().sort_values()

    fig, ax = plt.subplots(figsize = (14, 10))
    district_avg_temp.plot(kind='barh', color='blue')
    plt.title('Average Precipitation by District')
    plt.xlabel('rainfall in mm')
    return fig

def plot_actual_vs_predicted(y_test, ypred):
    """
    plot model regression 
    """
    fig, ax = plt.subplots(figsize = (10, 6))
    ax.scatter(y_test, ypred, alpha=0.7)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    ax.set_xlabel("Actual Temperature in Celcius")
    ax.set_ylabel("Predicted Temperature in Celcius")
    ax.set_title("Actual vs Predicted Temperatures")
    
    return fig

def plot_by_seasons(df):
    # Parse date and extract month
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['MONTH'] = df['Date'].dt.month

    #Assign season if not already present
    def assign_season(month):
        if pd.isna(month):
            return None
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        elif month in [12, 1, 2]:
            return 'Winter'
        return 'Unknown'

    df['Season'] = df['MONTH'].apply(assign_season)

    #Filter clean rows for plotting
    df = df.dropna(subset=['Season', 'Temp_2m'])

    # Create violin plot
    fig, ax = plt.subplots(figsize = (10, 6))
    sns.violinplot(
        data=df,
        x='Season',
        y='Temp_2m',
        hue='Season',           
        palette='viridis',
        dodge=False,            
        legend=False            
    )

    plt.title('Mean Temperature Season-wise', fontsize=14)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig
    