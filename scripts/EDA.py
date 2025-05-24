# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df=pd.read_csv('./data/dailyclimate.csv')
df.info()


# %%
df.isnull().sum()

# %%
df.describe()

# %%
df['Date'] = pd.to_datetime(df['Date'])  #  convert to datetime first

# separate month as period
df['Month'] = df['Date'].dt.to_period('M')

monthly_temp = df.groupby('Month')['Temp_2m'].mean()
monthly_temp.index = monthly_temp.index.to_timestamp()
plt.figure(figsize=(15,5))
plt.plot(monthly_temp.index, monthly_temp.values, color='red')
plt.title('Monthly Average Temperature Trend (1981-2020)')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()


# %%
df1=df.drop('Unnamed: 0',axis=1)
df1=df1.drop(['Date','District','Month'],axis=1)
plt.subplots(figsize=(14,8))
sns.boxplot(df1,label=True,orient='h')

# %%
# District-Wise Average Temperature
district_avg_temp = df.groupby('District')['Temp_2m'].mean().sort_values()

plt.figure(figsize=(14,8))
district_avg_temp.plot(kind='barh', color='blue')
plt.title('Average Temperature by District')
plt.xlabel('Temperature (°C)')
plt.show()

# %%


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
plt.figure(figsize=(8, 6))
sns.violinplot(
    data=df,
    x='Season',
    y='Temp_2m',
    hue='Season',           # Color by season
    palette='viridis',
    dodge=False,            # Overlay not split
    legend=False            # Hue = x, no extra legend needed
)

plt.title('Distribution of Mean Temperature by Season', fontsize=14)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %%



