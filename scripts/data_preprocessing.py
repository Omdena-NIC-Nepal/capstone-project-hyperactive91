# %%
import pandas as pd
import numpy as np

# %%
def load_data():
    df=pd.read_csv('../data/dailyclimate.csv')
    return(df)

# %%
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Set the base data path
data_path = Path("./data")

# Function to load CSV safely
def load_csv_safe(filepath, **kwargs):
    try:
        df = pd.read_csv(filepath, **kwargs)
        print(f"✅ Loaded: {filepath.name} with shape {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading {filepath.name}: {e}")
        return pd.DataFrame()

# Function to load shapefile safely
def load_shapefile_safe(filepath):
    try:
        gdf = gpd.read_file(filepath)
        print(f"✅ Loaded shapefile: {filepath.name} with {len(gdf)} records")
        return gdf
    except FileNotFoundError:
        print(f"❌ Shapefile not found: {filepath}")
        return gpd.GeoDataFrame()
    except Exception as e:
        print(f"❌ Error loading shapefile {filepath.name}: {e}")
        return gpd.GeoDataFrame()

# Load datasets
climate_df = load_csv_safe(data_path / 'dailyclimate.csv')
glacier_df = load_csv_safe(data_path / "glaciers_change_in_basins_subbasins_1980_1990_2000_2010.csv")
land_use_df = load_csv_safe(data_path / "land_use_statistics_1967_2010.csv")
agri_df = load_csv_safe(data_path / "nepal_agri_stats_cereal_197980_201314.csv")
local_units_gdf = load_shapefile_safe(data_path / "local_unit_shapefiles" / "local_unit.shp")

# Display basic summaries
datasets = {
    "Climate Data": climate_df,
    "Glacier Change Data": glacier_df,
    "Land Use Statistics": land_use_df,
    "Agricultural Statistics": agri_df,
    "Geospatial Data (Local Units)": local_units_gdf
}

for name, df in datasets.items():
    print(f"\n--- {name} ---")
    if df.empty:
        print("⚠️ Dataset is empty or failed to load.\n")
    else:
        df.info()
        print(df.head(), "\n")

# %%
#climate data preprocessing

# Load data
climate_path = './data/dailyclimate.csv'
climate_df = pd.read_csv(climate_path)

# Convert DATE to datetime safely
climate_df['Date'] = pd.to_datetime(climate_df['Date'], format='%m/%d/%Y', errors='coerce')

# Extract year, month, and day from DATE
climate_df['Year'] = climate_df['Date'].dt.year
climate_df['Month'] = climate_df['Date'].dt.month
climate_df['Day'] = climate_df['Date'].dt.day

# Define Nepal's meteorological seasons
def get_season(month):
    if pd.isna(month):
        return 'Unknown'
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    elif month in [12, 1, 2]:
        return 'Winter'
    return 'Unknown'

climate_df['Season'] = climate_df['Month'].apply(get_season)

# Report missing values
missing = climate_df.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    print("🔍 Missing Values:\n", missing)
else:
    print("✅ No missing values detected.")

# Display summary statistics for numeric columns
print("\n📊 Summary Statistics:")
print(climate_df.describe(include='number'))

# Preview the updated DataFrame
print("\n📋 Updated DataFrame Preview:")
print(climate_df.head())


# %%
#glacier data Preprocessing

import pandas as pd

# Step 1: Load glacier data
glacier_path = './data/glaciers_change_in_basins_subbasins_1980_1990_2000_2010.csv'
glacier_df = pd.read_csv(glacier_path)

# Step 2: Standardize column names
glacier_df.columns = (
    glacier_df.columns
    .str.strip()
    .str.lower()
    .str.replace('~', '', regex=False)
    .str.replace(' ', '_')
    .str.replace(r'\(km2\)', '', regex=True)
    .str.replace(r'\(km3\)', '', regex=True)
    .str.replace(r'\(masl\)', '', regex=True)
    .str.replace(r'[()]', '', regex=True)
)

# Step 3: Rename columns for reshaping
glacier_df.rename(columns={
    'glacier_no._in_1980': 'glacier_count_1980',
    'glacier_no._in_1990': 'glacier_count_1990',
    'glacier_no._in_2000': 'glacier_count_2000',
    'glacier_no._in_2010': 'glacier_count_2010',
    'glacier_area_in_1980': 'glacier_area_1980',
    'glacier_area_1990': 'glacier_area_1990',
    'glacier_area_2000': 'glacier_area_2000',
    'glacier_area_2010': 'glacier_area_2010',
    'estimated_ice_reserved_1980': 'ice_volume_1980',
    'estimated_ice_reserved_1990': 'ice_volume_1990',
    'estimated_ice_reserved2000': 'ice_volume_2000',
    'estimated_ice_reserved2010': 'ice_volume_2010',
    'minimum_elevation_in_1980': 'min_elev_1980',
    'minimum_elevation_in1990': 'min_elev_1990',
    'minimum_elevation_in2000': 'min_elev_2000',
    'minimum_elevation_in2010': 'min_elev_2010'
}, inplace=True)

# Step 4: Reshape from wide to long format
glacier_long = pd.wide_to_long(
    glacier_df,
    stubnames=['glacier_count', 'glacier_area', 'ice_volume', 'min_elev'],
    i=['basin', 'sub-basin'],
    j='year',
    sep='_',
    suffix='(1980|1990|2000|2010)'
).reset_index()

# Step 5: Convert year to integer
glacier_long['year'] = glacier_long['year'].astype(int)

# Step 6: Preview final output
print("✅ Glacier Data (Long Format with Year Column):")
print(glacier_long.head())


# %%
#Glacier change metrics
# Step 1: Pivot glacier_long to compare 1980 vs 2010
pivoted = glacier_long.pivot_table(
    index=['basin', 'sub-basin'],
    columns='year',
    values=['glacier_area', 'ice_volume', 'min_elev']
).reset_index()

# Step 2: Flatten multi-level column headers
pivoted.columns = [
    f"{var}_{int(year)}" if isinstance(year, (int, float)) else var
    for var, year in pivoted.columns.to_flat_index()
]

# Step 3: Calculate absolute changes between 1980 and 2010
pivoted['area_change_1980_2010'] = pivoted['glacier_area_2010'] - pivoted['glacier_area_1980']
pivoted['ice_loss_km3'] = pivoted['ice_volume_1980'] - pivoted['ice_volume_2010']
pivoted['elev_rise_m'] = pivoted['min_elev_2010'] - pivoted['min_elev_1980']

# Step 4: Preview key metrics
print("✅ Glacier Change Indicators (1980–2010):")
print(
    pivoted[
        ['basin', 'sub-basin', 'area_change_1980_2010', 'ice_loss_km3', 'elev_rise_m']
    ].round(2).head()
)


# %%
#Land use data preprocessing

import pandas as pd

# Step 1: Load land use data
land_use_path = './data/land_use_statistics_1967_2010.csv'
land_use_df = pd.read_csv(land_use_path)

# Step 2: Clean column names
land_use_df.columns = (
    land_use_df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('%', '', regex=False)  # Remove '%' sign
)

# Step 3: Display structure and preview
print("📄 Land Use Data Info:")
land_use_df.info()
print("\n🔍 Land Use Data Preview:")
print(land_use_df.head())

# Step 4: Melt to long format
land_use_long = land_use_df.melt(
    id_vars='land_use_type',
    var_name='year',
    value_name='percentage'
)

# Step 5: Extract numeric year from column names
land_use_long['year'] = pd.to_numeric(
    land_use_long['year'].str.extract(r'(\d{4})')[0],
    errors='coerce'
)

# Step 6: Standardize land use type names
land_use_long['land_use_type'] = (
    land_use_long['land_use_type']
    .str.strip()
    .str.lower()
    .str.replace(r'[^a-z0-9_]+', '', regex=True)  # Remove non-alphanum (e.g., '*')
)

# Step 7: Drop missing values
land_use_long.dropna(subset=['year', 'percentage'], inplace=True)

# Step 8: Preview final dataset
print("\n📊 Tidy Land Use Data (Long Format):")
print(land_use_long.head())


# %%
#land use trends
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.lineplot(data=land_use_long, x='year', y='percentage', hue='land_use_type', marker='o')
plt.title("Land Use Change in Nepal (1967–2010)")
plt.xlabel("Year")
plt.ylabel("Percentage of Land")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
#Cereal yeild data preprocessing
import pandas as pd

# Load data
agri_path = "./data/nepal_agri_stats_cereal_197980_201314.csv"
agri_df = pd.read_csv(agri_path)

# Overview
print("📄 Cereal Yield Data Info:")
agri_df.info()
print("\n🔍 Preview:")
print(agri_df.head())

# Step 1: Clean column names
agri_df.columns = agri_df.columns.str.strip().str.upper()

# Step 2: Identify yield columns (those with '_Y_')
yield_cols = [col for col in agri_df.columns if '_Y_' in col]

# Step 3: Subset dataframe
yield_df = agri_df[['DISTRICT_NAME'] + yield_cols].copy()

# Step 4: Melt to long format
yield_long = yield_df.melt(
    id_vars='DISTRICT_NAME',
    var_name='CROP_FY',
    value_name='YIELD'
)

# Step 5: Extract crop and fiscal year
extracted = yield_long['CROP_FY'].str.extract(r'([A-Z]+)_Y_(\d{6})')
yield_long['CROP'] = extracted[0].str.title()  # Capitalize crop names
yield_long['FY'] = extracted[1]

# Step 6: Format FY (e.g., "197980" → "1979/80")
yield_long['FY'] = yield_long['FY'].apply(lambda x: f"{x[:4]}/{x[4:]}" if pd.notna(x) else None)

# Step 7: Drop missing values
yield_long.dropna(subset=['DISTRICT_NAME', 'CROP', 'FY', 'YIELD'], inplace=True)

# Step 8: Final tidy DataFrame
yield_long = yield_long[['DISTRICT_NAME', 'CROP', 'FY', 'YIELD']]

# Preview tidy result
print("\n✅ Tidy Yield Data Preview:")
print(yield_long.head())


# %%
#yeild trend of crops

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.lineplot(data=yield_long, x='FY', y='YIELD', hue='CROP', estimator='mean', marker='o')
plt.title("Average Cereal Yield Over Time by Crop")
plt.xlabel("Fiscal Year")
plt.ylabel("Yield (kg/ha)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
#geospatial preprocessing

import geopandas as gpd
from pathlib import Path

# Step 1: Load the shapefile
data_path = Path("./data") / "local_unit_shapefiles" / "local_unit.shp"
gdf = gpd.read_file(data_path)

# Step 2: Normalize column names
gdf.columns = gdf.columns.str.strip().str.upper()

# Step 3: Standardize district names
gdf['DISTRICT_NAME'] = gdf['DISTRICT'].str.strip().str.lower()

# ✅ Step 4: Ensure 'GEOMETRY' is set as active geometry
if 'GEOMETRY' in gdf.columns:
    gdf = gdf.set_geometry('GEOMETRY')

# Step 5: Dissolve polygons to one per district
district_gdf = gdf.dissolve(by='DISTRICT_NAME', as_index=False)

# Step 6: Project to UTM Zone 45N (EPSG:32645) for spatial accuracy
district_gdf_proj = district_gdf.to_crs(epsg=32645)

# Step 7: Compute centroids (in projected CRS)
district_gdf_proj['CENTROID'] = district_gdf_proj.geometry.centroid

# Step 8: Convert centroids to WGS84 for lat/lon extraction
centroids_wgs84 = district_gdf_proj.set_geometry('CENTROID').to_crs(epsg=4326)
centroids_wgs84['CENTROID_LAT'] = centroids_wgs84.geometry.y
centroids_wgs84['CENTROID_LON'] = centroids_wgs84.geometry.x

# Step 9: Final output
district_centroids = centroids_wgs84[['DISTRICT_NAME', 'CENTROID_LAT', 'CENTROID_LON']]

# Step 10: Preview
print("✅ District Centroid Coordinates:")
print(district_centroids.head())


# %%



