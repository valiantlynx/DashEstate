import pandas as pd
import numpy as np

# --------------------------
# Load the data
# --------------------------
houses = pd.read_json("data/houses.jsonl", lines=True)
districts = pd.read_json("data/districts.jsonl", lines=True)
schools = pd.read_json("data/schools.jsonl", lines=True)
agents = pd.read_json("data/agents.jsonl", lines=True)

# --------------------------
# Merge DataFrames
# --------------------------
houses = houses.merge(districts, left_on='district_id', right_on='id', suffixes=('', '_district'), how='left')
if 'id_district' in houses.columns:
    houses.drop('id_district', axis=1, inplace=True)

houses = houses.merge(schools, left_on='school_id', right_on='id', suffixes=('', '_school'), how='left')
if 'id_school' in houses.columns:
    houses.drop('id_school', axis=1, inplace=True)

houses = houses.merge(agents, on='agent_id', how='left', suffixes=('', '_agent'))

# Convert 'rooms' to numeric
houses['rooms'] = houses['rooms'].str.extract('(\d+)').astype(float)

# --------------------------
# Basic Data Checks
# --------------------------
print("\n--- BASIC DATA OVERVIEW ---")
print("Number of houses:", len(houses))
print("Columns:", houses.columns.tolist())
print("Sample rows:\n", houses.head())

# --------------------------
# Check for obvious anomalies
# --------------------------

# 1. Negative 'remodeled' year
negative_remodeled = houses[houses['remodeled'] < 0]
if not negative_remodeled.empty:
    print("\nFound houses with negative remodeled year (possible tampering):")
    print(negative_remodeled[['price', 'year', 'remodeled', 'agent_id', 'district_id', 'school_id']].head(10))
    print("Count of such houses:", len(negative_remodeled))

# 2. Zero or unusual 'rooms' values
missing_rooms = houses[houses['rooms'].isna() | (houses['rooms'] == 0)]
if not missing_rooms.empty:
    print("\nHouses with missing or zero rooms (unusual):")
    print(missing_rooms[['price', 'rooms', 'agent_id', 'district_id', 'school_id']].head(10))
    print("Count of such houses:", len(missing_rooms))

# 3. Negative or zero 'size'
invalid_size = houses[houses['size'] <= 0]
if not invalid_size.empty:
    print("\nHouses with non-positive size (impossible):")
    print(invalid_size[['price', 'size', 'agent_id', 'district_id', 'school_id']].head(10))
    print("Count of such houses:", len(invalid_size))

# 4. Price per square meter outliers
houses['price_per_m2'] = houses['price'] / houses['size']
mean_ppm = houses['price_per_m2'].mean()
std_ppm = houses['price_per_m2'].std()
outliers_price = houses[houses['price_per_m2'] > mean_ppm + 3 * std_ppm]
if not outliers_price.empty:
    print("\nHouses with unusually high price per square meter (potential outliers):")
    print(outliers_price[['price', 'size', 'price_per_m2', 'agent_id', 'district_id']].head(10))
    print("Count of such houses:", len(outliers_price))

# 5. Low condition but very high price
suspicious_condition = houses[(houses['condition_rating'] < 3) & (houses['price'] > houses['price'].quantile(0.9))]
if not suspicious_condition.empty:
    print("\nHouses with low condition rating but very high price (suspicious):")
    print(suspicious_condition[['price', 'condition_rating', 'agent_id', 'district_id', 'school_id']].head(10))
    print("Count of such houses:", len(suspicious_condition))

# --------------------------
# District and School Analysis
# --------------------------
print("\n--- DISTRICT AND SCHOOL ANALYSIS ---")
district_group = houses.groupby('district_id').agg(
    avg_price=('price', 'mean'),
    avg_crime_rating=('crime_rating', 'mean'),
    avg_transport_rating=('public_transport_rating', 'mean'),
    count=('price', 'count')
).reset_index()
print("\nDistrict-level summary:")
print(district_group)

school_group = houses.groupby('school_id').agg(
    avg_price=('price', 'mean'),
    avg_school_rating=('rating', 'mean'),
    avg_capacity=('capacity', 'mean'),
    count=('price', 'count')
).reset_index()
print("\nSchool-level summary:")
print(school_group)

# Check for contradictory patterns:
# For example, high crime rating districts with very high avg_price
high_crime_districts = district_group[district_group['avg_crime_rating'] > 3]
if not high_crime_districts.empty:
    print("\nDistricts with high crime rating but possibly high prices:")
    print(high_crime_districts.sort_values('avg_price', ascending=False))

# Schools with very low rating but associated with high priced houses
low_rating_schools = school_group[school_group['avg_school_rating'] < 2]
if not low_rating_schools.empty:
    print("\nSchools with low ratings but high avg_price:")
    low_rating_schools = low_rating_schools.sort_values('avg_price', ascending=False)
    print(low_rating_schools)

# --------------------------
# Agent Analysis
# --------------------------
print("\n--- AGENT ANALYSIS ---")
agent_group = houses.groupby(['agent_id', 'name']).agg(
    avg_price=('price', 'mean'),
    avg_condition=('condition_rating', 'mean'),
    avg_sold_speed=('days_on_marked', 'mean'),
    count=('price', 'count')
).reset_index().sort_values('avg_price', ascending=False)
print("\nAgent-level summary:")
print(agent_group.head(20))

# Check if certain agents dominate the anomalous subsets:
if not negative_remodeled.empty:
    print("\nAgents with negative remodeled houses:")
    print(negative_remodeled['agent_id'].value_counts().head(10))

if not outliers_price.empty:
    print("\nAgents with unusual price per square meter outliers:")
    print(outliers_price['agent_id'].value_counts().head(10))

if not suspicious_condition.empty:
    print("\nAgents handling low-condition high-price houses:")
    print(suspicious_condition['agent_id'].value_counts().head(10))

# --------------------------
# Correlation Checks
# --------------------------
print("\n--- CORRELATION ANALYSIS ---")
numeric_cols = ['price', 'condition_rating', 'days_on_marked', 'external_storage_m2', 'kitchens', 'lot_w', 'size',
                'storage_rating', 'sun_factor', 'year', 'crime_rating', 'public_transport_rating', 'rating', 'capacity']
corr = houses[numeric_cols].corr()
print("\nCorrelation matrix between numeric variables:")
print(corr)

unusual_corr = corr[(corr.abs() > 0.7) & (corr != 1.0)]
print("\nUnusually high correlations (magnitude > 0.7, excluding perfect self-correlations):")
print(unusual_corr)

# If there's a strong correlation that shouldn't exist, investigate:
# For instance, if 'lot_w' and 'size' strongly correlate with price. Check any house that breaks that pattern.
if 'lot_w' in houses.columns and 'size' in houses.columns:
    large_lot_small_size = houses[(houses['lot_w'] > houses['lot_w'].quantile(0.9)) & (houses['size'] < houses['size'].quantile(0.1))]
    if not large_lot_small_size.empty:
        print("\nHouses with large lot width but very small size (contradiction):")
        print(large_lot_small_size[['price', 'lot_w', 'size', 'agent_id', 'district_id', 'school_id']].head(10))

# --------------------------
# Month-by-Month Analysis
# --------------------------
print("\n--- SEASONAL / MONTH ANALYSIS ---")
month_group = houses.groupby('sold_in_month')['price'].mean().sort_values()
print("\nAverage price by sold_in_month:")
print(month_group)

# Identify months with abnormally high prices:
threshold_price = month_group.mean() + 2 * month_group.std()
expensive_months = month_group[month_group > threshold_price]
if not expensive_months.empty:
    print("\nMonths with abnormally high average price:")
    print(expensive_months)

# Focus on a suspicious month (e.g. highest price month)
if not expensive_months.empty:
    suspicious_month = expensive_months.index[-1]
    print(f"\nDetailed look at houses sold in {suspicious_month}:")
    suspicious_month_houses = houses[houses['sold_in_month'] == suspicious_month]
    print(suspicious_month_houses[['price', 'condition_rating', 'lot_w', 'size', 'agent_id', 'district_id', 'school_id']].head(10))

# --------------------------
# Additional Outlier Detection
# --------------------------
# Let's do a quick z-score analysis on a few numeric columns to see if certain houses are consistently outliers.
def zscore(series):
    return (series - series.mean()) / series.std()

houses['z_price'] = zscore(houses['price'])
houses['z_size'] = zscore(houses['size'])
houses['z_lot_w'] = zscore(houses['lot_w'])
houses['z_kitchens'] = zscore(houses['kitchens'])

extreme_outliers = houses[(houses['z_price'].abs() > 3) | (houses['z_size'].abs() > 3) | (houses['z_lot_w'].abs() > 3) | (houses['z_kitchens'].abs() > 3)]
if not extreme_outliers.empty:
    print("\nHouses that are extreme outliers in at least one variable (z-score > 3):")
    print(extreme_outliers[['price', 'size', 'lot_w', 'kitchens', 'z_price', 'z_size', 'z_lot_w', 'z_kitchens']].head(10))
    print("Count of such outliers:", len(extreme_outliers))

# --------------------------
# Summary of Findings
# --------------------------
# At this point, we've:
# - Identified houses with negative remodel years.
# - Found houses with no rooms or impossible sizes.
# - Found price per square meter outliers.
# - Checked for suspicious conditions vs. price.
# - Explored district/school anomalies.
# - Investigated agents involved in suspicious listings.
# - Checked for unusual month-based pricing patterns.
# - Performed basic correlation and outlier analysis.

print("\n--- ANALYSIS COMPLETE ---")
print("Use the printed summaries and outliers to identify hidden hacks, anomalies, and suspicious patterns.")
