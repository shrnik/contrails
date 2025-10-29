import pandas as pd
import numpy as np

def clean_numeric(s: pd.Series) -> pd.Series:
    # convert to string, strip whitespace
    s = s.astype(str).str.strip()
    # make empty strings into NaN
    s = s.replace({'': np.nan})
    # remove commas and any stray spaces
    s = s.str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
    # finally, coerce to numeric
    return pd.to_numeric(s, errors='coerce')

def upsample_aircraft(group):
    """
    Upsample a single aircraft's data to 1-second intervals using linear interpolation.
    Adds an 'isUpsampled' column to indicate interpolated vs original data.
    """
    # Sort by time
    group = group.sort_values('time').reset_index(drop=True)
    
    # Mark original data as not upsampled
    group['isUpsampled'] = False
    
    # Get time range
    start_time = group['time'].min()
    end_time = group['time'].max()
    
    # Create 1-second intervals
    time_range = pd.date_range(start=start_time, end=end_time, freq='1S')
    
    # Create new dataframe with all seconds
    new_df = pd.DataFrame({'time': time_range})
    
    # Merge with original data
    merged = new_df.merge(group, on='time', how='left')
    
    # Convert numeric columns to proper float dtype BEFORE interpolation
    numeric_cols_to_interpolate = ['lon', 'lat', 'alt', 'alt_gnss', 'heading', 
                                     'alt_meters', 'alt_gnss_meters', 'distance_m']
    
    for col in numeric_cols_to_interpolate:
        if col in merged.columns:
            # Convert to numeric, forcing errors to NaN
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            # Interpolate
            merged[col] = merged[col].interpolate(method='linear', limit_area='inside')
    
    # Fill isUpsampled: True for interpolated rows, False for original
    merged['isUpsampled'] = merged['isUpsampled'].fillna(True)

    # Forward fill categorical/string columns (like ident, aircraft_type, etc.)
    categorical_cols = ['source_id', 'source', 'transponder_id', 'orig', 'dest', 
                       'ident', 'aircraft_type', 'clock_datetime']
    
    for col in categorical_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(method='ffill')

    return merged

def get_upsampled_df_for_day(df: pd.DataFrame, from_date: str, to_date: str) -> pd.DataFrame:
    """Load CSV, filter for date, and upsample."""
    df.columns = df.columns.str.strip()

    # Clean and convert numeric columns
    for col in ['alt_gnss_meters', 'distance_m']:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
    
    df = df.dropna(subset=['alt_gnss_meters'])
    df['time'] = pd.to_datetime(df['time'])
    from_dt = pd.to_datetime(from_date)
    to_dt = pd.to_datetime(to_date)
    df = df[(df['time'] >= from_dt) & (df['time'] < to_dt)]

        # convert to float
    df['alt_gnss_meters'] = df['alt_gnss_meters'].astype(float)
    df['distance_m'] = df['distance_m'].astype(float)

    # minimum altitude filter (8000 ft = 2438.4 m)
    df = df[(df['alt_gnss_meters'] > 2438.4)]

    print("Upsampling all aircraft...")
    # filter for 3 pm utc to 4 pm utc
    print(f"Processing {df['ident'].nunique()} unique aircraft...\n")

    # Group by ident and apply upsampling
    upsampled_groups = []
    for ident, group in df.groupby('ident'):
        upsampled = upsample_aircraft(group)
        upsampled_groups.append(upsampled)
        if len(upsampled_groups) % 10 == 0:
            print(f"Processed {len(upsampled_groups)} aircraft...")

    # Combine all upsampled data
    df_upsampled = pd.concat(upsampled_groups, ignore_index=True)
    df_upsampled = df_upsampled.sort_values(['ident', 'time']).reset_index(drop=True)
    df_upsampled= df_upsampled[df_upsampled["distance_m"] < 50000]

    # Check for NaN values in lat, lon, alt_gnss_meters
    nan_rows = df_upsampled[df_upsampled[['lat', 'lon', 'alt_gnss_meters']].isna().any(axis=1)]

    # Check for string values in lat, lon, alt_gnss_meters
    str_rows = df_upsampled[
        df_upsampled[['lat', 'lon', 'alt_gnss_meters']].applymap(lambda x: isinstance(x, str)).any(axis=1)
    ]

    if not nan_rows.empty:
        print(f"Warning: Found {len(nan_rows)} rows with NaN values in lat, lon, or alt_gnss_meters.")
        print(nan_rows[['ident', 'time', 'lat', 'lon', 'alt_gnss_meters']])

    if not str_rows.empty:
        print(f"Warning: Found {len(str_rows)} rows with string values in lat, lon, or alt_gnss_meters.")
        print(str_rows[['ident', 'time', 'lat', 'lon', 'alt_gnss_meters']])
    if not nan_rows.empty:
        print(f"Warning: Found {len(nan_rows)} rows with NaN values in lat, lon, or alt_gnss_meters.")
        print(nan_rows[['ident', 'time', 'lat', 'lon', 'alt_gnss_meters']])
    
    return df_upsampled