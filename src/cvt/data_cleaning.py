'''This script will take raw input data and clean it ready for model input'''

import pandas as pd
import geopandas as gpd

def clip_to_boundary(gdf, boundary):
    '''Takes a GeoDataFrame and a boundary and returns a GeoDataFrame'''
    boundary = boundary.to_crs(gdf.crs) # Match CRS
    gdf_boundary = gpd.clip(gdf, boundary) # Clip GDF to boundary
    return gdf_boundary

def standard_cleaning(gdf, columns_to_keep):
    '''Takes a GeoDataFrame, performs standard cleaning operations, and returns a GeoDataFrame'''
    gdf = gdf.drop_duplicates() # Drop duplicate rows
    gdf = gdf[columns_to_keep] # Filter for relevant columns

    # Remove rows with empty or null geometry
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.notnull()]


