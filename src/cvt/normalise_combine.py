from src.cvt.functional_rules import tfn_extreme_heat

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import geopandas as gpd

apply_temperature_risk(tfn_extreme_heat, 'tasmax_s_f_actual', 30, 'tasmax_risk', invert=False)  # Extreme Heat
apply_temperature_risk(tfn_extreme_cold, 'tasmin_w_f_actual', 0, 'tasmin_risk', invert=True)   # Extreme Cold

def normalise_combine(gdf, weights, variables, risk_column):
    scaler = MinMaxScaler()
    normalised = scaler.fit_transform(gdf[variables])

    weight_array = np.array(weights) # Convert to array
    gdf[risk_column] = normalised @ weight_array # Compute weighted sum
    return gpd.GeoDataFrame(gdf, geometry=gdf.geometry, crs=gdf.crs)

def spatial_smooth_zero_grids(gdf, variables):
    neighbours = gpd.sjoin(gdf, gdf, how='left',predicate='touches') # Find neighbouring grids

    # Calculate the average value of the neighbouring grids
    neighbours_avg = neighbours.groupby(neighbours.index)[[f"{var}_right" for var in variables]].mean()

    zero_condition = (gdf[variables] == 0).all(axis=1)

    for var in variables:
        gdf.loc[zero_condition, var] = gdf.loc[zero_condition].index.map(neighbours_avg[f"{var}_right"])