'''This script is for applying functional rules
to each hazard dataset in order to classify the raw data into actionable risk factors'''

import numpy as np

def apply_temperature_rules(df, column, threshold, risk_column, invert=False):
    if invert:
        df[risk_column] = np.maximum(threshold - df[column], 0)
    else:
        df[risk_column] = np.maximum(df[column] - threshold, 0)


def wind_risk_scaled(speed_mps):
    if speed_mps < 13.4: # Below 30 mph
        return 0
    elif speed_mps <= 20.1: # between 30 and 45 mph
        return (speed_mps - 13.4) / (20.1 - 13.4) # Scale to 0 - 1
    else:
        return 1 + (speed_mps - 20.1) / (25 - 20.1) # Scale beyond 1

def apply_extreme_wind_rules(df):
    df['wind_spd_risk'] = df['p99'].apply(wind_risk_scaled)
