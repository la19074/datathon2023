import numpy as np
import pandas as pd

def toNaN_in_range(df, col, range):
    df.loc[df[col].isin(range), col] = np.NaN

def toNaN_outside_range(df, col, range):
    df.loc[~df[col].isin(range), col] = np.NaN

def print_percent_NaN(df):
    print(df.isnull().sum() * 100 / len(df))

def convert_severity(df_c):
    df_c.loc[df_c["casualty_severity"].isin([1, 2]), "casualty_severity"] = 1
    df_c.loc[df_c["casualty_severity"].isin([3]), "casualty_severity"] = 0

def process_c(df_c):
    toNaN_outside_range(df_c, "casualty_class", [*range(1, 4)])
    toNaN_outside_range(df_c, "sex_of_casualty", [*range(1, 3)])
    toNaN_in_range(df_c, "age_of_casualty", [-1])
    toNaN_outside_range(df_c, "age_band_of_casualty", [*range(1, 12)])
    toNaN_outside_range(df_c, "casualty_severity", [*range(1, 4)])
    convert_severity(df_c)
    toNaN_outside_range(df_c, "pedestrian_location", [*range(0, 10)])
    toNaN_outside_range(df_c, "pedestrian_movement", [*range(0, 9)])
    toNaN_outside_range(df_c, "car_passenger", [*range(0, 3)])
    toNaN_outside_range(df_c, "bus_or_coach_passenger", [*range(0, 5)])
    toNaN_outside_range(df_c, "pedestrian_road_maintenance_worker", [*range(0, 4)])
    toNaN_outside_range(df_c, "casualty_type", [*range(0, 6)] + [*range(8, 11)] + [*range(16, 24)] + [90, 97, 98] + [*range(103, 106)] + [*range(108, 111)] + [113])
    toNaN_outside_range(df_c, "casualty_imd_decile", [*range(1, 11)])
    toNaN_outside_range(df_c, "casualty_home_area_type", [*range(1, 4)])

def process_v(df_v):
    toNaN_outside_range(df_v, "vehicle_type", [*range(1, 6)] + [*range(8, 12)] + [*range(16, 24)] + [97, 98] + [*range(103, 107)] + [*range(108, 111)] + [113])
    toNaN_outside_range(df_v, "towing_and_articulation", [*range(0, 6)])
    toNaN_outside_range(df_v, "vehicle_manoeuvre", [*range(1, 19)])
    toNaN_outside_range(df_v, "vehicle_direction_from", [*range(0, 9)])
    toNaN_outside_range(df_v, "vehicle_direction_to", [*range(0, 9)])
    toNaN_outside_range(df_v, "vehicle_location_restricted_lane", [*range(0, 11)])
    toNaN_outside_range(df_v, "junction_location", [*range(0, 9)])
    toNaN_outside_range(df_v, "skidding_and_overturning", [*range(0, 6)])
    toNaN_outside_range(df_v, "hit_object_in_carriageway", [*range(0, 13)])
    toNaN_outside_range(df_v, "vehicle_leaving_carriageway", [*range(0, 9)])
    toNaN_outside_range(df_v, "hit_object_off_carriageway", [*range(0, 12)])
    toNaN_outside_range(df_v, "first_point_of_impact", [*range(0, 5)])
    toNaN_outside_range(df_v, "vehicle_left_hand_drive", [*range(1, 3)])
    toNaN_outside_range(df_v, "journey_purpose_of_driver", [*range(1, 6)])
    toNaN_outside_range(df_v, "sex_of_driver", [*range(1, 3)])
    toNaN_in_range(df_v, "age_of_driver", [-1])
    toNaN_outside_range(df_v, "age_band_of_driver", [*range(1, 12)])
    toNaN_in_range(df_v, "engine_capacity_cc", [-1])
    toNaN_outside_range(df_v, "propulsion_code", [*range(1, 13)])
    toNaN_in_range(df_v, "age_of_vehicle", [-1])
    toNaN_in_range(df_v, "generic_make_model", [-1])
    toNaN_outside_range(df_v, "driver_imd_decile", [*range(1, 11)])
    toNaN_outside_range(df_v, "driver_home_area_type", [*range(1, 4)])

def process_data(df_c, df_v):
    process_c(df_c)
    process_v(df_v)

    percentage = 0.79
    df_c.dropna(axis = 1, thresh = int(percentage * df_c.shape[0] + 1), inplace = True) # Drop all columns with more than 21% NaN
    df_v.dropna(axis = 1, thresh = int(percentage * df_v.shape[0] + 1), inplace = True)
    df_c.dropna(inplace = True) # Drop all rows with NaN
    df_v.dropna(inplace = True)
    df_c.drop_duplicates(inplace = True) # Drop all duplicate rows
    df_v.drop_duplicates(inplace = True)

if __name__ == "__main__":
    df_c = pd.read_csv("../casualty_train.csv") # Load csv files into dataframes.
    df_v = pd.read_csv("../vehicle_train.csv")
    print(df_c.info())
    print(df_v.info())
    process_data(df_c, df_v)
    print(df_c.info())
    print(df_v.info())
