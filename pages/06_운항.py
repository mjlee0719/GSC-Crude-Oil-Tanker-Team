import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
SEED=123
from sklearn.metrics import mean_absolute_error
import geohash2
import optuna
import streamlit as st
import streamlit.components.v1 as components
import math
import openpyxl
import pydeck as pdk


st.markdown("""
    <style>
    .stSlider [data-baseweb=slider]{
        width: 25%;
    }
    </style>
    """,unsafe_allow_html=True)


####################데이터 로드####################

df = pd.read_excel('.cache/files/Template.xlsx', sheet_name='Raw')

##################################################

####################UTC 변환####################
df['UTC'] = pd.to_datetime(df['UTC'], format="%d/%m/%Y %I:%M:%S %p")
df['year']=df['UTC'].dt.year
df['month']=df['UTC'].dt.month
df['day']=df['UTC'].dt.day
df['hour']=df['UTC'].dt.hour
#################################################

####################Distance to Next Port + Dist OFB + Speed + Slip + RPM NaN값 -> 0으로 변환#####################
df['Dist. to Next Port'] = df['Dist. to Next Port'].fillna(0)
df['Dist. to St.John'] = df['Dist. to St.John'].fillna(0)
df['Dist. to St.John'] = pd.to_numeric(df['Dist. to St.John'], errors='coerce')
df['Dist. to OFB'] = df['Dist. to OFB'].fillna(0)
df.loc[(df['Dist. to OFB']=='-'),'Dist. to OFB']=0
df['Dist. to OFB'] = df['Dist. to OFB'].apply(lambda x: float(x) if isinstance(x, str) else x)
df['Spd. Present'] = df['Spd. Present'].fillna(0)
df['Spd. Daily Avg.'] = df['Spd. Daily Avg.'].fillna(0)
df['Slip'] = df['Slip'].fillna(0)
df['RPM'] = df['RPM'].fillna(0)
##################################################################################################################

####################HSFO Total + LSFO Total 열 추가#####################
df['Total_FO'] = df['Total HSFO'] + df['Total LSFO'] 
df['Total_FO_ROB']=df['HSFO ROB'] + df['LSFO ROB'] 
########################################################################

# ####################Steaming Time Str -> Float 변환####################
# def time_to_float_hour(time_str):
#     hours, minutes = map(int, time_str.split(":"))
#     return hours + minutes / 60
# df['Steaming Time'] = df['Steaming Time'].apply(time_to_float_hour)
# ########################################################################

####################Lat, Long -> Decimal Degree 변환####################
df['N/S']=df['N/S'].apply(lambda x: 1 if x=='N' else -1)
df['E/W']=df['E/W'].apply(lambda x: 1 if x=='E' else -1)
df['LatDD']=df['N/S']*(df['Lat1']+df['Lat2']/60)
df['LongDD']=df['E/W']*(df['Long1']+df['Long2']/60)
########################################################################

####################선박명 변환####################
df['Vessel_Full']=df['Vessel']

def vessel_name(x) :
    if x=='G.Dream':
        return 1
    elif x== 'G.Hope':
        return 2
    elif x== 'G.Future':
        return 3
    elif x== 'Universal Victor':
        return 4
    elif x== 'Universal Creator':
        return 5
    elif x== 'Universal Honor':
        return 6
    elif x== 'Universal Glory':
        return 7
    elif x== 'V. Harmony':
        return 8
    elif x== 'V. Prosperity':
        return 9
    elif x== 'V. Glory':
        return 10
    elif x== 'V. Advance':
        return 11
    elif x== 'SM Venus1':
        return 12
    elif x== 'SM Venus2':
        return 13
    else:
        return 14
df['Vessel']=df['Vessel'].apply(vessel_name)
df['Vessel'].value_counts()
##########################################################

####################Ballast/Laden 변환####################
def status(x) :
    if x=='B' or x=='HL':
        return 0
    else :
        return 2
df['Status']=df['Status'].apply(status)
##########################################################

#########################Sea 변환##########################
def sea(x) :
    if x=='Smooth':
        return 1
    elif x=='Slight':
        return 2
    elif x=='Moderate':
        return 3
    elif x=='Rough':
        return 4
    elif x=='Very Rough':
        return 5
    elif x=='High':
        return 6
    elif x=='Very High':
        return 7
    elif x=='Phenomenal':
        return 8
df['Sea']=df['Sea'].apply(sea)
##########################################################

########################ETA 변환##########################
df['ETA OFB']=pd.to_datetime(df['ETA OFB'], format="%d/%m/%Y %I:%M:%S %p")
df['ETA Ordered']=pd.to_datetime(df['ETA Ordered'], format="%d/%m/%Y %I:%M:%S %p")
df['ETA Expected']=pd.to_datetime(df['ETA Expected'], format="%d/%m/%Y %I:%M:%S %p",errors='coerce')
df['ETA Present']=pd.to_datetime(df['ETA Present'], format="%d/%m/%Y %I:%M:%S %p",errors='coerce')
df['ETA Daily Avg.']=pd.to_datetime(df['ETA Daily Avg.'], format="%d/%m/%Y %I:%M:%S %p",errors='coerce')
df['ETA Total Avg.']=pd.to_datetime(df['ETA Total Avg.'], format="%d/%m/%Y %I:%M:%S %p",errors='coerce')
df['ETA St. John']=pd.to_datetime(df['ETA St. John'], format="%d/%m/%Y %I:%M:%S %p",errors='coerce')
##########################################################

######################## Geohash 변환 ##########################
def coordinates_to_geohash(latitude, longitude, precision=2):
    return geohash2.encode(latitude, longitude, precision)
df['Geohash']=df.apply(lambda x: coordinates_to_geohash(x.LatDD, x.LongDD), axis=1)
df['Geohash_num'] = pd.factorize(df['Geohash'])[0]
################################################################

########################## Voyage 진행률 ########################
df['Voyage Completion Ratio'] = df['Dist. Total']/(df['Dist. to Next Port']+df['Dist. Total'])
################################################################


##################################################################################################################
################################################운항 현황 트래킹###################################################
##################################################################################################################

tab1, tab2, tab3 = st.tabs(["선박별 운항 현황", "선박별 RPM & FOC", "운항 시뮬레이션"])

# vsl_name = st.selectbox(
#     "선박명",
#     ("G.Dream", "G.Hope", "G.Future", 'Universal Honor', 'Universal Glory','V. Prosperity','V. Glory','V. Advance','V. Harmony', 'V. Progress')
# )
# date=pd.to_datetime(st.date_input("Enter Date", datetime.datetime.now()))
# spd_exp_b = st.slider("Ballast Spd to Destination", min_value=0.1, max_value=17.0, value=10.5, step=0.1)
# spd_exp_l_bofb = st.slider("Laden Spd to OFB", min_value=0.1, max_value=15.0, value=12.5, step=0.1)
# spd_exp_l_aofb = st.slider("Laden Spd from St. John to Dest", min_value=0.1, max_value=15.0, value=12.5, step=0.1)
# spd_pred = st.slider("Spd for Route Prediction", min_value=0.1, max_value=15.0, value=12.5, step=0.1)

# df['ord_spd_l_bofb'] = df.apply(lambda x: x['Dist. to OFB']/((x['ETA OFB'] - x['UTC']).total_seconds() / 3600) if (x['Status'] == 2) and (x['ETA OFB']>x['UTC']) else None, axis=1)
# df['ord_spd_l_aofb'] = df.apply(lambda x: (x['Dist. to Next Port']-x['Dist. to St.John'])/((x['ETA Ordered']-x['ETA St. John']).total_seconds() / 3600) if (x['Status'] == 2) and (x['ETA St. John']>x['UTC']) else None, axis=1)


# df['exp_eta_l_ofb'] = df.apply(lambda x: x['UTC'] + datetime.timedelta(hours=x['Dist. to OFB'] / spd_exp_l_bofb) if (x['Status'] == 2) else None,axis=1)
# df['exp_eta_l_ofb'] = df['exp_eta_l_ofb'].apply(lambda x: x if 7 <= x.hour <= 21 else x + datetime.timedelta(hours=9))
# df['exp_eta_l'] = df.apply(lambda x: x['exp_eta_l_ofb']+datetime.timedelta(hours=(x['Dist. to Next Port']-x['Dist. to OFB']) / spd_exp_l_aofb) if (x['Status'] == 2) else None,axis=1)
# df['exp_eta_b'] = df.apply(lambda x: x['UTC'] + datetime.timedelta(hours=x['Dist. to Next Port'] / spd_exp_b) if (x['Status'] == 0) or (x['Status'] == 1) else None,axis=1)


###########################################################################################################################################################################
# Waypoints with names and coordinates
waypoints_b = [
    (34.7447, 127.7379),
    (28.0000, 124.500),
    (21.0000, 121.0000), 
    (12.0000, 112.5000),  
    (1.5000, 104.6666),
    (1.2590, 103.8125),
    (2.5000, 101.0000),
    (6.0000, 97.5000), 
    (5.3333, 80.2000),
    (10.0000, 72.0000),
    (20.0000, 65.0000),
    (26.5000, 56.5000),
]
waypoints_l = [
    (26.5000, 56.5000),
    (20.0000, 65.0000),
    (10.0000, 72.0000),
    (5.3333, 80.2000),
    (6.0000, 97.5000),
    (2.5000, 101.0000),
    (1.2590, 103.8125),
    (1.5000, 104.6666),
    (12.0000, 112.5000),
    (21.0000, 121.0000),
    (28.0000, 124.5000),
    (34.7447, 127.7379),
]

def interpolate_points(waypoints, num_points=40):  
    new_waypoints = []
    
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        
        new_waypoints.append(start)  # Add the start point
        
        # Calculate intermediate points
        for j in range(1, num_points + 1):
            lat = start[0] + (end[0] - start[0]) * j / (num_points + 1)
            lon = start[1] + (end[1] - start[1]) * j / (num_points + 1)
            new_waypoints.append((lat, lon))
    
    new_waypoints.append(waypoints[-1])  # Add the last waypoint
    return new_waypoints

# Get the new list of waypoints with denser interpolated points
waypoints_b = interpolate_points(waypoints_b, num_points=40)
waypoints_l = interpolate_points(waypoints_l, num_points=40)

def haversine(lat1, lon1, lat2, lon2):
    R = 3440  # Radius of Earth in nautical miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# Find the closest waypoint
def find_closest_waypoint(vessel_pos, waypoints):
    closest_wp = None
    min_distance = float('inf')
    
    for wp in waypoints:
        distance = haversine(vessel_pos[0], vessel_pos[1], wp[0], wp[1])
        if distance < min_distance:
            min_distance = distance
            closest_wp = wp
            
    return closest_wp, min_distance

# Calculate the new position after a given number of days
def calculate_position(vessel_pos, waypoints, speed, days):
    # Start at the closest waypoint
    closest_wp, _ = find_closest_waypoint(vessel_pos, waypoints)
    
    # Total distance the vessel will travel
    total_distance = speed * 24 * days  # Speed is in nautical miles per hour, and we multiply by 24 hours per day
    
    # Travel along the waypoints
    current_wp_index = waypoints.index(closest_wp)
    distance_traveled = 0
    current_pos = closest_wp
    
    while total_distance > 0 and current_wp_index < len(waypoints) - 1:
        next_wp = waypoints[current_wp_index + 1]
        leg_distance = haversine(current_pos[0], current_pos[1], next_wp[0], next_wp[1])
        
        if total_distance >= leg_distance:
            # Move to the next waypoint
            total_distance -= leg_distance
            current_pos = next_wp
            current_wp_index += 1
        else:
            # Travel partway towards the next waypoint
            fraction_of_leg = total_distance / leg_distance
            lat_diff = next_wp[0] - current_pos[0]
            lon_diff = next_wp[1] - current_pos[1]
            new_lat = current_pos[0] + fraction_of_leg * lat_diff
            new_lon = current_pos[1] + fraction_of_leg * lon_diff
            current_pos = (new_lat, new_lon)
            total_distance = 0
    
    return current_pos

###########################################################################################################################################################################





with tab1:

    vsl_name = st.selectbox(
    "선박명",
    ("G.Dream", "G.Hope", "G.Future", 'Universal Honor', 'Universal Glory','V. Prosperity','V. Glory','V. Advance','V. Harmony', 'V. Progress')
    )
    date=pd.to_datetime(st.date_input("Enter Date", datetime.datetime.now()-datetime.timedelta(days=1)))
    spd_exp_b = st.slider("Ballast Spd to Destination", min_value=0.1, max_value=17.0, value=10.5, step=0.1)
    spd_exp_l_bofb = st.slider("Laden Spd to OFB", min_value=0.1, max_value=15.0, value=12.5, step=0.1)
    spd_exp_l_aofb = st.slider("Laden Spd from St. John to Dest", min_value=0.1, max_value=15.0, value=12.5, step=0.1)
    spd_pred = st.slider("Spd for Route Prediction", min_value=0.1, max_value=15.0, value=12.5, step=0.1)

    df['ord_spd_l_bofb'] = df.apply(lambda x: x['Dist. to OFB']/((x['ETA OFB'] - x['UTC']).total_seconds() / 3600) if (x['Status'] == 2) and (x['ETA OFB']>x['UTC']) else None, axis=1)
    df['ord_spd_l_aofb'] = df.apply(lambda x: (x['Dist. to Next Port']-x['Dist. to St.John'])/((x['ETA Ordered']-x['ETA St. John']).total_seconds() / 3600) if (x['Status'] == 2) and (x['ETA St. John']>x['UTC']) else None, axis=1)


    df['exp_eta_l_ofb'] = df.apply(lambda x: x['UTC'] + datetime.timedelta(hours=x['Dist. to OFB'] / spd_exp_l_bofb) if (x['Status'] == 2) else None,axis=1)
    df['exp_eta_l_ofb'] = df['exp_eta_l_ofb'].apply(lambda x: x if 7 <= x.hour <= 21 else x + datetime.timedelta(hours=9))
    df['exp_eta_l'] = df.apply(lambda x: x['exp_eta_l_ofb']+datetime.timedelta(hours=(x['Dist. to Next Port']-x['Dist. to OFB']) / spd_exp_l_aofb) if (x['Status'] == 2) else None,axis=1)
    df['exp_eta_b'] = df.apply(lambda x: x['UTC'] + datetime.timedelta(hours=x['Dist. to Next Port'] / spd_exp_b) if (x['Status'] == 0) or (x['Status'] == 1) else None,axis=1)

    # 날짜에 해당하는 데이터가 있는지 검증
    data_validation_df = df.loc[df['UTC'].dt.date == date.date()]
    col1, col2, col3, col4 = st.columns(4)
    if len(data_validation_df) > 0:
        # 있는 경우

        if df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date == date.date()), "Status"].values[0] == 2 :
            waypoints=waypoints_l
        else :
            waypoints=waypoints_b
    else:
        # 없는 경우
        st.error("조회하신 날짜에는 데이터가 없습니다")
        st.stop()

    position=[]
    for i in range(7) :
        position.append(calculate_position(tuple(df.loc[(df['Vessel_Full']==vsl_name)&(df['UTC'].dt.date==date.date()),['LatDD', 'LongDD']].values[0]),waypoints,spd_pred,i+1))
    position.append(tuple(df.loc[(df['Vessel_Full']==vsl_name)&(df['UTC'].dt.date==date.date()),['LatDD', 'LongDD']].values[0]))                     
    if df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date == date.date()), "Status"].values[0] == 2:
        st.map(pd.DataFrame(position,columns=["lat", "lon"]), color='#FF0000')
        components.html(
            """
            <iframe 
                width="750" 
                height="550" 
                src="https://embed.windy.com/embed.html?type=map&location=coordinates&metricRain=default&metricTemp=default&metricWind=default&zoom=4&overlay=wind&product=ecmwf&level=surface&lat=19.034&lon=118.213" 
                frameborder="0">
            </iframe>
            """,
            height=550,
            width=750,
        )
    else :
        st.map(pd.DataFrame(position,columns=["lat", "lon"]), color='#0000FF')
        components.html(
        """
        <iframe 
            width="750" 
            height="550" 
            src="https://embed.windy.com/embed.html?type=map&location=coordinates&metricRain=default&metricTemp=default&metricWind=default&zoom=4&overlay=wind&product=ecmwf&level=surface&lat=19.034&lon=118.213" 
            frameborder="0">
        </iframe>
        """,
        height=550,
        width=700,
        )
#############################################RPM Cons.예측 모델####################################################################
    
    if st.button('Estimate RPM & Bunker Cons.') : 

        features_rpm = [
            #"UTC",
            "Steaming Time",
            #'LatDD',
            #'LongDD',
            "Dist. to Next Port",
            'Dist. Daily',
            'Dist. Total',
            #'Geohash_num',
            "Dist. to OFB",
            #'Dist. to St.John'
            "Spd. Present",
            'HSFO ROB',
            'LSFO ROB',
            #'Total_FO_ROB',
            "Spd. Daily Avg.",
            #'Spd. Total Avg.',
            #"Wind Dir.",
            "Wind B.S",
            #"Sea",
            "Slip",
            'HSFO M/E',
            'LSFO M/E',
            'Voyage Completion Ratio',
            "Total_FO",
            "Status",
            "Vessel",
            #"RPM",
            #"year",
            "month",
            #"day",
            #"hour"
            ]

        x_rpm=df[features_rpm]

        y_rpm=df['RPM']
        x_train_rpm, x_test_rpm, y_train_rpm, y_test_rpm = train_test_split(x_rpm, y_rpm, test_size=0.2, random_state=123)
        model_rpm = XGBRegressor(
            n_estimators=4000,
            learning_rate=10/4000,
            max_depth=7,
            subsample=0.7,
            colsample_bytree=0.9,
            random_state=123,
            n_jobs=-1,
        )

        model_rpm.fit(x_train_rpm, y_train_rpm)

        pred_rpm = model_rpm.predict(x_test_rpm)
        pred_rpm[pred_rpm < 0] = 0

    ##################################################################################################################################
    #############################################Bunker Cons.예측 모델################################################################        
        features_bunker = [
            #"UTC",
            "Steaming Time",
            #'LatDD',
            #'LongDD',
            "Dist. to Next Port",
            'Dist. Daily',
            'Dist. Total',
            #'Geohash_num',
            #"Dist. to OFB",
            "Spd. Present",
            'HSFO ROB',
            'LSFO ROB',
            #'Total_FO_ROB',
            "Spd. Daily Avg.",
            #'Spd. Total Avg.',
            #"Wind Dir.",
            "Wind B.S",
            "Sea",
            "Slip",
            'Voyage Completion Ratio',
            #"Total_FO",
            "Status",
            "Vessel",
            "RPM",
            "year",
            "month",
            #"day",
            #"hour"
        ]

        x_bunker=df[features_bunker]
        y_bunker=df['Total_FO']
        x_train_bunker, x_test_bunker, y_train_bunker, y_test_bunker = train_test_split(x_bunker, y_bunker, test_size=0.2, random_state=123)
        model_bunker = XGBRegressor(
            n_estimators=3500,
            learning_rate=10/3500,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.9,
            random_state=123,
            n_jobs=-1,
        )

        model_bunker.fit(x_train_bunker, y_train_bunker)

        pred_bunker = model_bunker.predict(x_test_bunker)
        pred_bunker[pred_bunker < 0] = 0

        with col1:
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'From Port (A)']
            df.loc[(df['Status']!=2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'exp_eta_b']
            df.loc[(df['Status']==2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'exp_eta_l_ofb']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ETA OFB']
            model_rpm.predict(df[features_rpm])[df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date())].index]
            mae_rpm = round(mean_absolute_error(y_test_rpm, pred_rpm),1)
            st.write(f"RPM Error: {mae_rpm}")
    
        with col2:
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'To Port (B)']
            df.loc[(df['Status']==2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'exp_eta_l']
            df.loc[(df['Status']==2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ord_spd_l_bofb']
            df.loc[(df['Status']==2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ord_spd_l_aofb']
            model_bunker.predict(df[features_bunker])[df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date())].index]
            mae_bunker = round(mean_absolute_error(y_test_bunker, pred_bunker),1)
            st.write(f"Bunker Cons. Error: {mae_bunker}")

        with col3:
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ETA Ordered']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ETA Expected']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ETA Daily Avg.']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ETA Total Avg.']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'RPM']

        with col4:
            df.loc[(df['Status']!=2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'Spd. Ordered']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'Spd. Expected']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'Spd. Daily Avg.']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'Spd. Total Avg.']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'Total_FO']

    else :
        with col1:
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'From Port (A)']
            df.loc[(df['Status']!=2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'exp_eta_b']
            df.loc[(df['Status']==2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'exp_eta_l_ofb']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ETA OFB']
    
        with col2:
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'To Port (B)']
            df.loc[(df['Status']==2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'exp_eta_l']
            df.loc[(df['Status']==2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ord_spd_l_bofb']
            df.loc[(df['Status']==2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ord_spd_l_aofb']

        with col3:
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ETA Ordered']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ETA Expected']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ETA Daily Avg.']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'ETA Total Avg.']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'RPM']

        with col4:
            df.loc[(df['Status']!=2)&(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'Spd. Ordered']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'Spd. Expected']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'Spd. Daily Avg.']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'Spd. Total Avg.']
            df.loc[(df["Vessel_Full"]==vsl_name)&(df['UTC'].dt.date==date.date()), 'Total_FO']


with tab2:

    anal_windbs = 1
    anal_sea = 1
    anal_slip = 0
    anal_month = 6

    features_rpm_anal = [
            #"UTC",
            "Steaming Time",
            #'LatDD',
            #'LongDD',
            "Dist. to Next Port",
            'Dist. Daily',
            'Dist. Total',
            #'Geohash_num',
            "Dist. to OFB",
            #'Dist. to St.John'
            "Spd. Present",
            'HSFO ROB',
            'LSFO ROB',
            #'Total_FO_ROB',
            "Spd. Daily Avg.",
            #'Spd. Total Avg.',
            #"Wind Dir.",
            "Wind B.S",
            "Sea",
            "Slip",
            #'HSFO M/E',
            #'LSFO M/E',
            'Voyage Completion Ratio',
            #"Total_FO",
            "Status",
            "Vessel",
            #"RPM",
            #"year",
            "month",
            #"day",
            #"hour"
            ]
    features_anal = [
            #"UTC",
            "Steaming Time",
            #'LatDD',
            #'LongDD',
            "Dist. to Next Port",
            'Dist. Daily',
            'Dist. Total',
            #'Geohash_num',
            "Dist. to OFB",
            #'Dist. to St.John'
            "Spd. Present",
            'HSFO ROB',
            'LSFO ROB',
            #'Total_FO_ROB',
            "Spd. Daily Avg.",
            #'Spd. Total Avg.',
            #"Wind Dir.",
            "Wind B.S",
            "Sea",
            "Slip",
            #'HSFO M/E',
            #'LSFO M/E',
            'Voyage Completion Ratio',
            "Total_FO",
            "Status",
            "Vessel",
            #"RPM",
            #"year",
            "month",
            #"day",
            #"hour"
            ]

    features_bunker_anal = [
        #"UTC",
        "Steaming Time",
        #'LatDD',
        #'LongDD',
        "Dist. to Next Port",
        'Dist. Daily',
        'Dist. Total',
        #'Geohash_num',
        "Dist. to OFB",
        "Spd. Present",
        'HSFO ROB',
        'LSFO ROB',
        #'Total_FO_ROB',
        "Spd. Daily Avg.",
        #'Spd. Total Avg.',
        #"Wind Dir.",
        "Wind B.S",
        "Sea",
        "Slip",
        'Voyage Completion Ratio',
        #"Total_FO",
        "Status",
        "Vessel",
        "RPM",
        #"year",
        "month",
        #"day",
        #"hour"
        ]
    
    if st.button('RPM & Bunker 분석') : 

        x_rpm_anal=df[features_rpm_anal]
        y_rpm_anal=df['RPM']
        model_anal_rpm = XGBRegressor(
            n_estimators=4000,
            learning_rate=10/4000,
            max_depth=7,
            subsample=0.7,
            colsample_bytree=0.9,
            random_state=123,
            n_jobs=-1,
        )
        model_anal_rpm.fit(x_rpm_anal, y_rpm_anal)
        df_anal_rpm = pd.DataFrame(columns=features_rpm_anal, index=range(17*14*2))
        df_anal_rpm['Steaming Time']=1
        df_anal_rpm['Dist. to Next Port']=df['Dist. to Next Port'].mean()
        df_anal_rpm['Dist. Total'] = df['Dist. Total'].mean()
        df_anal_rpm['Dist. to OFB'] = df['Dist. to OFB'].mean()
        repeated_speeds1 = np.tile([9*24, 9.5*24, 10*24, 10.5*24, 11*24, 11.5*24, 12*24, 12.5*24, 13*24, 13.5*24, 14*24, 14.5*24, 15*24, 15.5*24, 16*24, 16.5*24, 17*24], 28)
        repeated_speeds2 =np.tile([9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17], 28)
        df_anal_rpm['Dist. Daily'] = repeated_speeds1
        df_anal_rpm['Spd. Present'] = repeated_speeds2
        df_anal_rpm['Spd. Daily Avg.'] = repeated_speeds2
        df_anal_rpm['HSFO ROB'] = df['HSFO ROB'].mean()
        df_anal_rpm['LSFO ROB'] = df['LSFO ROB'].mean()
        df_anal_rpm['Wind B.S']=anal_windbs
        df_anal_rpm['Sea']=anal_sea
        df_anal_rpm['Slip']=anal_slip
        df_anal_rpm['Voyage Completion Ratio']=df['Voyage Completion Ratio'].mean()
        df_anal_rpm['Status'] = np.tile(np.repeat([0, 2], 17), 14)
        df_anal_rpm["month"]=anal_month
        repeated_vessel = np.repeat(range(1, 15), 17*2)
        df_anal_rpm['Vessel'] = repeated_vessel
        pred_anal_rpm = model_anal_rpm.predict(df_anal_rpm)
        df_anal_rpm['RPM']=pred_anal_rpm


        x_bunker_anal=df[features_bunker_anal]
        y_bunker_anal=df['Total_FO']
        model_anal_bunker = XGBRegressor(
            n_estimators=3500,
            learning_rate=10/3500,
            max_depth=6,
            subsample=0.7,
            colsample_bytree=0.9,
            random_state=123,
            n_jobs=-1,
        )
        model_anal_bunker.fit(x_bunker_anal, y_bunker_anal)
        df_anal_bunker = pd.DataFrame(columns=features_bunker_anal, index=range(17*14*2))
        df_anal_bunker['Steaming Time']=1
        df_anal_bunker['Dist. to Next Port']=df['Dist. to Next Port'].mean()
        df_anal_bunker['Dist. Total'] = df['Dist. Total'].mean()
        df_anal_bunker['Dist. to OFB'] = df['Dist. to OFB'].mean()
        repeated_speeds1 = np.tile([9*24, 9.5*24, 10*24, 10.5*24, 11*24, 11.5*24, 12*24, 12.5*24, 13*24, 13.5*24, 14*24, 14.5*24, 15*24, 15.5*24, 16*24, 16.5*24, 17*24], 28)
        repeated_speeds2 =np.tile([9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17], 28)
        df_anal_bunker['Dist. Daily'] = repeated_speeds1
        df_anal_bunker['Spd. Present'] = repeated_speeds2
        df_anal_bunker['Spd. Daily Avg.'] = repeated_speeds2
        df_anal_bunker['HSFO ROB'] = df['HSFO ROB'].mean()
        df_anal_bunker['LSFO ROB'] = df['LSFO ROB'].mean()
        df_anal_bunker['Wind B.S']=anal_windbs
        df_anal_bunker['Sea']=anal_sea
        df_anal_bunker['Slip']=anal_slip
        df_anal_bunker['Voyage Completion Ratio']=df['Voyage Completion Ratio'].mean()
        df_anal_bunker['Status'] = np.tile(np.repeat([0, 2], 17), 14)
        df_anal_bunker["month"]=anal_month
        df_anal_bunker['RPM']=pred_anal_rpm
        repeated_vessel = np.repeat(range(1, 15), 17*2)
        df_anal_bunker['Vessel'] = repeated_vessel
        pred_anal_bunker = model_anal_bunker.predict(df_anal_bunker)
        df_anal_bunker['Total_FO']=pred_anal_bunker


        x_anal=df[features_anal]
        y_anal=df['RPM']
        model_anal = XGBRegressor(
            n_estimators=4000,
            learning_rate=10/4000,
            max_depth=7,
            subsample=0.7,
            colsample_bytree=0.9,
            random_state=123,
            n_jobs=-1,
        )
        model_anal.fit(x_anal, y_anal)
        df_anal = pd.DataFrame(columns=features_anal, index=range(17*14*2))
        df_anal['Steaming Time']=1
        df_anal['Dist. to Next Port']=df['Dist. to Next Port'].mean()
        df_anal['Dist. Total'] = df['Dist. Total'].mean()
        df_anal['Dist. to OFB'] = df['Dist. to OFB'].mean()
        repeated_speeds1 = np.tile([9*24, 9.5*24, 10*24, 10.5*24, 11*24, 11.5*24, 12*24, 12.5*24, 13*24, 13.5*24, 14*24, 14.5*24, 15*24, 15.5*24, 16*24, 16.5*24, 17*24], 28)
        repeated_speeds2 =np.tile([9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17], 28)
        df_anal['Dist. Daily'] = repeated_speeds1
        df_anal['Spd. Present'] = repeated_speeds2
        df_anal['Spd. Daily Avg.'] = repeated_speeds2
        df_anal['HSFO ROB'] = df['HSFO ROB'].mean()
        df_anal['LSFO ROB'] = df['LSFO ROB'].mean()
        df_anal['Wind B.S']=anal_windbs
        df_anal['Sea']=anal_sea
        df_anal['Slip']=anal_slip
        df_anal['Voyage Completion Ratio']=df['Voyage Completion Ratio'].mean()
        df_anal['Total_FO']=pred_anal_bunker
        df_anal['Status'] = np.tile(np.repeat([0, 2], 17), 14)
        df_anal["month"]=anal_month
        repeated_vessel = np.repeat(range(1, 15), 17*2)
        df_anal['Vessel'] = repeated_vessel
        pred_anal = model_anal.predict(df_anal)
        df_anal['RPM']=pred_anal




        col3, col4 = st.columns(2)

        with col3:
            
            anal_windbs = st.slider("Wind B.S.", min_value=1, max_value=12, value=1, step=1)
            anal_sea = st.slider("Sea", min_value=1, max_value=8, value=1, step=1)
            anal_slip = st.slider("Slip", min_value=-40, max_value=70, value=0, step=1)
            anal_month = st.slider("Month", min_value=1, max_value=12, value=6, step=1)
            
            st.header("Ballast")
            st.write("<h1 style='text-align: center; font-size: 25px'>G.Dream</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==1), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>G.Hope</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==2), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>G.Future</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==3), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>U. Honor</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==6), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>U. Glory</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==7), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>V. Harmony</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==8), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>V. Prosperity</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==9), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>V. Glory</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==10), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>V. Advance</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==11), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>SM Venus2</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==13), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>V. Progess</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==0) & (df_anal_rpm['Vessel']==14), ['Spd. Daily Avg.','RPM', 'Total_FO']]

        with col4:
            fig, ax = plt.subplots()
            x_values = df_anal.groupby('Spd. Daily Avg.')['Total_FO'].mean().reset_index()['Spd. Daily Avg.']
            y_values = df_anal.groupby('Spd. Daily Avg.')['Total_FO'].mean().reset_index()['Total_FO']
            ax.plot(x_values, y_values)
            ax.set_xlabel('Speed')
            ax.set_ylabel('Total FO')
            ax.set_title('Total FO vs Speed')
            st.pyplot(fig)

            st.header("Laden")
            st.write("<h1 style='text-align: center; font-size: 25px'>G.Dream</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==1), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>G.Hope</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==2), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>G.Future</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==3), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>U. Honor</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==6), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>U. Glory</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==7), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>V. Harmony</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==8), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>V. Prosperity</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==9), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>V. Glory</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==10), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>V. Advance</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==11), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>SM Venus2</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==13), ['Spd. Daily Avg.','RPM', 'Total_FO']]
            st.write("<h1 style='text-align: center; font-size: 25px'>V. Progess</h1>", unsafe_allow_html=True)
            df_anal.loc[(df_anal_rpm['Status']==2) & (df_anal_rpm['Vessel']==14), ['Spd. Daily Avg.','RPM', 'Total_FO']]



with tab3:
    spd_sim = st.slider("Spd for Simulation", min_value=0.1, max_value=15.0, value=12.5, step=0.1)

    status_sim = st.radio(
    'Ballast or Laden',
    ["Ballast", "Laden"],
    index=1,
    )

    if status_sim=='Ballast' :
        waypoints=waypoints_l
    else :
        waypoints=waypoints_b


    lat_sim = st.number_input(
    "latitude", value=1.13, placeholder="Type a number..."
    )
    long_sim = st.number_input(
    "longitude", value=103.5, placeholder="Type a number..."
    )




    position_sim=[]
    for i in range(7) :
        position_sim.append(calculate_position((lat_sim,long_sim),waypoints,spd_sim,i+1))
    position_sim.append((lat_sim,long_sim))                     
    if status_sim == 'Laden':
        st.map(pd.DataFrame(position_sim,columns=["lat", "lon"]), color='#FF0000')
        components.html(
            """
            <iframe 
                width="750" 
                height="550" 
                src="https://embed.windy.com/embed.html?type=map&location=coordinates&metricRain=default&metricTemp=default&metricWind=default&zoom=4&overlay=wind&product=ecmwf&level=surface&lat=19.034&lon=118.213" 
                frameborder="0">
            </iframe>
            """,
            height=550,
            width=750,
        )
    else :
        st.map(pd.DataFrame(position,columns=["lat", "lon"]), color='#0000FF')
        components.html(
        """
        <iframe 
            width="750" 
            height="550" 
            src="https://embed.windy.com/embed.html?type=map&location=coordinates&metricRain=default&metricTemp=default&metricWind=default&zoom=4&overlay=wind&product=ecmwf&level=surface&lat=19.034&lon=118.213" 
            frameborder="0">
        </iframe>
        """,
        height=550,
        width=700,
        )