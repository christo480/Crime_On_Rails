from lib2to3.pgen2.pgen import DFAState
from turtle import color
import pandas as pd
import numpy as np
import re
from datetime import datetime
from geojson import Point
from geopy import distance
import folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
from scipy.special import expit
"""
Name: Christian Gillies
Email: Christian.Gillies04@myhunter.cuny.edu
Resources:
NOTE: Code from previous assignments has been modified to keep consitiency and style

Used https://pandas.pydata.org/ for pandas reference
Used https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.append.html
            https://www.kite.com/python/answers/how-to-select-rows-by-multiple-label-conditions-with-pandas-loc-in-python
            https://www.geeksforgeeks.org/how-to-select-multiple-columns-in-a-pandas-dataframe/
Used    https://www.w3schools.com/python/python_regex.asp
    https://www.guru99.com/python-dictionary-append.html
    https://stackoverflow.com/questions/15081516/how-to-create-an-empty-if-statement-in-python
    https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it

Used https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-a-certain-column-is-nan to help figure out dropping issue
Used https://nostarch.com/download/automate2e_SampleCh7.pdf
Used https://stackoverflow.com/questions/466345/converting-string-into-datetime
Used https://note.nkmk.me/en/python-function-return-multiple-values/#:~:text=Return%20multiple%20values%20using%20commas,the%20return%20%2C%20separated%20by%20commas.
Used https://dataindependent.com/pandas/add-column-to-dataframe-pandas/
Used https://python-graph-gallery.com/map-read-geojson-with-python-geopandas
https://gis.stackexchange.com/questions/130963/write-geojson-into-a-geojson-file-with-python
Used https://matthew-brett.github.io/teaching/string_formatting.html
Used https://favtutor.com/blogs/list-to-dataframe-python

Calculation
https://www.calculator.net/distance-calculator.html?type=3&la1=38.8976&lo1=-77.0366&la2=39.9496&lo2=-75.1503&ctype=dec&lad1=38&lam1=53&las1=51.36&lau1=n&lod1=77&lom1=2&los1=11.76&lou1=w&lad2=39&lam2=56&las2=58.56&lau2=n&lod2=75&lom2=9&los2=1.08&lou2=w&x=80&y=24#latlog
https://www.geeksforgeeks.org/get-minimum-values-in-rows-or-columns-with-their-index-position-in-pandas-dataframe/
"""

def make_df(file_name):
    df = pd.read_csv(file_name)
    #prefrom any adjustments here on df
    return df

def look_up(arrest_df,station_name): #Matches up station name with distance df
    
    distance_df_name ="DISTANCE_TO_"+station_name +".csv"
    distance_df = make_df(distance_df_name)
    station_df = distance_df.merge(arrest_df,how='left', left_on=['ARREST_KEY'], right_on=['ARREST_KEY'])
    return station_df,distance_df_name

def assign_distance_catagories_catagory(distances):
    arrest_distance_catagories= ["Within 0.5 but more than 0.4","Within 0.4 but more than 0.3","Within 0.3 but more than 0.2","Within 0.2 but more than 0.1","Within 0.1 "]
    catagory_list = list()
    for distance in distances:
        if(distance>"0.4"):
            catagory_list.append(arrest_distance_catagories[0])
        elif(distance<"0.4" and distance>"0.3"):
            catagory_list.append(arrest_distance_catagories[1])
        elif(distance<"0.3" and distance>"0.2"):
            catagory_list.append(arrest_distance_catagories[2])
        elif(distance<"0.2" and distance>"0.1"):
            catagory_list.append(arrest_distance_catagories[3])
        elif(distance<"0.1" ):
            catagory_list.append(arrest_distance_catagories[4])
    return catagory_list
"""    
def assign_distance_catagory_probability(df,col,distance_catagories,produced_col_name):
    total = len(df)
    catagories =list()
    for c in df:
        if c not in 
        if(c[col]==distance_catagories[0]):
            catagories.append(len(df[df[[col]==distance_catagories[0]]])/total)
        elif(c[col]==distance_catagories[1]):
            catagories.append(len(df[df[[col]==distance_catagories[1]]])/total)
        elif(c[col]==distance_catagories[2]):
            catagories.append(len(df[df[[col]==distance_catagories[2]]])/total)
        elif(c[col]==distance_catagories[3]):
            catagories.append(len(df[df[[col]==distance_catagories[3]]])/total)
        else:
            catagories.append(len(df[df[[col]==distance_catagories[4]]])/total)
    df[produced_col_name] = pd.DataFrame(catagories)"""
   


#Code here is designed to interpet each DISTANCE_TO_{Station}
station_df =make_df('Station_GEODATA.csv')

arrest_df  =make_df('NYPD_Arrest_Data__Year_to_Date_.csv')
arrest_df['ARREST_DATE'] = pd.to_datetime(arrest_df['ARREST_DATE'])
arrest_df.sort_values(by='ARREST_DATE')

#DISTANCE_TO_df= make_df('DISTANCE_TO_Grand Central - 42nd St.csv') #Distance to Df

station_to_look_up ='Grand Central - 42nd St'
DISTANCE_TO_df,station_name= look_up(arrest_df,station_to_look_up)

DISTANCE_TO_df =DISTANCE_TO_df.merge(arrest_df,how='left', left_on=['ARREST_KEY'], right_on=['ARREST_KEY']) 
DISTANCE_TO_df =DISTANCE_TO_df.sort_values(by=(station_name.replace('.csv','')))
DISTANCE_TO_df =DISTANCE_TO_df.reset_index(drop=True)
print(DISTANCE_TO_df)
print(DISTANCE_TO_df.columns)
"""
print(station_df)
print(arrest_df)
print(arrest_df[['Latitude', 'Longitude']])
"""
#distances = range(0,4,0.5) #0.5 km is the standard of walking distance
#arrest_map = folium.Map(location=[40.7128,-74.0060],zoom_start=20) #https://python-visualization.github.io/folium/quickstart.html#GeoJSON/TopoJSON-Overlays
#DISTANCE_TO_df['DISTANCE_TO_'+station_to_look_up] = DISTANCE_TO_df['DISTANCE_TO_'+station_to_look_up].replace(" km","")
#DISTANCE_TO_df['DISTANCE_TO_'+station_to_look_up] = pd.to_numeric(DISTANCE_TO_df['DISTANCE_TO_'+station_to_look_up])

arrest_distance_catagories= ["Within 0.5 but more than 0.4","Within 0.4 but more than 0.3","Within 0.3 but more than 0.2","Within 0.2 but more than 0.1","Within 0.1 "]
arrests_limited = list()
arrests_limited_count = list()

arrests_with_walking_distance = DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.5"]
total_arrests_within_walking_distance = DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.5"].count()
arrests_with_walking_distance.loc[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]>"0.4","Chance of Arrest given within 0.5km"] =DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.5"].count()/total_arrests_within_walking_distance
print(arrests_with_walking_distance.dropna())
arrests_with_walking_distance['Within 0.5km']=(DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.5")
arrests_with_walking_distance['Within 0.4km']=(DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.4")
arrests_with_walking_distance['Within 0.3km']=(DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.3")
arrests_with_walking_distance['Within 0.2km']=(DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.2")
arrests_with_walking_distance['Within 0.1km']=(DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.1")
"""for arrest in arrests_with_walking_distance:
    if(arrest['Within 0.5km']==True)"""


print(arrests_with_walking_distance)
#Loop through each arrest and create a list with the probabilities we got from the below portion. This will let you make a y col for the sk learn
"""
print(arrests_with_walking_distance)
#catagory_list =assign_distance_catagories_catagory(arrests_with_walking_distance)
arrests_with_walking_distance['Distance_Catagory'] =DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]>"0.4" & DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.5"]
print(arrests_with_walking_distance['Distance_Catagory'])

catagory_list = DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]>"0.3"] and DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.4",arrest_distance_catagories[1],catagory_list])
catagory_list = np.where(DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]>"0.2"] and DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.3",arrest_distance_catagories[2],catagory_list])
catagory_list = np.where(DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]>"0.1"] and DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.2",arrest_distance_catagories[3],catagory_list])
catagory_list = np.where(DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.1",arrest_distance_catagories[4],catagory_list])
print(catagory_list)
arrests_with_walking_distance['Distance_Catagory'] =pd.DataFrame(catagory_list)
print(arrests_with_walking_distance)
#arrests_with_walking_distance = arrests_with_walking_distance.dropna()
for i in range(len(arrest_distance_catagories)):
    print("Probability of arrest in",arrest_distance_catagories[i])
    print(len(arrests_with_walking_distance[arrests_with_walking_distance['Distance_Catagory']==arrest_distance_catagories[i]]))

print(arrests_with_walking_distance)
"""
"""
arrests_with_walking_distance = DISTANCE_TO_df[DISTANCE_TO_df[('DISTANCE_TO_'+station_to_look_up)]<"0.5"]
limit = arrests_with_walking_distance[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]>"0.4" ]
limit = limit.reset_index(drop=True)
arrests_limited_count.append(len(limit))
range_text =" (Within 0.5 but more than 0.4)"
print(range_text)
print("Count:",len(limit))
print(limit)
arrests_limited.append(limit)

for i in range(len(limit)):#from index of 0 to coord
    folium.Marker([limit['Latitude_y'][i], limit['Longitude_y'][i]], popup="<i>{}</i>".format(str(limit['ARREST_DATE_y'][i])+":"+str(limit['PD_DESC_x'][i])+range_text),icon=folium.Icon(color='red')).add_to(arrest_map)


arrests_with_walking_distance = arrests_with_walking_distance[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]<"0.4" ]
limit = arrests_with_walking_distance[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]>"0.3" ]
limit = limit.reset_index(drop=True)
arrests_limited_count.append(len(limit))
range_text =" (Within 0.4 but more than 0.3)"
print(range_text)
print("Count:",len(limit))
print(limit)
arrests_limited.append(limit)

for i in range(len(limit)):#from index of 0 to coord
    folium.Marker([limit['Latitude_y'][i], limit['Longitude_y'][i]], popup="<i>{}</i>".format(str(limit['ARREST_DATE_y'][i])+":"+str(limit['PD_DESC_x'][i])+range_text),icon=folium.Icon(color='red')).add_to(arrest_map)

arrests_with_walking_distance = arrests_with_walking_distance[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]<"0.3" ]
limit = arrests_with_walking_distance[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]>"0.2" ]
limit = limit.reset_index(drop=True)
arrests_limited_count.append(len(limit))
range_text =" (Within 0.3 but more than 0.2)"
print(range_text)
print("Count:",len(limit))
print(limit)
arrests_limited.append(limit)

for i in range(len(limit)):#from index of 0 to coord
    folium.Marker([limit['Latitude_y'][i], limit['Longitude_y'][i]], popup="<i>{}</i>".format(str(limit['ARREST_DATE_y'][i])+":"+str(limit['PD_DESC_x'][i])+range_text),icon=folium.Icon(color='red')).add_to(arrest_map)


arrests_with_walking_distance = arrests_with_walking_distance[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]<"0.2" ]
limit = arrests_with_walking_distance[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]>"0.1" ]
limit = limit.reset_index(drop=True)
arrests_limited_count.append(len(limit))
range_text =" (Within 0.2 but more than 0.1)"
print(range_text)
print("Count:",len(limit))
print(limit)
arrests_limited.append(limit)

for i in range(len(limit)):#from index of 0 to coord
    folium.Marker([limit['Latitude_y'][i], limit['Longitude_y'][i]], popup="<i>{}</i>".format(str(limit['ARREST_DATE_y'][i])+":"+str(limit['PD_DESC_x'][i])+range_text),icon=folium.Icon(color='red')).add_to(arrest_map)

arrests_with_walking_distance = arrests_with_walking_distance[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]<"0.1" ]
limit = arrests_with_walking_distance
limit = limit.reset_index(drop=True)
arrests_limited_count.append(len(limit))
range_text =" (Within 0.1)"
print(range_text)
print("Count:",len(limit))
print(limit)
arrests_limited.append(limit)

for i in range(len(limit)):#from index of 0 to coord
    folium.Marker([limit['Latitude_y'][i], limit['Longitude_y'][i]], popup="<i>{}</i>".format(str(limit['ARREST_DATE_y'][i])+":"+str(limit['PD_DESC_x'][i])+range_text),icon=folium.Icon(color='red')).add_to(arrest_map)

html_name= station_name.replace('.csv','')+".html"
arrest_map.save(html_name) #Saves map to html file


#Bar Graph of nearby arrests
dataset = pd.DataFrame(list(zip(arrest_distance_catagories,arrests_limited)))
dataset_count = pd.DataFrame(list(zip(arrest_distance_catagories,arrests_limited_count)),columns= ['Distances','Number of Arrests'])
dataset.to_csv(station_to_look_up+"_distance_data.csv")
dataset_count.to_csv(station_to_look_up+"_distance_data_count.csv")
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="Distances", y="Number of Arrests", data=dataset_count)
#plt.show()"""