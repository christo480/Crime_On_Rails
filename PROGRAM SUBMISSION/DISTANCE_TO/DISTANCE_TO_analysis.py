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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
import os.path 
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

"""
    make_df(file_name): This function takes one input:
        file_name: the name of a CSV file containing Arrest Data from OpenData NYC.
    The function should open the file file_name as a DataFrame. If the total is null for a row, that row should be dropped. 
    The column nta2010 should be renamed NTA Code. The resulting DataFrame is returned.
"""
def make_df(file_name):
    df = pd.read_csv(file_name)
    #prefrom any adjustments here on df
    return df

"""
    look_up_DISTANCE_TO(arrest_df,station_name): 
    This function takes two inputs:
        -arrest_df: data frame with arrest data 
        -station_name: Used to make the appropiate file name. Generated files are of form "DISTANCE_TO_"+station_name +".csv"
        The DISTANCE_TO_ file  is then used to merge with the arrest df before being returned along with the name of the file used.
    
"""
def look_up_DISTANCE_TO(arrest_df,station_name): #Matches up station name with distance df
    
    distance_df_name ="DISTANCE_TO_"+station_name +".csv"
    file_exists =  os.path.exists(distance_df_name)
    if file_exists:
        distance_df = make_df(distance_df_name)
        station_df = distance_df.merge(arrest_df,how='left', left_on=['ARREST_KEY'], right_on=['ARREST_KEY'])
    else:
        exit(distance_df_name,'not found. Please generate with station_analysis.py')
    return station_df,distance_df_name


"""
    limit_col(df,col,upperbound,lowerbound):
    limits df[col] into [lowerbound<x<upperbound]
"""
def limit_col(df,col,upperbound,lowerbound):
    df = df[df[col]<upperbound]
    df = df[df[col]>lowerbound]
    return df

def catagorize_distances(row):
    if row > 0.5:
      val= 'Outside Walking Distance'
    elif row < 0.5 and row > 0.4:
        val = 'Within 0.5 but more than 0.4'
    elif row < 0.4 and row > 0.3:
        val = 'Within 0.4 but more than 0.3'
    elif row < 0.3 and row > 0.2:
        val = 'Within 0.3 but more than 0.2'
    elif row < 0.2 and row > 0.1:
        val = 'Within 0.2 but more than 0.1'
    else:
        val = 'Within 0.1'
    return val

#Code here is designed to interpet each DISTANCE_TO_{Station}
station_df =make_df('Station_GEODATA.csv')
arrest_df  =make_df('NYPD_Arrest_Data__Year_to_Date_.csv')
arrest_df['ARREST_DATE'] = pd.to_datetime(arrest_df['ARREST_DATE'])
arrest_df.sort_values(by='ARREST_DATE')
"""
This first portion prepares the data we need for our model.
-Station_DATA.csv contains the coodinates for each station
-NYPD_Arrest_Data__Year_to_Date_.csv contains numerous arrests yeat to date of about Febuary 26th, 2022

We will convert all the dates into proper date times to make it easier to compare dates should we need to.
"""

#DISTANCE_TO_df= make_df('DISTANCE_TO_Grand Central - 42nd St.csv') #Distance to Df

"""
'station_look_up' can be any station we have generated data for with station_analysis.py

NOTE: It may be useful to generate the requested station as we have the data to do so here. Perhaps a get_distance_from station(station_name,station_df,arrest_df) may be useful

After looking up DISTANCE_TO file it merges on arrest key
"""
station_to_look_up ='Grand Central - 42nd St'


DISTANCE_TO_df,DISTANCE_TO_filename= look_up_DISTANCE_TO(arrest_df,station_to_look_up)
DISTANCE_TO_colname =DISTANCE_TO_filename.replace('.csv','')

DISTANCE_TO_df =DISTANCE_TO_df.merge(arrest_df,how='left', left_on=['ARREST_KEY'], right_on=['ARREST_KEY']) 
DISTANCE_TO_df =DISTANCE_TO_df.sort_values(by=(DISTANCE_TO_colname))
DISTANCE_TO_df =DISTANCE_TO_df.reset_index(drop=True)
print(DISTANCE_TO_df)
print(DISTANCE_TO_df.columns)
"""
#Used to view all coordinates
print(station_df)
print(arrest_df)
print(arrest_df[['Latitude', 'Longitude']])
"""

"""
We need to clean the 'DISTANCE_TO' cols as they will have the numbers in string format
we do this my removing last two characters from each member and converting to floats
"""
"""
Creates map centered on New York for plotting
"""
arrest_map = folium.Map(location=[40.7128,-74.0060],zoom_start=20) #https://python-visualization.github.io/folium/quickstart.html#GeoJSON/TopoJSON-Overlays

DISTANCE_TO_df[DISTANCE_TO_colname] = DISTANCE_TO_df[DISTANCE_TO_colname].apply(lambda x : x[0:-2])
DISTANCE_TO_df[DISTANCE_TO_colname] = pd.to_numeric(DISTANCE_TO_df[DISTANCE_TO_colname])
print(DISTANCE_TO_df[DISTANCE_TO_colname])
#distances = range(0,4,0.5) #0.5 km is the standard of walking distance


#DISTANCE_TO_df['DISTANCE_TO_'+station_to_look_up] = pd.to_numeric(DISTANCE_TO_df['DISTANCE_TO_'+station_to_look_up])
"""
Now that our data is clean we can look at the clustering of arrests around the station. We can limit our arrests to those within walking distance(0.5 kms). 
"""
arrest_distance_catagories= ["Within 0.5 but more than 0.4","Within 0.4 but more than 0.3","Within 0.3 but more than 0.2","Within 0.2 but more than 0.1","Within 0.1 "]

arrests_with_walking_distance = limit_col(DISTANCE_TO_df,DISTANCE_TO_colname,0.5,0)
arrests_with_walking_distance['Distance_Catagory'] = arrests_with_walking_distance[DISTANCE_TO_colname].apply(catagorize_distances)
arrest_distance_catagories_count = list()


size = len(arrests_with_walking_distance)



#arrest_distance_catagories= ["Within 0.5 but more than 0.4","Within 0.4 but more than 0.3","Within 0.3 but more than 0.2","Within 0.2 but more than 0.1","Within 0.1 "]
arrests_limited = list()
arrests_limited_count = list()

"""
Using frequencies of distance catagories we can generate probabilities. With these we can finally make predictions.
"""
# Source: https://stackoverflow.com/questions/54541962/how-to-add-calculated-column-to-dataframe-counting-frequency-in-column-in-pandas
arrests_with_walking_distance['Probability'] =arrests_with_walking_distance['Distance_Catagory'].map(arrests_with_walking_distance['Distance_Catagory'].value_counts())/size
#Arrests within walking distance from a station is defined as all arrests within 0.5 km


print("Count:",len(arrests_with_walking_distance))
value_counts = arrests_with_walking_distance['Distance_Catagory'].value_counts()
print(arrests_with_walking_distance['Distance_Catagory'].value_counts())
print(value_counts[0])

print(arrests_with_walking_distance['Probability'])

arrests_with_walking_distance.to_csv(DISTANCE_TO_colname+"_Walking_Distance_Crime_Probabilities.csv")
"""
#Graph only the arrests within distance
for i in range(len(limit)):#from index of 0 to coord
    folium.Marker([limit['Latitude_y'][i], limit['Longitude_y'][i]], popup="<i>{}</i>".format(str(limit['ARREST_DATE_y'][i])+":"+str(limit['PD_DESC_x'][i])+range_text),icon=folium.Icon(color='red')).add_to(arrest_map)
"""

"""

arrests_with_walking_distance = arrests_with_walking_distance[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]<0.4 ]
limit = arrests_with_walking_distance[arrests_with_walking_distance[('DISTANCE_TO_'+station_to_look_up)]>0.3 ]
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
"""

"""
for i in range(len(limit)):#from index of 0 to coord
    folium.Marker([limit['Latitude_y'][i], limit['Longitude_y'][i]], popup="<i>{}</i>".format(str(limit['ARREST_DATE_y'][i])+":"+str(limit['PD_DESC_x'][i])+range_text),icon=folium.Icon(color='red')).add_to(arrest_map)
"""

"""
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

html_name= DISTANCE_TO_filename.replace('.csv','')+".html"
arrest_map.save(html_name) #Saves map to html file
"""

#Bar Graph of nearby arrests
"""
dataset = pd.DataFrame(list(zip(arrest_distance_catagories,arrests_limited)))
dataset_count = pd.DataFrame(list(zip(arrest_distance_catagories,arrests_limited_count)),columns= ['Distances','Number of Arrests'])
dataset.to_csv(station_to_look_up+"_distance_data.csv")
dataset_count.to_csv(station_to_look_up+"_distance_data_count.csv")
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="Distances", y="Number of Arrests", data=dataset_count)
plt.show()
"""
# Map all the stations

"""
for i in range(len(station_df)):
    folium.Marker([station_df['LAT'][i], station_df['LON'][i]], popup="<i>{}</i>".format(station_df['NAME'][i])).add_to(arrest_map)

# Map only station we are looking at

station_df = station_df[station_df['NAME']==station_to_look_up]
station_df = station_df.reset_index(drop=True)
print(station_df)

for i in range(1,len(station_df)):
    folium.Marker([station_df['LAT'][i], station_df['LON'][i]], popup="<i>{}</i>".format(station_df['NAME'])).add_to(arrest_map)

# Maps arrests set range(size for all)

for i in range(1000):#from index of 0 to coord
    folium.Marker([DISTANCE_TO_df['Latitude_y'][i], DISTANCE_TO_df['Longitude_y'][i]], popup="<i>{}</i>".format(str(DISTANCE_TO_df['ARREST_DATE_y'][i])+":"+str(DISTANCE_TO_df['PD_DESC_x'][i])),icon=folium.Icon(color='red')).add_to(arrest_map)

html_name= station_name.replace('.csv','')+".html"
arrest_map.save(html_name) #Saves map to html file
"""
