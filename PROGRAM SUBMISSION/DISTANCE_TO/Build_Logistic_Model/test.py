from lib2to3.pgen2.pgen import DFAState
from turtle import color, distance
import pandas as pd
import numpy as np
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
Used https://gis.stackexchange.com/questions/130963/write-geojson-into-a-geojson-file-with-python
Used https://matthew-brett.github.io/teaching/string_formatting.html
Used https://favtutor.com/blogs/list-to-dataframe-python
Used https://www.pythontutorial.net/python-basics/python-check-if-file-exists/

Calculation
https://www.calculator.net/distance-calculator.html?type=3&la1=38.8976&lo1=-77.0366&la2=39.9496&lo2=-75.1503&ctype=dec&lad1=38&lam1=53&las1=51.36&lau1=n&lod1=77&lom1=2&los1=11.76&lou1=w&lad2=39&lam2=56&las2=58.56&lau2=n&lod2=75&lom2=9&los2=1.08&lou2=w&x=80&y=24#latlog
https://www.geeksforgeeks.org/get-minimum-values-in-rows-or-columns-with-their-index-position-in-pandas-dataframe/
"""
def make_df(file_name):
    df = pd.read_csv(file_name)
    #preform any adjustments here on df
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
    DISTANCE_TO_DIR = "DISTANCE_TO\\"
    file_exists =  os.path.exists(distance_df_name)
    if file_exists:
        distance_df = make_df(DISTANCE_TO_DIR+distance_df_name)
        station_df = distance_df.merge(arrest_df,how='left', left_on=['ARREST_KEY'], right_on=['ARREST_KEY'])
    else:
        exit(distance_df_name,'not found. Please generate with station_analysis.py')
    return station_df,distance_df_name

def look_up__Walking_Distance_Crime_Probabilities(arrest_df,station_name): #Matches up station name with distance df
    
    distance_df_name ="DISTANCE_TO_"+station_name+"_Walking_Distance_Crime_Probabilities.csv"
    DISTANCE_TO_DIR = "DISTANCE_TO\\"
    file_exists =  os.path.exists(DISTANCE_TO_DIR+distance_df_name)
    if file_exists:
        distance_df = make_df(distance_df_name)
        station_df = distance_df.merge(arrest_df,how='left', left_on=['ARREST_KEY'], right_on=['ARREST_KEY'])
    else:
        exit(distance_df_name,'not found. Please generate with station_analysis.py')
    return station_df,distance_df_name
"""
    Set Up
"""
def set_up():
    station_df =make_df('Station_GEODATA.csv')
    arrest_df  =make_df('NYPD_Arrest_Data__Year_to_Date_.csv')
    arrest_df['ARREST_DATE'] = pd.to_datetime(arrest_df['ARREST_DATE'])
    arrest_df.sort_values(by='ARREST_DATE')
    return station_df,arrest_df

"""
    Now that we have probabilities from our stations we can use them to fit models. 
"""

station_df,arrest_df = set_up()

station_to_look_up ='Grand Central - 42nd St'


DISTANCE_TO_df,DISTANCE_TO_filename= look_up_DISTANCE_TO(arrest_df,station_to_look_up)
DISTANCE_TO_colname =DISTANCE_TO_filename.replace('.csv','')

