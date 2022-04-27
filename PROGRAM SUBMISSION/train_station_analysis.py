
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
import pathlib
from geopy import distance
import sklearn
from sklearn.cluster import KMeans

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
Used https://stackoverflow.com/questions/3964681/find-all-files-in-a-directory-with-extension-txt-in-python

Calculation
https://www.calculator.net/distance-calculator.html?type=3&la1=38.8976&lo1=-77.0366&la2=39.9496&lo2=-75.1503&ctype=dec&lad1=38&lam1=53&las1=51.36&lau1=n&lod1=77&lom1=2&los1=11.76&lou1=w&lad2=39&lam2=56&las2=58.56&lau2=n&lod2=75&lom2=9&los2=1.08&lou2=w&x=80&y=24#latlog
https://www.geeksforgeeks.org/get-minimum-values-in-rows-or-columns-with-their-index-position-in-pandas-dataframe/

Title: Crime on Rails
URL: https://christo480.github.io/DATASCI%20PROJECT/Data_Sci_project.html
"""
"""
HW 10 SKLEARN DEPENDENCIES:
"""
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import *
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

"""FUNCTIONS"""

def make_df(file_name):
    
    file_exists =  os.path.exists(file_name)
    if file_exists:
        df = pd.read_csv(file_name)
    else:
        exit(file_name+' not found.')
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
    DISTANCE_TO_DIR = "DISTANCE_TO/"
    path = DISTANCE_TO_DIR+distance_df_name
    file_exists =  os.path.exists(path)
    if file_exists:
        distance_df = make_df(path)
        station_df = distance_df.merge(arrest_df,how='left', left_on=['ARREST_KEY'], right_on=['ARREST_KEY'])
    else:
        exit(distance_df_name+'not found. Please generate with station_analysis.py')
    return station_df,distance_df_name
"""
    look_up__Walking_Distance_Crime_Probabilities(arrest_df,station_name):
    This function takes two inputs:
        -arrest_df: data frame with arrest data 
        -station_name: Used to make the appropiate file name. Generated files are of form "DISTANCE_TO_"+station_name+"_Walking_Distance_Crime_Probabilities.csv"
        The Walking_Distance_Crime_Probabilities file  is then used to merge with the arrest df before being returned along with the name of the file used.
    
"""
def look_up__Walking_Distance_Crime_Probabilities(arrest_df,station_name): #Matches up station name with distance df
    
    distance_df_name ="DISTANCE_TO_"+station_name+"_Walking_Distance_Crime_Probabilities.csv"
    DISTANCE_TO_DIR = "DISTANCE_TO/"
    path = DISTANCE_TO_DIR+distance_df_name
    file_exists =  os.path.exists(path)
    if file_exists:
        distance_df = make_df(path)
        station_df = distance_df.merge(arrest_df,how='left', left_on=['ARREST_KEY'], right_on=['ARREST_KEY'])
    else:
        exit(distance_df_name,'not found. Please generate with station_analysis.py')
    return station_df,distance_df_name
"""
    set_up():
        Before we can do anything with this data we need the coordinates of each station stored in 'Station_GEODATA.csv
        and the arrest data in NYPD_Arrest_Data__Year_to_Date_.csv'
        returns both as station_df and arrest_df and return them in that order.

"""
def set_up():
    station_df =make_df('Station_GEODATA.csv')
    arrest_df  =make_df('NYPD_Arrest_Data__Year_to_Date_.csv')
    arrest_df['ARREST_DATE'] = pd.to_datetime(arrest_df['ARREST_DATE'])
    arrest_df.sort_values(by='ARREST_DATE')
    return station_df,arrest_df
"""
    list_files(Directory, print='yes'):
        Takes a Directory and prints each file in it ending with .csv.
        Returns list of all files it has with calculated DISTANCE_TO
"""
def list_files(Directory, print='yes'):
    usable_files = list()
    for file in os.listdir(Directory):
        if file.startswith('DISTANCE_TO') and file.endswith(".csv"):
            if(print=='yes'):
                print(os.path.join(DISTANCE_TO_DIR, file))
            usable_files.append(file)
    return usable_files
"""
    Function used to calculate distances bettween arrests and a given station. Originally used on entire data set at once...
    [INCOMPLETE]
"""
def generate_DISTANCE_TO(station_df,arrest_df,station_name):
    
    """
    for i in range(106,len(station_df)):

    distance_to_i = list()
    arrest_number = list()
    """
    station_locale= (station_df['LAT'].where(station_df['NAME']==station_name).values[0],station_df[['LON']].where(station_df['NAME']==station_name).values[0])
    distance_col_name= station_df['NAME'].where(station_df['NAME']==station_name).values[0]
    distances_to_station = list()
    #Query name of the file
    distance_col_name= 'DISTANCE_TO_'+str(distance_col_name)
    print(distance_col_name)
    for j in range(len(arrest_df)):#For every arrest in the data frame
        arrest_locale = (arrest_df['Latitude'][j],arrest_df['Longitude'][j])
        distances_to_station.append(distance.distance(arrest_locale,station_locale))
        
        print("Distance to ",arrest_df['ARREST_KEY'][j],": ",distance.distance(arrest_locale,station_locale))
    
    #arrest_df[distance_col_name] = distance_to_i
    temp_df = ['ARREST_KEY',distance_col_name]
    arrest_df[temp_df].to_csv(distance_col_name+".csv") 
  
def get_close_match(raw_input,list):
    for i in range(len(list)):
        if(raw_input in list[i]):
            return list[i][(list[i].find('TO_')+3):(list[i].find('.csv'))]
    
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
"""
    catagorize_crime(row):
    I tried my best to sort out which crimes were violent based on them being "rape and sexual assault, robbery, assault and murder."
"""
def catagorize_crime(row):
    if(row=='ASSAULT 3 & RELATED OFFENSES' or row=='FELONY ASSAULT' or row=='SEX CRIMES' or row == 'MURDER & NON-NEGL. MANSLAUGHTE' or row == 'RAPE' or row =='\"HOMICIDE-NEGLIGENT,UNCLASSIFIE\"' or row=='FELONY SEX CRIMES'): #Violent Crimes
        return 1
    else:
        return 0
"""
    map_walking_distance(station_df,arrests_with_walking_distance):
    Creates map of all arrests in arrests_with_walking_distance and all stations in station df
"""
def map_walking_distance(station_df,arrests_with_walking_distance):
    arrest_map = folium.Map(location=[40.7128,-74.0060],zoom_start=20)
    limit=arrests_with_walking_distance.reset_index()
    #Graph all stations
    for i in range(len(station_df)):
        folium.Marker([station_df['LAT'][i], station_df['LON'][i]], popup="<i>{}</i>".format(station_df['NAME'][i])).add_to(arrest_map)
    print(limit.columns)
    print(limit)
    #print(str(limit.at[1,'Latitude']), str(limit.at[1,'Longitude']))
    #Graph only the arrests within distance
    for i in range(len(limit)):#from index of 0 to coord
        if(limit.at[i,'is_violent_crime?'] == 1): #Violent crimes are darkpurple
            folium.Marker([limit.at[i,'Latitude'], limit.at[i,'Longitude']], popup="<i>{}</i>".format(str(limit.at[i,'ARREST_DATE'])+","+str(limit.at[i,'PD_DESC'])),icon=folium.Icon(color='darkpurple')).add_to(arrest_map)
        else:
            folium.Marker([limit.at[i,'Latitude'], limit.at[i,'Longitude']], popup="<i>{}</i>".format(str(limit.at[i,'ARREST_DATE'])+","+str(limit.at[i,'PD_DESC'])),icon=folium.Icon(color='red')).add_to(arrest_map)
        #folium.Marker([limit['Latitude'][i], limit['Longitude'][i]], popup="<i>{}</i>".format(str(limit['ARREST_DATE'][i])+","+str(limit['PD_DESC'][i])),icon=folium.Icon(color='red')).add_to(arrest_map)

    html_name= station_name.replace('.csv','')+".html"
    arrest_map.save(html_name) #Saves map to html file
"""
    map_walking_distance_with_KMEANS(station_df,arrests_with_walking_distance):
    station_df

    Creates map of all arrests in arrests_with_walking_distance and all stations in station df with added KMeans clustered point
"""
def map_walking_distance_with_KMEANS(station_df,arrests_with_walking_distance, centroid, label):
    arrest_map = folium.Map(location=[40.7128,-74.0060],zoom_start=20)
    limit=arrests_with_walking_distance.reset_index()
    # K MEANS Centroid
    folium.Marker([centroid[0][0], centroid[0][1]], popup="<i>{}</i>".format(label),icon=folium.Icon(color='green')).add_to(arrest_map)
    
    #Graph all stations
    for i in range(len(station_df)):
        folium.Marker([station_df['LAT'][i], station_df['LON'][i]], popup="<i>{}</i>".format(station_df['NAME'][i])).add_to(arrest_map)
    """print(limit.columns)
    print(limit)"""
    #print(str(limit.at[1,'Latitude']), str(limit.at[1,'Longitude']))
    #Graph only the arrests within distance
    for i in range(len(limit)):#from index of 0 to coord
        if(limit.at[i,'is_violent_crime?'] == 1): #Violent crimes are darkpurple
            folium.Marker([limit.at[i,'Latitude'], limit.at[i,'Longitude']], popup="<i>{}</i>".format(str(limit.at[i,'ARREST_DATE'])+","+str(limit.at[i,'PD_DESC'])),icon=folium.Icon(color='darkpurple')).add_to(arrest_map)
        else:
            folium.Marker([limit.at[i,'Latitude'], limit.at[i,'Longitude']], popup="<i>{}</i>".format(str(limit.at[i,'ARREST_DATE'])+","+str(limit.at[i,'PD_DESC'])),icon=folium.Icon(color='red')).add_to(arrest_map)
        #folium.Marker([limit['Latitude'][i], limit['Longitude'][i]], popup="<i>{}</i>".format(str(limit['ARREST_DATE'][i])+","+str(limit['PD_DESC'][i])),icon=folium.Icon(color='red')).add_to(arrest_map)

    html_name= station_name.replace('.csv','')+"_WITH_KMEANS.html"
    arrest_map.save(html_name) #Saves map to html file

"""
run_station_prog_init():
Provides user interface for running analysis.
Sets options for analysis. Can be automated in variant.
returns station_df,arrest_df
"""
def run_station_prog_init():
    station_df,arrest_df = set_up()
    DISTANCE_TO_DIR = "DISTANCE_TO"

    DISTANCE_TO_files = list_files(DISTANCE_TO_DIR,print='no')
    print("Station Analysis Program")
    print("Show stations available for analysis? (y/n)")
    choice = input()
    if(choice=='y' or choice=='Y'):
        for file in DISTANCE_TO_files:
            print(file)

    #station_to_look_up ='Grand Central - 42nd St'; Grand Central Station is the example used when testing

    print('Enter station you want to analyze:')
    station_to_look_up =input()
    station_to_look_up_filename ="DISTANCE_TO_"+station_to_look_up+".csv"

    #print(str(station_df['NAME'].where(station_df['NAME']==station_to_look_up)))
    in_files=(((station_df['NAME'].where(station_df['NAME']==station_to_look_up)).dropna().count())) #checks if name matches with station in Geo Data
    #print(in_files)
    #if(DISTANCE_TO_DIR+station_to_look_up_filename in DISTANCE_TO_files): 
    possible_DISTANCE_TO =get_close_match(station_to_look_up,DISTANCE_TO_files)
    if(possible_DISTANCE_TO!=None):
        print('Distances available for this station:')
        station_to_look_up= possible_DISTANCE_TO
    else:#Make sure all stations we don't have data for cause an exit condition
        exit("Data for "+station_to_look_up+" is unavailable.")

    print("Show Graph?(y/n)")
    show_graph = input()
    if(show_graph=='y'):
        show_graph= True
    else:
        show_graph= False
    """
        Portion of Driver Code to allow lookups
    elif(in_files>0):
        print('Distances not available for this station but station data is available. Generating... ')
        #If we have station data we generate it..

        NOTE:Testing Portion: where we check searches on input are beng correctly preformed
        print((station_df['NAME'].where(station_df['NAME']==station_to_look_up)).dropna())
        #print(station_df['LAT'].where(station_df['NAME']==station_to_look_up).values[0])
        station_locale= (station_df['LAT'].where(station_df['NAME']==station_to_look_up).values[0],station_df['LON'].where(station_df['NAME']==station_to_look_up).values[0])
        print(station_locale)
        if(station_locale[0]=='nan' or station_locale[1]=='nan'):
            exit("Data for "+station_to_look_up+" is unavailable.")
        generate_DISTANCE_TO(station_df,arrest_df,station_to_look_up)
    """
    return station_df,arrest_df,station_to_look_up,station_to_look_up_filename,DISTANCE_TO_DIR,show_graph

"""
run_station_prog(): [INCOMPLETE]
Automated version of running program to set up station
data.
station: valid station name
Sets options for analysis. Can be automated in variant.
returns station_df,arrest_df
"""
def run_station_prog(station,choice = 'n',show_graph='n'):
    station_df,arrest_df = set_up()
    DISTANCE_TO_DIR = "DISTANCE_TO"

    DISTANCE_TO_files = list_files(DISTANCE_TO_DIR,print='no')
    """print("Station Analysis Program")
    print("Show stations available for analysis? (y/n)")
    choice = input()"""
    if(choice=='y' or choice=='Y'):
        for file in DISTANCE_TO_files:
            print(file)

    #station_to_look_up ='Grand Central - 42nd St'; Grand Central Station is the example used when testing

    #print('Enter station you want to analyze:')
    station_to_look_up =station
    station_to_look_up_filename ="DISTANCE_TO_"+station_to_look_up+".csv"

    #print(str(station_df['NAME'].where(station_df['NAME']==station_to_look_up)))
    in_files=(((station_df['NAME'].where(station_df['NAME']==station_to_look_up)).dropna().count())) #checks if name matches with station in Geo Data
    #print(in_files)
    #if(DISTANCE_TO_DIR+station_to_look_up_filename in DISTANCE_TO_files): 
    possible_DISTANCE_TO =get_close_match(station_to_look_up,DISTANCE_TO_files)
    if(possible_DISTANCE_TO!=None):
        #print('Distances available for this station:')
        station_to_look_up= possible_DISTANCE_TO
    else:#Make sure all stations we don't have data for cause an exit condition
        exit("Data for "+station_to_look_up+" is unavailable.")

    print("Show Graph?(y/n)")
    show_graph = input()
    if(show_graph=='y'):
        show_graph= True
    else:
        show_graph= False
    """
        Portion of Driver Code to allow lookups
    elif(in_files>0):
        print('Distances not available for this station but station data is available. Generating... ')
        #If we have station data we generate it..

        NOTE:Testing Portion: where we check searches on input are beng correctly preformed
        print((station_df['NAME'].where(station_df['NAME']==station_to_look_up)).dropna())
        #print(station_df['LAT'].where(station_df['NAME']==station_to_look_up).values[0])
        station_locale= (station_df['LAT'].where(station_df['NAME']==station_to_look_up).values[0],station_df['LON'].where(station_df['NAME']==station_to_look_up).values[0])
        print(station_locale)
        if(station_locale[0]=='nan' or station_locale[1]=='nan'):
            exit("Data for "+station_to_look_up+" is unavailable.")
        generate_DISTANCE_TO(station_df,arrest_df,station_to_look_up)
    """
    return station_df,arrest_df,station_to_look_up,station_to_look_up_filename,DISTANCE_TO_DIR,show_graph

""" FUNTIONS END"""


"""DRIVER CODE"""

station_df,arrest_df,station_to_look_up,station_to_look_up_filename,DISTANCE_TO_DIR,show_graph = run_station_prog_init()

DISTANCE_TO_df,DISTANCE_TO_filename= look_up_DISTANCE_TO(arrest_df,station_to_look_up)

DISTANCE_TO_colname =DISTANCE_TO_filename.replace('.csv','')

print(DISTANCE_TO_df)
#print(arrest_df)

"""
We need to clean the 'DISTANCE_TO' cols as they will have the numbers in string format
we do this my removing last two characters from each member and converting to floats
(Turn into function clean_distance_data())

"""
"""
Creates map centered on New York for plotting
"""
#arrest_map = folium.Map(location=[40.7128,-74.0060],zoom_start=20) #https://python-visualization.github.io/folium/quickstart.html#GeoJSON/TopoJSON-Overlays

DISTANCE_TO_df[DISTANCE_TO_colname] = DISTANCE_TO_df[DISTANCE_TO_colname].apply(lambda x : x[0:-2])
DISTANCE_TO_df[DISTANCE_TO_colname] = pd.to_numeric(DISTANCE_TO_df[DISTANCE_TO_colname])



#DISTANCE_TO_df['DISTANCE_TO_'+station_to_look_up] = pd.to_numeric(DISTANCE_TO_df['DISTANCE_TO_'+station_to_look_up])
"""
Now that our data is clean we can look at the clustering of arrests around the station. We can limit our arrests to those within walking distance(0.5 kms). 
"""

"""
Debugging: Used to check the types of crimes to account for in catagorize_crime()
print(DISTANCE_TO_df.columns)
print(DISTANCE_TO_df['OFNS_DESC'].value_counts())
offense_value_counts =DISTANCE_TO_df['OFNS_DESC'].value_counts()"""

arrest_distance_catagories= ["Within 0.5 but more than 0.4","Within 0.4 but more than 0.3","Within 0.3 but more than 0.2","Within 0.2 but more than 0.1","Within 0.1 "]

arrests_with_walking_distance = limit_col(DISTANCE_TO_df,DISTANCE_TO_colname,0.5,0)

#Add new features here

#Assign Distance Catagory
arrests_with_walking_distance['Distance_Catagory'] = arrests_with_walking_distance[DISTANCE_TO_colname].apply(catagorize_distances)
arrest_distance_catagories_count = list()

arrests_with_walking_distance['is_violent_crime?'] = arrests_with_walking_distance['OFNS_DESC'].apply(catagorize_crime)
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


"""print("Count:",len(arrests_with_walking_distance))
value_counts = arrests_with_walking_distance['Distance_Catagory'].value_counts()
print(arrests_with_walking_distance['Distance_Catagory'].value_counts())
print(value_counts[0])

print(arrests_with_walking_distance)"""

arrests_with_walking_distance.to_csv(DISTANCE_TO_colname+"_Walking_Distance_Crime_Probabilities.csv")
station_name =DISTANCE_TO_colname+"_Walking_Distance_Crime_Probabilities.csv"

map_walking_distance(station_df,arrests_with_walking_distance)

"""
NOTE: Failed attempt to make probablistic inference of arrest based on proximity to station
Now that we have probabilities from our stations we can use them to fit models. We can fit a logistic model by making a 
frequency table from frequency of arrests in each range. Edit: Distance catagories is unsuitable for a logistical model its better to use other features

distances_from_station = arrests_with_walking_distance[DISTANCE_TO_colname]
probabilites = arrests_with_walking_distance['Probability']

sns.scatterplot(x=distances_from_station ,y=probabilites)



#plt.show()
plt.clf()
X_train, X_test, y_train, y_test = train_test_split(distances_from_station.array, probabilites.array, test_size=0.2)"""

# K means clustering based on coordinates
#https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
"""
    We can also use the original coordinates to calculate a center for the arrest clustering within 0.5kms of the station.
    K means modeling can be used to determine the center of the arrests. 
    
    This can be used to check if clustering occurs at the station
    We can then display the degree to which a station accounts for all the arrests around it.
    
    Lets add the calculated K-MEANs center to the map:
"""
x_coord = arrests_with_walking_distance['Latitude']
y_coord = arrests_with_walking_distance['Longitude']
coord = list(zip(x_coord,y_coord))

model = KMeans(n_clusters=1)
model.fit(coord)
centroid =  model.cluster_centers_
print("KMEANS Clustering Coordinate: (",centroid[0][0],centroid[0][1],")")
plt.scatter(centroid[0][0],centroid[0][1], s=200,c='r',)
if(show_graph):
    plt.show()
plt.clf() # Clears the graph

map_walking_distance_with_KMEANS(station_df,arrests_with_walking_distance,centroid,str(centroid))

station_locale = (station_df['LAT'].where(station_df['NAME']==station_to_look_up).dropna().reset_index(drop=True)[0]),((station_df['LON'].where(station_df['NAME']==station_to_look_up).dropna().reset_index(drop=True)[0]))
clustered_center =  (centroid[0][0],centroid[0][1])
distance_from_cluster =distance.distance(clustered_center,station_locale)
print("Discrepency bettween station and KMEANS Cluster:",distance_from_cluster)

"""
Violent Crime Classifier: By using the distance from the station and the classifier 'is_violent_crime?' we can make a random forest model for violent crime.

"""
# https://www.statology.org/plot-logistic-regression-in-python/
distances =  arrests_with_walking_distance[DISTANCE_TO_colname]
#distances = distances.array.reshape(1, -1)
is_violent_crime = arrests_with_walking_distance['is_violent_crime?']
#is_violent_crime = is_violent_crime.array.reshape(1, -1)


if(show_graph):
    plt.show()
#model = LogisticRegression()
#model.fit(distances,is_violent_crime)
X_train, X_test, y_train, y_test = train_test_split(distances.array, is_violent_crime.array, test_size=0.2)

fig = plt.figure()

model =LogisticRegression()
model.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))
plt.scatter(X_test,y_test, s=200,c='b',)
predicted = model.predict(X_test.reshape(-1, 1))
plt.scatter(X_test,predicted, s=200,c='r',)
if(show_graph):
    plt.show()
"""
# Uncomment to check what the model got right by hand
print(y_test)
print(predicted)"""
print("Score of Logistic Regression:",model.score(X_test.reshape(-1,1),np.ravel(y_test)))# Causes warning .ravel().reshape(-1,1)

plt.clf()
model =RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))
plt.scatter(X_test,y_test, s=200,c='b',)
predicted = model.predict(X_test.reshape(-1, 1))
plt.scatter(X_test,predicted, s=200,c='r',)
if(show_graph):
    plt.show()
"""
# Uncomment to check what the model got right by hand
print(y_test)
print(predicted)
"""
print("Score of Random Forrests:",model.score(X_test.reshape(-1,1),y_test.reshape(-1,1)))

#https://stackoverflow.com/questions/52996057/how-can-i-combine-two-lists-of-dimensions-to-a-list-of-points-in-python
#data = [list(pair) for pair in zip(distances, probabilites)]


#plt.plot(X_test,loss)



