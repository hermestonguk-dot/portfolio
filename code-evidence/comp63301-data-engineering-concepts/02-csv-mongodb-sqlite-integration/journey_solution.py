"""
This is a stub for COMP63301 Coursework 04.
Do not edit or delete any lines given in this file that are marked with a "(s)".
(you can move them to different lines as long as you do not change the overall structure)

Use the test zone at the bottom of this stub to test your code locally.

Place your code below the comments marked "#Your code here" and before the next method.

Remove the comments from the lines that save the CSV files to enable saving the files for use in subsequent questions.

Each method is documented to explain what functionality is to be placed within it.

NOTE: You can create as many more methods as you need. However, you need to add 
self as a parameter of the new method and to call it with the prefix self.name 

EXAMPLE:

    def clean(self,result):
        cleaned = [item[0] for item in result]
        return cleaned

"""

import pandas as pd#(s)
import os#(s)
from sklearn.preprocessing import OneHotEncoder#(s)

class Journey:#(s)
 
    # ---Section 1 --- #
 
    #(Question 1)
    def prepare_csv_data(self,df_1083,df_1415):#(s) 
        """
              
                Use the schema of the CSV file for Site 1083 as the reference schema. Standardise the column names, data types, and formats in the Site 1415 CSV file to match the reference schema. Ensure consistency in naming conventions, date/time formats, and categorical values.
                Merge the harmonised datasets from both sites into a single CSV file. The integrated file should contain all observations from both sites, structured according to the reference schema.
    
                Returns: Tuple: [('Site ID', 'Date', 'Lane', 'Lane Name', 'Direction', 'Direction Name', 'Class Scheme', 'Class', 'Class Name', 'Length (m)', 'Headway (s)', 'Gap (s)', 'Speed (mph)', 'Vehicle Id'), 
                                   (2005, '15/03/2020 14:25:07.123', 3, 'SB_OUT', 2, 'South', 2042, 4, 'Van', 5.8, nan, 1.789, 52.678, 9999),
                                   (2505, '03/08/2020 08:25:09.111', 3, 'SB_OUT', 2, 'South', 2042, 4, 'CAR', 5.8, nan, 0.809, 13.243, 8888), ...]
        
        """
        answers_dir = os.path.join(os.path.dirname(__file__), "student_answer_csvs")#(s)
        os.makedirs(answers_dir, exist_ok=True)#(s)
        output_path = os.path.join(answers_dir, "question1.csv")#(s)

        #Your code here

        df_1083["Date"] = pd.to_datetime(df_1083["Date"])
        df_1083["Date"] = df_1083["Date"].dt.strftime("%d/%m/%Y %H:%M:%S.%f").str[:-3]
        
        df_1415["Date"] = pd.to_datetime(df_1415["Date"])
        df_1415["Date"] = df_1415["Date"].dt.strftime("%d/%m/%Y %H:%M:%S.%f").str[:-3]
        
        


        #Your code here        
        df_merged = pd.concat([df_1083,df_1415], ignore_index=True)
        df_merged = df_merged.sort_values("Date")
        
        #  Remove comment from next line to save the dataframe as a CSV file for use in the next question.
        df_merged.to_csv(output_path, index=False)#(s)
        return_list = list(df_merged.head(10).itertuples(index=False, name=None))
        return_list.insert(0, tuple(df_merged.columns))
        #print(return_list)
        return return_list

    #(Question 2)
    def prepare_csv_json_data(self,df_q1,collection):#(s)
        """
                Integrate the traffic and weather datasets. A key for integration needs to be identified for this purpose. Ensure both fields (used as key for the integration) are properly formatted and aligned in terms of date/time granularity.
                To enable integration, check whether aggregation of any of the involved datasets needs to be performed and perform it, if necessary. 
                
                For each time interval:
                    Compute averages for all relevant numeric columns (e.g., Headway, Gap, Speed).
                    Replace individual vehicle-level data with these aggregated values.
                    Count the number of vehicles and include this as a Vehicle Count column.

                Harmonise categorical values where necessary:
                For example, merge directional values such as ‘North’ and ‘Northeast’ into a single category called ‘NorthernQuadrant’, and assign it a code of ‘1’.
        
                Returns: Tuple: [ ('local_time', 'StartSite ID', 'EndSite ID', 'Direction', 'Direction Name', 'AVG Headway (s)', 'AVG Gap (s)', 'AVG Speed (mph)', 'Vehicle Count', 'city', 'position', 'weather_conditions', 'temperature', 'pressure', 'wind_speed', 'humidity'),
                                 ('01/01/2020 00:00:00.000', 1000, 2000, 1, 'NorthernQuadrant', 15.500000, 14.700000, 39.876543, 500, 'SampleCity', '16.2228,-1.2150', 'Clear', 10.0, 1012, 4.0, 65), 
                                 ('03/08/2018 18:20:03.000', 1111, 1515, 1, 'NorthQuadrant', 28.316009, 28.118414, 36.563302, 516, 'Leicester', '11.4808,-2.2300', 'Snow', 2.6, 992, 0.7, 77), ,...]
        """
        answers_dir = os.path.join(os.path.dirname(__file__), "student_answer_csvs")#(s)
        os.makedirs(answers_dir, exist_ok=True)#(s)
        output_path = os.path.join(answers_dir, "question2.csv")#(s) 

        
        #Your code here
        mongo = pd.DataFrame(list(collection.find()))

        mongo["local_time"] = pd.to_datetime(mongo["local_time"])
        mongo = mongo.sort_values("local_time")

        #normalizing Date to every hour to be used as key
        df_q1["Date"] = pd.to_datetime(df_q1["Date"], dayfirst=True, errors="coerce")
        df_q1["local_time"] = df_q1["Date"].dt.floor('h')
        
        hour_bin = (df_q1.groupby(["local_time","Direction"], as_index=False)
                        .agg({
                            "Headway (s)": "mean",
                            "Gap (s)": "mean",
                            "Speed (mph)": "mean",
                        }))
        vehicle_bin = (df_q1.groupby(["local_time","Direction"], as_index=False).size().rename(columns={"size": "Vehicle Count"}))


        df_merged = (hour_bin.merge(vehicle_bin, on=["local_time","Direction"])
                            .rename(columns={
                                "Headway (s)": "AVG Headway (s)",
                                "Gap (s)": "AVG Gap (s)",
                                "Speed (mph)": "AVG Speed (mph)",
                            }))

        df_merged[["AVG Headway (s)","AVG Gap (s)","AVG Speed (mph)"]] = df_merged[["AVG Headway (s)","AVG Gap (s)","AVG Speed (mph)"]].round(6)

        df_merged.insert(1,"StartSite ID", "1083")
        df_merged.insert(2,"EndSite ID", "1415")
        df_merged["Direction"] = 1
        df_merged["Direction Name"] = "NorthernQuadrant"

        df_merged = df_merged.merge(       
            mongo,        
            on="local_time",    
            how="left"    
        )
        
        #df_merged.insert(2,"EndSite ID", "1415")
        df_merged = df_merged.sort_values("local_time")

        df_merged = df_merged.drop(["_id"],axis=1)
        expected_cols = [
            "local_time",
            "StartSite ID",
            "EndSite ID",
            "Direction",
            "Direction Name",
            "AVG Headway (s)",
            "AVG Gap (s)",
            "AVG Speed (mph)",
            "Vehicle Count",
            "city",
            "position",
            "weather_conditions",
            "temperature",
            "pressure",
            "wind_speed",
            "humidity"
        ]
        df_merged = df_merged[expected_cols]
        
        df_merged["local_time"] = df_merged["local_time"].dt.strftime("%d/%m/%Y %H:%M:%S.%f").str[:-3]
  
                               
                        
       #  Remove comment from next line to save the dataframe as a CSV file for use in the next question.
        df_merged.to_csv(output_path, index=False)#(s)

        
        #Your code here
        return_list = list(df_merged.head(10).itertuples(index=False, name=None))
        return_list.insert(0, tuple(df_merged.columns))
        #print(return_list)
        return return_list

    #(Question 3)
    def prepare_csv_relational_data(self,conn,df_q2):#(s) 
        """
            Extract the relevant site and distance data from the SQLite database, specifically for the route from origin site 1083 to destination site 1415. It is worth pointing out that the distance data values appearing in the database are given in km.
            Join this data with the CSV file resulting from Task 2. The result should be a single CSV file that includes all previously integrated traffic and weather data, along with the distance between the two sites.
         
            Returns: Tuple: [('local_time', 'StartSite ID', 'EndSite ID', 'Distance', 'Direction', 'Direction Name', 'AVG Headway (s)', 'AVG Gap (s)', 'AVG Speed (mph)', 'Vehicle Count', 'city', 'position', 'weather_conditions', 'temperature', 'pressure', 'wind_speed', 'humidity'),
                              ('01/01/2020 00:00:00.000', 1000, 2000, 2.500000, 1, 'NorthQuadrant', 15.000000, 14.800000, 40.123456, 500, 'SampleCity', '00.0000,00.0000', 'Clear', 10.0, 1010, 5.0, 60),
                              ('09/15/2019 14:30:00.000', 1157, 1389, 2.034512, 0, 'EasternQuadrant', 29.782145, 29.615907, 35.482199, 472, 'Leeds', '53.8008,-1.5491', 'Rain', 3.1, 987, 0.5, 81), ....]
        """
        answers_dir = os.path.join(os.path.dirname(__file__), "student_answer_csvs")#(s)
        os.makedirs(answers_dir, exist_ok=True)#(s)
        output_path = os.path.join(answers_dir, "question3.csv")#(s) 

        #Your code here
        df_dist = pd.read_sql_query("SELECT * FROM DistanceBetweenSites", conn)
        
        df_dist = df_dist[(df_dist["StartSiteID"] == 1083) & (df_dist["EndSiteID"] == 1415) ]
        
        df_dist["Distance"] =  (df_dist["Distance"] * 0.621371).round(6)
        df_dist.rename(columns={"StartSiteID": "StartSite ID", "EndSiteID": "EndSite ID"}, inplace=True)
        
        df_merged = pd.merge(
            df_q2,
            df_dist[["StartSite ID", "EndSite ID", "Distance"]],
            on=["StartSite ID", "EndSite ID"],
            how="left"
        )
        expected_cols = [
            "local_time",
            "StartSite ID",
            "EndSite ID",
            "Distance",
            "Direction",
            "Direction Name",
            "AVG Headway (s)",
            "AVG Gap (s)",
            "AVG Speed (mph)",
            "Vehicle Count",
            "city",
            "position",
            "weather_conditions",
            "temperature",
            "pressure",
            "wind_speed",
            "humidity"
        ]
        df_merged = df_merged[expected_cols]

        df_merged["local_time"] = pd.to_datetime(df_merged["local_time"], dayfirst=True, errors="coerce")
        df_merged = df_merged.sort_values("local_time")
        df_merged["local_time"] = df_merged["local_time"].dt.strftime("%d/%m/%Y %H:%M:%S.%f").str[:-3]

        #  Remove comment from next line to save the dataframe as a CSV file for use in the next question.
        df_merged.to_csv(output_path, index=False, float_format="%.6f")#(s)

        #Your code here


        return_list = [tuple(df_merged.columns)] + list(df_merged.head(10).itertuples(index=False, name=None))
        #print(return_list)
        return return_list
        

    #(Question 4)
    def prepare_final_data(self,df_q3,JT_archive):#(s) 
        """
            Join the CSV file from Question 3 with the journey time archive file. A join key needs to be identified for this purpose.
            The result should be a single CSV file that is going to be build that includes all previously integrated traffic, weather, and site data, along with the corresponding Journey Time values.
            
            Returns: Tuple:  [('local_time', 'StartSite ID', 'EndSite ID', 'Journey Time', 'Distance', 'Direction', 'Direction Name', 'AVG Headway (s)', 'AVG Gap (s)', 'AVG Speed (mph)', 'Vehicle Count', 'city', 'position', 'weather_conditions', 'temperature', 'pressure', 'wind_speed', 'humidity'),
                             ('01/01/2020 00:00:00.000', 1000, 2000, 2.500000, 1, 'NorthQuadrant', 15.000000, 14.800000, 40.123456, 500, 'SampleCity', '00.0000,00.0000', 'Clear', 10.0, 1010, 5.0, 60, 0.1000),
                             ('11/23/2019 09:15:00.000', 1128, 1392, 2.045783, 0, 'WesternQuadrant', 21.487923, 21.264812, 37.104827, 812, 'Liverpool', '53.4084,-2.9916', 'Light Snow', -0.8, 1004, 4.3, 68, 0.0472), .... ]
        """
        answers_dir = os.path.join(os.path.dirname(__file__), "student_answer_csvs")#(s)
        os.makedirs(answers_dir, exist_ok=True)#(s)
        output_path = os.path.join(answers_dir, "question4.csv")#(s) 

        #Your code here
        df_q3["local_time"] = pd.to_datetime(df_q3["local_time"], dayfirst=True, errors="coerce")
        JT_archive["local_time"] = pd.to_datetime(JT_archive["local_time"], dayfirst=True, errors="coerce")
    
        df_q3["local_time"] = df_q3["local_time"].dt.floor("s")
        JT_archive["local_time"] = JT_archive["local_time"].dt.floor("s")
        df_merged = pd.merge(df_q3,JT_archive, on=["local_time", "StartSite ID", "EndSite ID"], how = "left" )

        expected_cols = [
            "local_time",
            "StartSite ID",
            "EndSite ID",
            "Journey Time",
            "Distance",
            "Direction",
            "Direction Name",
            "AVG Headway (s)",
            "AVG Gap (s)",
            "AVG Speed (mph)",
            "Vehicle Count",
            "city",
            "position",
            "weather_conditions",
            "temperature",
            "pressure",
            "wind_speed",
            "humidity"
        ]
        df_merged = df_merged[expected_cols]

        
        df_merged = df_merged.sort_values("local_time")
        df_merged["local_time"] = df_merged["local_time"].dt.strftime("%d/%m/%Y %H:%M:%S.%f").str[:-3]
        

        #  Remove comment from next line to save the dataframe as a CSV file for use in the next question.
        df_merged.to_csv(output_path, index=False, float_format="%.6f")#(s)

        #Your code here
        return_list = [tuple(df_merged.columns)] + list(df_merged.head(10).itertuples(index=False, name=None))
        #print(return_list)
        return return_list
      
    #(Question 5)
    def result_and_encode_data(self,df_q4):#(s)
        """
         	Apply one-hot encoding to the weather_conditions column in your dataset. This process will automatically convert each unique weather condition into a separate binary feature, allowing the model to interpret categorical weather data numerically.
            After encoding, select the following columns as input features for your machine learning model.
         
            Returns: Tuple:( ('local_time', 'StartSite ID', 'EndSite ID', 'Distance', 'AVG Speed (mph)', 'AVG Headway (s)', 'AVG Gap (s)', 'Vehicle Count', 'city', 'position', 'temperature', 'pressure', 'wind_speed', 'humidity', 'Journey Time', 'weather_Clear', 'weather_Cloudy', 'weather_Fog', 'weather_Heavy Rain', 'weather_Light Rain', 'weather_Mist', 'weather_Partly Cloudy', 'weather_Snow'),
                            ('01/01/2020 00:00:00.000', 1000, 2000, 2.500000, 40.123456, 15.000000, 14.500000, 500, 'SampleCity', '00.0000,00.0000', 10.0, 1010, 5.0, 60, 0.1000, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                           ('10/12/2019 09:30:00.000', 1142, 1388, 2.037654, 37.214589, 21.583421, 21.302774, 796, 'Leeds', '53.8008,-1.5491', -0.9, 1003, 4.6, 67, 0.0483, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0), ..)

        """

        ohe = OneHotEncoder(sparse_output=False, drop=None) #(s)
        #Your code here
        encoded = ohe.fit_transform(df_q4[["weather_conditions"]])

        encoded_cols = [f"weather_{con}" for con in ohe.categories_[0]]

        df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df_q4.index)

        df_final = pd.concat([df_q4.drop(columns=["weather_conditions"]), df_encoded], axis=1)

        expected_cols = [
            "local_time",
            "StartSite ID",
            "EndSite ID",
            "Distance",
            "AVG Speed (mph)",
            "AVG Headway (s)",
            "AVG Gap (s)",
            "Vehicle Count",
            "city",
            "position",
            "temperature",
            "pressure",
            "wind_speed",
            "humidity",
            "Journey Time",
            "weather_Clear",
            "weather_Cloudy",
            "weather_Fog",
            "weather_Heavy Rain",
            "weather_Light Rain",
            "weather_Mist",
            "weather_Partly Cloudy",
            "weather_Snow",
        ]
        
        for col in expected_cols:
            if col not in df_final.columns:
                df_final[col] = 0.0

        df_final = df_final[expected_cols]
        
        df_final["local_time"] = pd.to_datetime(df_final["local_time"], dayfirst=True, errors="coerce")
        df_final = df_final.sort_values("local_time")
        df_final["local_time"] = df_final["local_time"].dt.strftime("%d/%m/%Y %H:%M:%S.%f").str[:-3]


        final_list = tuple([tuple(df_final.columns)] + list(
            df_final.head(10).itertuples(index=False, name=None)
        ))
        #print(final_list)
        return final_list

# ---Test Zone --- #

"""
Remove the comments below to test your solution locally. Once the solution is finalised, reapply the comments as they are currently.
"""

# from pymongo.mongo_client import MongoClient
# import sqlite3

# if __name__ == '__main__':

#     # Build paths to the CSV files in the parent folder
#     df_1083_path = os.path.join(os.path.dirname(__file__), "rawpvr_2018-02-01_28d_1083.csv")
#     df_1415_path = os.path.join(os.path.dirname(__file__), "rawpvr_2018-02-01_28d_1415.csv")

#     #You can place any ad-hoc testing here 
#     my_instance = Journey()
#     df_1083  = pd.read_csv(df_1083_path)
#     df_1415 = pd.read_csv(df_1415_path)

#     print("Question 1")
#     print(my_instance.prepare_csv_data(df_1083,df_1415))

#     # local mongoDB db, DO NOT CHANGE
#     uri = "mongodb://localhost:27017/"

#     # Create a new client and connect to the server
#     client = MongoClient(uri)

#     db = client["coursework4"]
#     collection = db["February2018"]

#     df_q1_path = os.path.join(os.path.dirname(__file__), "student_answer_csvs/question1.csv")
#     df_q1 = pd.read_csv(df_q1_path)

#     # print("Question 2")
#     print(my_instance.prepare_csv_json_data(df_q1,collection))

#     df_q2_path = os.path.join(os.path.dirname(__file__), "student_answer_csvs/question2.csv")
#     df_q2 = pd.read_csv(df_q2_path)

#     ORIGINAL_DB = "coursework4.db"
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     original_db_path = os.path.join(base_dir, ORIGINAL_DB)
#     conn = sqlite3.connect(original_db_path)

#     print("Question 3")
#     print(my_instance.prepare_csv_relational_data(conn,df_q2))

#     conn.close()

#     df_q3_path = os.path.join(os.path.dirname(__file__), "student_answer_csvs/question3.csv")
#     JT_archive_path = os.path.join(os.path.dirname(__file__), "JTarchive_1083to1415_Tues_Sun_00-23hours_01-28_Feb_2018.csv")

#     df_q3 = pd.read_csv(df_q3_path)
#     JT_archive = pd.read_csv(JT_archive_path)

#     print("Question 4")
#     print(my_instance.prepare_final_data(df_q3,JT_archive))

#     df_q4_path = os.path.join(os.path.dirname(__file__), "student_answer_csvs/question4.csv")

#     df_q4 = pd.read_csv(df_q4_path)

#     print("Question 5")
#     print(my_instance.result_and_encode_data(df_q4))





