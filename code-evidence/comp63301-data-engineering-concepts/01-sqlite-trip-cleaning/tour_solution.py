"""
This is a stub for COMP63301 Coursework 02.
Do not edit or delete any lines given in this file that are marked with a "(s)".
(you can move them to different lines as long as you do not change the overall structure)

Only the SQLite3 package should be imported for testing purposes; refrain from importing any other packages.

Use the test zone at the bottom of this stub to test your code locally.

Place your code below the comments marked "#Your code here" and before the next method.

Each method is documented to explain what functionality is to be placed within.

NOTE: You can create as many more methods as you need. However, you need to add 
self as a parameter of the new method and to call it with the prefix self.name 

EXAMPLE:

    def clean(self,result):
        cleaned = [item[0] for item in result]
        return cleaned

"""

class Tours:#(s)
 
    # ---Section 1 --- #
 
    #(Question 1)
    def replace_NaN_with_stop(self,conn):#(s)
        """
                a.	Identify all missing values in the end_location_name column, 
                        saving the trip_id associated with the first 10 rows with a missing value for end_location_name.

                b.	Replace each missing entry using each of the three strategies described above, 
                        one-at-a-time. You will need to restore each of the rows that had the value of end_location_name filled to its state, 
                        before its end_location_name value was filled (i.e., a missing end_location_name), before applying the next strategy.

                c.	For each strategy and for each of the first 10 replacements, 
                        compute and return the total number of occurrences of the associated, new, pair <start_location_name, end_location_name>.
                    
                Returns: List: [('trip_id', 'start_location_name', 'end_location_name', 'pair_count'), (263511, '80 Whitworth BM, Albuquerque, NY 87888, ILD', '898 Mountain Rd NW, Albuquerque, NM 87104, USA', 3) , (333333, '71 Whitworth NW, Albuquerque, NM 16161, USA', 'Stop St.', 3), .....]
        """
        cursor = conn.cursor()#(s)
        trip_id = []
        return_list = [('trip_id', 'start_location_name', 'end_location_name', 'pair_count')]
        
        #1.a identify trip_id
        cursor.execute("SELECT trip_id FROM trip WHERE end_location_name IS NULL or TRIM(end_location_name) = '' LIMIT 10")
        errors = cursor.fetchall()
        ids = [row[0] for row in errors]
        
        


        #strat 1
        conn.execute("SAVEPOINT s1")
    
        cursor.execute("""
        UPDATE trip
        SET end_location_name = 'Stop St.'
        WHERE end_location_name IS NULL or TRIM(end_location_name) = ''
        """)
    
        for i in ids:
            cursor.execute("SELECT start_location_name, end_location_name FROM trip WHERE trip_id = ?", (i,))
            pair = cursor.fetchall()
            
            cursor.execute(f"SELECT start_location_name, end_location_name, count(*) FROM trip WHERE '{pair[0][0]}' = start_location_name AND '{pair[0][1]}'  = end_location_name")
            count = cursor.fetchall()

            

            return_list.append((i,pair[0][0],pair[0][1],count[0][2]))

    



        
        conn.execute("ROLLBACK TO s1")
        
        #strat 2
        #Finds the most common end_location_name
        cursor.execute("""
          SELECT end_location_name, COUNT(*)
          FROM trip
          WHERE end_location_name IS NOT NULL
            AND TRIM(end_location_name) <> ''
          GROUP BY end_location_name
          ORDER BY COUNT(*) DESC
          LIMIT 1
        """)

        
        common_location = cursor.fetchone()[0]

        cursor.execute(f"""
        UPDATE trip
        SET end_location_name = '{common_location}'
        WHERE end_location_name IS NULL or TRIM(end_location_name) = ''
        """)

        for i in ids:
            cursor.execute("SELECT start_location_name, end_location_name FROM trip WHERE trip_id = ?", (i,))
            pair = cursor.fetchall()
            
            cursor.execute(f"SELECT start_location_name, end_location_name, count(*) FROM trip WHERE '{pair[0][0]}' = start_location_name AND '{pair[0][1]}'  = end_location_name")
            count = cursor.fetchall()

            

            return_list.append((i,pair[0][0],pair[0][1],count[0][2]))
    

        conn.execute("ROLLBACK TO s1")
        #strat 3
          
        cursor.execute(f"""
        UPDATE trip
        SET end_location_name = start_location_name
        WHERE (end_location_name IS NULL or TRIM(end_location_name) = '') AND (start_location_name IS NOT NULL AND TRIM(start_location_name) <> '')
        """)
        
        cursor.execute(f"""
        UPDATE trip
        SET end_location_name = '{common_location}'
        WHERE end_location_name IS NULL or TRIM(end_location_name) = ''
        """)

        for i in ids:
            cursor.execute("SELECT start_location_name, end_location_name FROM trip WHERE trip_id = ?", (i,))
            pair = cursor.fetchall()
            
            cursor.execute(f"SELECT start_location_name, end_location_name, count(*) FROM trip WHERE '{pair[0][0]}' = start_location_name AND '{pair[0][1]}'  = end_location_name")
            count = cursor.fetchall()

            

            return_list.append((i,pair[0][0],pair[0][1],count[0][2]))

        #after strat 3
        conn.commit()
        
        #conn.execute("ROLLBACK TO s1") #SHOULD BE REMOVED WHEN CODE IS DONE
        #conn.execute("RELEASE s1")

        

        return return_list

    #(Question 2)
    def replace_NaN_with_start(self,conn):#(s)
        """
                a.	Identify all missing values in the start_location_name column, saving the trip_id associated with the first 6 rows with a missing value for start_location_name.

                b.	Replace each missing entry with the most frequent start_location_name value in the table.

                c.	For each of the first 6 replacements, compute and return the total number of occurrences of the associated, new, pair <start_location_name, end_location_name>
        
                Returns: List: [['trip_id', 'start_location_name', 'end_location_name', 'pair_count'], (555555, '80 Whitworth BM, Albuquerque, NY 87888, ILD', '898 Mountain Rd NW, Albuquerque, NM 87104, USA', 3) , (333333, '71 Whitworth NW, Albuquerque, NM 123123, USA', 'Stop St.', 3), .....]
        """
        cursor = conn.cursor()#(s)
        #Your code here
        trip_id = []
        return_list = [['trip_id', 'start_location_name', 'end_location_name', 'pair_count']]
        
        #1.a identify trip_id
        cursor.execute("SELECT trip_id FROM trip WHERE start_location_name IS NULL or TRIM(start_location_name) = '' LIMIT 6")
        errors = cursor.fetchall()
        ids = [row[0] for row in errors]


        #find most frequent start_location
        cursor.execute("""
          SELECT start_location_name, COUNT(*)
          FROM trip
          WHERE start_location_name IS NOT NULL
            AND TRIM(start_location_name) <> ''
          GROUP BY start_location_name
          ORDER BY COUNT(*) DESC
          LIMIT 1
        """)
        common_location = cursor.fetchone()[0]

        
        cursor.execute(f"""
        UPDATE trip
        SET start_location_name = '{common_location}'
        WHERE start_location_name IS NULL or TRIM(start_location_name) = ''
        """)

        for i in ids:
            cursor.execute("SELECT start_location_name, end_location_name FROM trip WHERE trip_id = ?", (i,))
            pair = cursor.fetchall()
            
            cursor.execute(f"SELECT start_location_name, end_location_name, count(*) FROM trip WHERE '{pair[0][0]}' = start_location_name AND '{pair[0][1]}'  = end_location_name")
            count = cursor.fetchall()

            

            return_list.append((i,pair[0][0],pair[0][1],count[0][2]))

        conn.commit()
        return return_list

    #(Question 3)
    def column_name_change(self,conn):#(s)
        """
            Standardise the naming convention of the DURATION column to align with typical lowercase formatting used in data schemas,
            by having this column name replaced with duration. Once the standardisation is performed, return the new column name and the first 5 values in this column.
          
            Returns: List: (['duration'], ['h:mm:ss', 'h:mm:ss', 'h:mm:ss', 'h:mm:ss', 'h:mm:ss'])
        """
        cursor = conn.cursor()#(s)
        #Your code here

        duration_list = []
        cursor.execute("ALTER TABLE trip RENAME COLUMN DURATION TO duration;")

        cursor.execute("SELECT duration FROM trip;")
        tables = cursor.fetchall()
        


        
        cursor.execute("PRAGMA table_info(trip)")
        names = cursor.fetchall()


        cursor.execute("SELECT duration FROM trip LIMIT 5;")
        durations = cursor.fetchall()

        
        for i in durations:
            duration_list.append(i[0])
            
        return_list = ([names[5][1]],duration_list)

        conn.commit()
        return return_list
        

    #(Question 4)
    def normalize(self,conn):#(s)
        """
            Normalize the values in the month column so that all entries are in lowercase (i.e., june), regardless of their original format.
            Once the normalisation is performed, return, for only the following columns, the first 5 data values. 
            Note that the column names should appear in the results: month, trip_id, duration. 
            
            Returns: List: (['month', 'trip_id', 'duration'], [('month', 1613335, 'h:mm:ss'), ('month', 1613639, 'h:mm:ss'), ('month', 1613708, 'h:mm:ss'), ('month', 1613867, 'h:mm:ss'), ('month', 1636714, 'h:mm:ss')])

        """
        cursor = conn.cursor()#(s)
        #Your code here
        cursor.execute("""
        UPDATE trip
        SET month = LOWER(month);
        """)

        cursor.execute("""
        SELECT month, trip_id, duration
        FROM trip
        LIMIT 5;
        """)
        result = cursor.fetchall()

        return_tuple = (['month', 'trip_id', 'duration'], result)
        
        conn.commit()
        return return_tuple

    #(Question 5)
    def trip_duration(self,conn):#(s)
        """
            For entries in the duration column that are missing, derive the appropriate values using the started_at and ended_at columns. 
            Ensure these columns are in a format that supports datetime operations. Once the missing values are filled with the derived values, 
            return count of Nan values in column duration, as well as the data values in columns trip_id and duration for ONLY the rows with 
            the following trip_id values: 1821126, 1821158, 1821204, 1821289, 2047623. Make sure that the column names appear in the results.
         
            Returns: List: (NaN_count, ['trip_id', 'duration'], [(1821126, 'h:mm:ss'), (1821158, 'h:mm:ss'), (1821204, 'h:mm:ss'), (1821289, 'h:mm:ss'), (2047623, 'h:mm:ss')])
        """
        cursor = conn.cursor()#(s)
        #Your code here
        cursor.execute("SAVEPOINT s4")
        cursor.execute("SELECT COUNT(*) FROM 'trip' WHERE duration IS NULL or TRIM(duration) = '' LIMIT 1")
        count = cursor.fetchall()

        cursor.execute("""
        UPDATE trip
        SET started_at = printf('%04d-%02d-%02d %02d:%s',
          substr(started_at, instr(started_at,' ')-4, 4),                    
          substr(started_at, 1, instr(started_at,'/')-1),                     
          substr(substr(started_at, instr(started_at,'/')+1),
                 1, instr(substr(started_at, instr(started_at,'/')+1),'/')-1),
          CAST(substr(started_at, instr(started_at,' ')+1,
                      instr(started_at, ':') - instr(started_at,' ') - 1) AS INTEGER), 
          substr(started_at, instr(started_at, ':')+1)                        
        )
        WHERE started_at LIKE '%/%/%';

        """)

        
        cursor.execute("""
        UPDATE trip
        SET ended_at = printf('%04d-%02d-%02d %02d:%s',
          substr(ended_at, instr(ended_at,' ')-4, 4),                     
          substr(ended_at, 1, instr(ended_at,'/')-1),                     
          substr(substr(ended_at, instr(ended_at,'/')+1),
                 1, instr(substr(ended_at, instr(ended_at,'/')+1),'/')-1),
          CAST(substr(ended_at, instr(ended_at,' ')+1,
                      instr(ended_at, ':') - instr(ended_at,' ') - 1) AS INTEGER), 
          substr(ended_at, instr(ended_at, ':')+1)                        
        )
        WHERE ended_at LIKE '%/%/%';

        """)

        cursor.execute("""
        SELECT COUNT(*), duration FROM trip WHERE duration = '0:00:00' or duration = '00:00:00'
        """)
        
        cursor.execute("SELECT JULIANDAY(started_at), ended_at, duration FROM 'trip' WHERE trip_id = 1821126") 

        cursor.execute("""
        UPDATE trip
        SET duration = (
          WITH s AS (
            SELECT CAST(ROUND((julianday(ended_at) - julianday(started_at)) * 86400.0) AS INTEGER) AS tot
          )
          SELECT printf('%d:%02d:%02d', tot/3600, (tot%3600)/60, tot%60) FROM s
        )
        WHERE duration IS NULL or TRIM(duration) = '';
        """)
        cursor.execute("SELECT started_at, ended_at, duration FROM 'trip' WHERE trip_id = 1821126")


        check_list = [1821126, 1821158, 1821204, 1821289, 2047623]
        durations = []
        for i in check_list:
            cursor.execute(f"SELECT duration FROM trip WHERE trip_id = {i}")
            durations.append((i,cursor.fetchall()[0][0]))


        cursor.execute("""
        SELECT COUNT(*), duration FROM trip WHERE duration = '0:00:00' or duration = '00:00:00'
        """)
        conn.commit()
        #cursor.execute("ROLLBACK TO s4")

        return (count[0][0], ['trip_id', 'duration'],durations)


    #(Question 6)
    def shortest_trip(self,conn):#(s)
        """
            Return the shortest trip duration in the database and the number of distinct 
                <start_location_name, end_location_name> pair(s) associated with this duration value. 

            Returns: List: [('h:mm:ss', distinct_pairs_number)]
        """
        cursor = conn.cursor()#(s)
        #Your code here
        cursor.execute("SAVEPOINT s3")

        cursor.execute("""
        WITH dur AS (
            SELECT
                start_location_name,
                end_location_name,
                duration,
                -- convert duration string 'H:MM:SS' to total seconds for ordering
                (CAST(substr(duration, 1, instr(duration, ':') - 1) AS INTEGER) * 3600) +
                (CAST(substr(duration, instr(duration, ':') + 1,
                             instr(substr(duration, instr(duration, ':') + 1), ':') - 1) AS INTEGER) * 60) +
                (CAST(substr(duration, length(duration) - 1, 2) AS INTEGER)) AS total_seconds
            FROM trip
        ),
        minval AS (
            SELECT MIN(total_seconds) AS min_sec
            FROM dur
        )
        SELECT
            d.duration AS hms,
            COUNT(DISTINCT d.start_location_name || '→' || d.end_location_name) AS distinct_pairs
        FROM dur d
        JOIN minval m ON d.total_seconds = m.min_sec
        GROUP BY d.duration;
        """)
        
        conn.commit()
        #cursor.execute("ROLLBACK TO s3")
        return cursor.fetchall()


    #(Question 7)
    def median_duration_trip(self, conn):#(s)
        """
              Return the median trip duration in the database and the number of distinct  <start_location_name, end_location_name> pair(s) associated with this duration value.

            Returns: List: [('h:mm:ss', distinct_pairs_number)]
        """
        cursor = conn.cursor()#(s)
        #Your code here

        
        cursor.execute("""
        WITH dur AS (
          SELECT
            start_location_name,
            end_location_name,
            duration,
            -- convert 'H:MM:SS' to total seconds
            (CAST(substr(duration, 1, instr(duration, ':') - 1) AS INTEGER) * 3600) +
            (CAST(substr(duration, instr(duration, ':') + 1,
                         instr(substr(duration, instr(duration, ':') + 1), ':') - 1) AS INTEGER) * 60) +
            (CAST(substr(duration, length(duration) - 1, 2) AS INTEGER)) AS total_seconds
          FROM trip
        ),
        ordered AS (
          SELECT
            start_location_name,
            end_location_name,
            duration,
            total_seconds,
            ROW_NUMBER() OVER (ORDER BY total_seconds) AS rn,
            COUNT(*)     OVER ()                       AS cnt
          FROM dur
        ),
        medval AS (
          SELECT
            CASE
              WHEN cnt % 2 = 1 THEN
                (SELECT total_seconds FROM ordered WHERE rn = (cnt + 1) / 2)
              ELSE
                -- average the two middle values; ROUND to nearest second
                ROUND((
                  (SELECT total_seconds FROM ordered WHERE rn =  cnt      / 2) +
                  (SELECT total_seconds FROM ordered WHERE rn = (cnt / 2) + 1)
                ) / 2.0)
            END AS med_sec
          FROM ordered
          LIMIT 1
        )
        SELECT
          d.duration AS hms,
          COUNT(DISTINCT d.start_location_name || '→' || d.end_location_name) AS distinct_pairs
        FROM dur d
        JOIN medval m ON d.total_seconds = m.med_sec
        GROUP BY d.duration;
        """)
        conn.commit()
        return cursor.fetchall()
 
    #(Question 8)
    def trips_distance_in_miles(self,conn):#(s)
        """

            Using the Haversine formula, based on coordinates in table geocode, calculate the distance in miles between <start_location_name, end_location_name> of the following trip_ids:
            1637299, 1640777, 1652952
          
            Returns: List: [(1637299, distance_in_miles), (1640777, distance_in_miles), (1652952, distance_in_miles)]   
        """
        cursor = conn.cursor()#(s)

        cursor.execute( """
        WITH addr AS (
          SELECT
            t.trip_id,
            TRIM(CASE WHEN instr(t.start_location_name, ',') > 0
                      THEN substr(t.start_location_name, 1, instr(t.start_location_name, ',') - 1)
                      ELSE t.start_location_name END) AS start_street,
            TRIM(CASE WHEN instr(t.end_location_name, ',') > 0
                      THEN substr(t.end_location_name, 1, instr(t.end_location_name, ',') - 1)
                      ELSE t.end_location_name END)   AS end_street
          FROM trip t
          WHERE t.trip_id IN (1637299, 1640777, 1652952)
        ),
        coords AS (
          SELECT
            a.trip_id,
            gs.x AS lon1, gs.y AS lat1,
            ge.x AS lon2, ge.y AS lat2
          FROM addr a
          JOIN geocode gs ON LOWER(a.start_street) = LOWER(gs.street)
          JOIN geocode ge ON LOWER(a.end_street)   = LOWER(ge.street)
        )
        SELECT
          trip_id,
          2 * 3958.8 * ASIN(
            EXP(0.5 * LN(
              SIN(RADIANS(lat2 - lat1) / 2) * SIN(RADIANS(lat2 - lat1) / 2) +
              COS(RADIANS(lat1)) * COS(RADIANS(lat2)) *
              SIN(RADIANS(lon2 - lon1) / 2) * SIN(RADIANS(lon2 - lon1) / 2)
            ))
          ) AS distance_miles
        FROM coords
        ORDER BY CASE trip_id
                   WHEN 1637299 THEN 1
                   WHEN 1640777 THEN 2
                   WHEN 1652952 THEN 3
                 END;
        """)
        conn.commit()
        return cursor.fetchall()

# ---Test Zone --- #

"""
Remove the comments below to test your solution locally. Once the solution is finalised, reapply the comments as they are currently.
"""

# import sqlite3

# if __name__ == '__main__':

#     #You can place any ad-hoc testing here 
#     conn = sqlite3.connect("coursework2.db")
#     my_instance = Tours()
#     print("Question 1")
#     print(my_instance.replace_NaN_with_stop(conn))
#     print("Question 2")
#     print(my_instance.replace_NaN_with_start(conn))
#     print("Question 3")
#     print(my_instance.column_name_change(conn))
#     print("Question 4")
#     print(my_instance.normalize(conn))
#     print("Question 5")
#     print(my_instance.trip_duration(conn))
#     print("Question 6")
#     print(my_instance.shortest_trip(conn))
#     print("Question 7")
#     print(my_instance.median_duration_trip(conn))
#     print("Question 8")
#     print(my_instance.trips_distance_in_miles(conn))
#     conn.close()  # Close the connection when done
    
    
