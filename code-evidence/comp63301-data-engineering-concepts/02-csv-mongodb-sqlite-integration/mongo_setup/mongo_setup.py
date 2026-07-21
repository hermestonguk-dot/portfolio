from pymongo.mongo_client import MongoClient
import json
import os

"""
NOTE: File path for the JSON file may need to be changed depending on where file is stored
"""
import os

json_file_path = os.path.join(os.path.dirname(__file__), "weather_manchester_february_2018.json")

# json_file_path = "weather_manchester_february_2018.json"

# local mongoDB db, DO NOT CHANGE
uri = "mongodb://localhost:27017/"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!\n")
except Exception as e:
    print(e)

db = client["coursework4"]
collection = db["February2018"]

# Load the JSON file
with open(json_file_path, "r") as f:
    weather_data = json.load(f)           # this should be a list of dicts

# Insert all documents
collection.insert_many(weather_data)

# Check how many documents were inserted
print(f"Inserted {collection.count_documents({})} documents.")