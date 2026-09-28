# Lab 02: CSV, MongoDB, and SQLite Integration

## contents

This lab integrates multiple data sources: traffic CSVs, weather documents in MongoDB, a SQLite route-distance database, and journey-time archive data. The submitted code is in `journey_solution.py`.

## Contents

- `journey_solution.py`: completed coursework solution code.
- `rawpvr_2018-02-01_28d_1083.csv`: traffic data for site 1083.
- `rawpvr_2018-02-01_28d_1415.csv`: traffic data for site 1415.
- `JTarchive_1083to1415_Tues_Sun_00-23hours_01-28_Feb_2018.csv`: journey-time archive.
- `coursework4.db`: SQLite route-distance database.
- `mongo_setup/mongo_setup.py`: loads weather data into MongoDB.
- `mongo_setup/weather_manchester_february_2018.json`: weather documents for MongoDB.
- `requirements.txt`: Python packages for local execution.

## How To Run

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start a local MongoDB server, then load the weather data:

```bash
python mongo_setup/mongo_setup.py
```

The test-zone block at the bottom of `journey_solution.py` is commented out in the submitted file. Uncomment it, then run:

```bash
python journey_solution.py
```

The script writes intermediate CSVs to `student_answer_csvs/`.


