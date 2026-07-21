# Lab 01: SQLite Trip Cleaning

## What This Shows

This lab implements SQLite data-cleaning and query logic over trip data. The submitted code is in `tour_solution.py`, and the local database needed by the code is `coursework2.db`.

## Contents

- `tour_solution.py`: completed coursework solution code.
- `coursework2.db`: SQLite database used by the lab.

## How To Run

The original coursework test block is commented out inside `tour_solution.py`. To test locally:

1. Open `tour_solution.py`.
2. Uncomment the test-zone block at the bottom.
3. Run:

```bash
python tour_solution.py
```

The code mutates the database during cleaning tasks, so run it on a copy of `coursework2.db` if you want to preserve the original database state.

## Skills Demonstrated

- SQLite queries and updates
- missing-value handling
- schema standardisation
- duration calculations
- distance calculation from geocoded data

