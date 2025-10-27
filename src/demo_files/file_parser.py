# file_parser.py
# A script to read and parse CSV files.

import csv

def parse_csv_file(filepath):
    """
    Reads a CSV file and returns a list of dictionaries.
    """
    data = []
    try:
        with open(filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []