# api_client.py
# A function to fetch data from a web API

import requests
import json

def get_user_data(user_id):
    """
    Fetches user data from a public API (JSONPlaceholder).
    """
    try:
        url = f"https://jsonplaceholder.typicode.com/users/{user_id}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None