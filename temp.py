import requests
import pandas as pd
from io import StringIO

API_KEY = "K757OWEW19L34ML9"

url = f"https://www.alphavantage.co/query?function=WTI&interval=monthly&datatype=csv&apikey={API_KEY}"
resp = requests.get(url)

df = pd.read_csv(StringIO(resp.text))
print(df.head())
