import http.client
import json

conn = http.client.HTTPSConnection("apex.oracle.com")
conn.request("GET", "/pls/apex/ryo/km-api/km/")

response = conn.getresponse()
if response.status == 200:
    data = response.read().decode("utf-8")
    json_data = json.loads(data)
    print(json_data)
else:
    print(f"Error: {response.status}")
