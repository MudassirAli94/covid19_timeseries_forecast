import pandas as pd
import requests
from gcp_functions import upload_dataframe_to_gcs
from tqdm import tqdm
import json

with open('config.json') as config_file:
    config = json.load(config_file)

YOUR_BUCKET_NAME = config["bucket_name"]
PROJECT_ID = config["project_id"]

url = "https://healthdata.gov/resource/j8mb-icvb.csv"
api_limit = 1000
params = {"$limit": api_limit}

dfs = []
offset = 0

while True:
    params["$offset"] = offset
    response = requests.get(url, params=params)

    if response.status_code == 200:
        df = pd.read_csv(response.url)
        dfs.append(df)
        assert len(dfs)

        if len(df) < api_limit:
            break
        else:
            offset += api_limit
    else:
        print("Failed to fetch data. Status code:", response.status_code)
        break


    tqdm.write(f"Downloaded {offset} rows")


final_df = pd.concat(dfs, ignore_index=True)

assert final_df is not None, "Dataframe is empty. Check the data source."
assert len(final_df) == 238919, "Dataframe is too small. Check the data source."

upload_dataframe_to_gcs(bucket_name=YOUR_BUCKET_NAME,
                        df=final_df,
                        file_name='covid_19_timeseries_raw',
                        project_id=PROJECT_ID)

final_df.to_csv("covid_19_timeseries_raw.csv", index=False)


