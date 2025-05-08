from airflow.decorators import dag, task
from datetime import datetime
import pandas as pd
import requests
import json
import boto3

# AWS and API Config
BASE_URL = "https://publicapi.traffy.in.th/teamchadchart-stat-api/geojson/v1"
QUERY_PARAMS = {
    "output_format": "json",
    "name": "vijak khajornritdacha",
    "org": "datasci",
    "purpose": "educational",
    "email": "vijak13019@gmail.com",
    "limit": 100,
    "state_type": "finish",
}

import random
import string

file_name_length = 8
file_name = "".join(
    random.choice(string.ascii_uppercase + string.digits)
    for _ in range(file_name_length)
)

S3_BUCKET = "dsde-final-proj"
S3_KEY = f"test-folder/{file_name}.jsonl"
AWS_REGION = "us-east-1"

STR_COLS = [
    "ticket_id",
    "comment",
    "photo",
    "photo_after",
    "address",
    "subdistrict",
    "district",
    "province",
    "state",
]


@dag(
    schedule="@daily",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["traffy", "cleaning", "s3"],
)
def fetch_to_s3_dag():

    @task
    def fetch_data() -> list:
        response = requests.get(BASE_URL, params=QUERY_PARAMS)
        response.raise_for_status()
        return response.json()["features"]

    @task
    def transform_data(features: list[dict]) -> pd.DataFrame:
        rows = []
        for feature in features:
            props = feature["properties"]
            coords = (
                feature["geometry"]["coordinates"]
                if feature.get("geometry")
                else [None, None]
            )

            row = {
                "ticket_id": props.get("ticket_id"),
                "type": props.get("problem_type_fondue"),
                "organization": props.get("org"),
                "comment": props.get("description"),
                "photo": props.get("photo_url"),
                "photo_after": props.get("after_photo"),
                "address": props.get("address"),
                "subdistrict": props.get("subdistrict"),
                "district": props.get("district"),
                "province": props.get("province"),
                "timestamp": pd.to_datetime(
                    props.get("timestamp"), utc=True, errors="coerce"
                ),
                "state": props.get("state"),
                "star": props.get("star"),
                "count_reopen": props.get("count_reopen"),
                "last_activity": pd.to_datetime(
                    props.get("last_activity"), utc=True, errors="coerce"
                ),
                "lat": coords[1] if coords else None,
                "lon": coords[0] if coords else None,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    @task
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        df[STR_COLS] = df[STR_COLS].astype("string")
        for col in STR_COLS:
            df = df[df[col].notna() & (df[col].str.strip() != "")]
        df = df.dropna(subset=["ticket_id", "star"])
        df = df[df["type"].notna() & (df["type"].str.strip("{}") != "")]
        df = df[df["lat"].notna() & df["lon"].notna()]
        df["count_reopen"] = df["count_reopen"].fillna(0).astype(int)
        df = df[df["state"] == "เสร็จสิ้น"]
        df["duration_min"] = (
            df["last_activity"] - df["timestamp"]
        ).dt.total_seconds() / 60
        return df

    @task
    def upload_to_s3(df: pd.DataFrame):
        s3 = boto3.client("s3", region_name=AWS_REGION)
        jsonl_body = "\n".join(
            df.to_json(orient="records", lines=True, force_ascii=False).splitlines()
        )
        s3.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=jsonl_body.encode("utf-8"))
        print(f"✅ Uploaded to s3://{S3_BUCKET}/{S3_KEY} with {len(df)} records")

    # DAG flow
    raw = fetch_data()
    transformed = transform_data(raw)
    cleaned = clean_data(transformed)
    upload_to_s3(cleaned)


dag_instance = fetch_to_s3_dag()
