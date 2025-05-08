import boto3
import time
import json

# Config
ATHENA_DATABASE = "dsde-finalproj-traffy"
S3_OUTPUT = "s3://dsde-final-proj/athena-results/"
AWS_REGION = "us-east-1"

athena = boto3.client("athena", region_name=AWS_REGION)

# SQL Query
QUERY = """
SELECT
  org,
  MIN(duration_min) AS min_duration,
  approx_percentile(duration_min, 0.25) AS q1,
  approx_percentile(duration_min, 0.5) AS median,
  approx_percentile(duration_min, 0.75) AS q3,
  MAX(duration_min) AS max_duration,
  COUNT(*) AS count_reports
FROM "dsde-finalproj-traffy"."test_folder",
UNNEST(organization) AS t(org)
WHERE duration_min IS NOT NULL
GROUP BY org
ORDER BY count_reports DESC
"""

def lambda_handler(event, context):
    # Start Athena query
    response = athena.start_query_execution(
        QueryString=QUERY,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={"OutputLocation": S3_OUTPUT},
    )

    query_execution_id = response["QueryExecutionId"]

    # Wait until finished
    while True:
        status = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(1)

    if state != "SUCCEEDED":
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Athena query failed: {state}"})
        }

    # Get results
    result_set = athena.get_query_results(QueryExecutionId=query_execution_id)
    rows = result_set["ResultSet"]["Rows"]

    # Extract headers
    header = [col["VarCharValue"] for col in rows[0]["Data"]]

    # Define numeric columns
    numeric_fields = {
        "min_duration",
        "q1",
        "median",
        "q3",
        "max_duration",
        "count_reports"
    }

    # Parse rows
    data = []
    for row in rows[1:]:
        parsed = {}
        for i, col in enumerate(row["Data"]):
            col_name = header[i]
            val = col.get("VarCharValue")
            if col_name in numeric_fields:
                try:
                    parsed[col_name] = float(val) if val is not None else None
                except:
                    parsed[col_name] = None
            else:
                parsed[col_name] = val
        data.append(parsed)

    return {
        "statusCode": 200,
        "body": json.dumps({"data": data}, ensure_ascii=False)
    }
