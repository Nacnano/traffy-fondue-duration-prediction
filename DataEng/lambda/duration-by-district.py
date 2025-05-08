import boto3
import time
import json

# Configs
ATHENA_DATABASE = "dsde-finalproj-traffy"
S3_OUTPUT = "s3://dsde-final-proj/athena-results/"
AWS_REGION = "us-east-1"

athena = boto3.client("athena", region_name=AWS_REGION)

QUERY = """
SELECT
  pt AS problem_type,
  district,
  AVG(duration_min) AS avg_duration_minutes
FROM "dsde-finalproj-traffy"."test_folder",
UNNEST("type") AS t(pt)
GROUP BY pt, district
ORDER BY avg_duration_minutes DESC
"""

def lambda_handler(event, context):
    # Start Athena query
    response = athena.start_query_execution(
        QueryString=QUERY,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={"OutputLocation": S3_OUTPUT},
    )

    query_execution_id = response["QueryExecutionId"]

    # Poll until query finishes
    while True:
        result = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = result["QueryExecution"]["Status"]["State"]
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(1)

    if state != "SUCCEEDED":
        return {
            "statusCode": 500,
            "body": json.dumps({"error": f"Athena query failed: {state}"})
        }

    # Fetch results
    result_set = athena.get_query_results(QueryExecutionId=query_execution_id)
    rows = result_set["ResultSet"]["Rows"]

    # Parse header + rows
    header = [col["VarCharValue"] for col in rows[0]["Data"]]

    # Identify which columns should be numeric
    numeric_fields = {"avg_duration_minutes"}

    data = []
    for row in rows[1:]:
        row_data = {}
        for i, col in enumerate(row["Data"]):
            col_name = header[i]
            value = col.get("VarCharValue")
            if col_name in numeric_fields and value is not None:
                try:
                    row_data[col_name] = float(value)
                except ValueError:
                    row_data[col_name] = None
            else:
                row_data[col_name] = value
        data.append(row_data)

    return {
        "statusCode": 200,
        "body": json.dumps({"data": data}, ensure_ascii=False)
    }

