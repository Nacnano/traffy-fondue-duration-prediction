import boto3
import time
import json

# Athena & AWS Config
ATHENA_DATABASE = "dsde-finalproj-traffy"
S3_OUTPUT = "s3://dsde-final-proj/athena-results/"
AWS_REGION = "us-east-1"

athena = boto3.client("athena", region_name=AWS_REGION)

# Athena SQL query
QUERY = """
SELECT
  pt AS problem_type,
  AVG(duration_min) AS duration_minutes
FROM "dsde-finalproj-traffy"."test_folder",
UNNEST("type") AS t(pt)
GROUP BY pt
ORDER BY duration_minutes DESC
"""

def lambda_handler(event, context):
    # Start Athena query
    response = athena.start_query_execution(
        QueryString=QUERY,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={"OutputLocation": S3_OUTPUT},
    )

    query_execution_id = response["QueryExecutionId"]

    # Wait for completion
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

    # Get query results
    result_set = athena.get_query_results(QueryExecutionId=query_execution_id)
    rows = result_set["ResultSet"]["Rows"]

    # Extract headers
    header = [col["VarCharValue"] for col in rows[0]["Data"]]

    # Convert duration_minutes to float
    numeric_fields = {"duration_minutes"}
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
