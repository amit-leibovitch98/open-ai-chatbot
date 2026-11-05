import json
import os
import time
import boto3
from boto3.dynamodb.conditions import Key
from openai import OpenAI

# Initialize clients outside the handler so they are reused across warm invocations
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
table = dynamodb.Table("chat-history")

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "content-type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def respond(status, body, extra_headers=None):
    return {
        "statusCode": status,
        "headers": {**CORS_HEADERS, **(extra_headers or {})},
        "body": json.dumps(body) if body is not None else "",
    }


def lambda_handler(event, context):
    http_method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        # The Lambda console Test tab sends no method; treat it as POST (Task 3)
        or "POST"
    )

    # 1. CORS preflight (Task 5)
    if http_method == "OPTIONS":
        return respond(200, None)

    # 2. Reject non-POST methods (Task 4)
    if http_method != "POST":
        return respond(
            405,
            {"error": f"Method {http_method} not allowed. Please use POST."},
            {"Allow": "POST, OPTIONS"},
        )

    # 3. Parse input body
    body = event.get("body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return respond(400, {"error": "Invalid JSON in request body."})
    elif body is None:
        body = {}

    user_id = body.get("user_id", "default_user")

    # Load saved history so the UI can show past messages after login/refresh
    if body.get("action") == "get_history":
        try:
            ddb_response = table.query(
                KeyConditionExpression=Key("user_id").eq(user_id),
                ScanIndexForward=True,  # oldest first
            )
            history = [
                {
                    "role": item.get("role", "user"),
                    "content": item.get("content", ""),
                    "timestamp": int(item["timestamp"]),
                }
                for item in ddb_response.get("Items", [])
            ]
        except Exception as e:
            print(f"Error loading history from DynamoDB: {e}")
            return respond(500, {"error": f"Could not load history: {e}"})
        return respond(200, {"history": history})

    user_prompt = body.get("user_prompt")
    if not user_prompt:
        return respond(400, {"error": "Missing required field: 'user_prompt'"})

    # 4. Task 7: retrieve only the last 6 messages for this user
    try:
        ddb_response = table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
            ScanIndexForward=False,  # newest first
            Limit=6,
        )
        recent_items = ddb_response.get("Items", [])
        recent_items.reverse()  # oldest -> newest
    except Exception as e:
        print(f"Error querying DynamoDB: {e}")
        recent_items = []

    # Build the conversation as role-tagged messages for the Responses API
    messages = [
        {"role": item.get("role", "user"), "content": item.get("content", "")}
        for item in recent_items
    ]
    messages.append({"role": "user", "content": user_prompt})

    # 5. Task 9: measure OpenAI API response time
    start_time = time.time()
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=messages,
        )
        ai_reply = response.output_text
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return respond(502, {"error": f"OpenAI request failed: {str(e)}"})
    openai_response_time_ms = round((time.time() - start_time) * 1000)

    # 6. Task 7: save both the user message and the AI reply
    try:
        now_ms = int(time.time() * 1000)
        table.put_item(Item={
            "user_id": user_id,
            "timestamp": now_ms,
            "role": "user",
            "content": user_prompt,
        })
        table.put_item(Item={
            "user_id": user_id,
            "timestamp": now_ms + 1,  # keeps the reply after the question
            "role": "assistant",
            "content": ai_reply,
        })
    except Exception as e:
        print(f"Error writing to DynamoDB: {e}")

    return respond(200, {
        "ai_reply": ai_reply,
        "openai_response_time_ms": openai_response_time_ms,
    })
