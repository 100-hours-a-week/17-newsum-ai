# sse_client_test.py
import httpx
import asyncio
import json

async def main():
    sse_url = "http://localhost:8000/stream-workflow-sse"
    print(f"Connecting to SSE endpoint: {sse_url}")

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", sse_url) as response:
                if response.status_code == 200:
                    print("Connected. Waiting for events...")
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            try:
                                data_str = line[len("data:"):].strip()
                                event_data = json.loads(data_str)
                                print(f"Received Event: {event_data}")
                                if event_data.get("node") == "__end__" or event_data.get("node") == "__error__":
                                    print("Workflow finished or error occurred.")
                                    break
                            except json.JSONDecodeError:
                                print(f"Received non-JSON data line: {line}")
                        else:
                             print(f"Received line: {line}") # Keep-alive 등 다른 라인
                else:
                    print(f"Error connecting to SSE endpoint: {response.status_code}")
                    print(await response.aread())

    except httpx.RequestError as e:
        print(f"HTTP request error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())