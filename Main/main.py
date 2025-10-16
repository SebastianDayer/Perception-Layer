import asyncio
import websockets
import json
import ollama

model_name = "llama3.2-vision:11b"
client = ollama.AsyncClient()

async def listen():
    uri = "ws://127.0.0.1:8765"
    print(f"Connecting to {uri}....")

    conversation = [
        {
            "role": "system",
            "content": (
                "You are an AI watching a live camera feed. "
                "Given a list of detected objects, produce a short, fast description "
                "of the scene in one scentence."
                "\n\nDetected Objects: \n" +json.dumps(detections)
            )
        }
    ]

    async with websockets.connect(uri) as websocket:
        print("✅ Connected to YOLO stream.")

        async for message in websocket:
            if not message:
                continue

            # Parse YOLO JSON
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print("⚠️ Skipping invalid JSON.")
                continue

            # Add current frame as new user message
            detections = data.get("detections", [])
            summary = ", ".join([d["label"] for d in detections]) or "no objects detected"
            frame_text = f"Frame {data.get('frame_id')}: {summary}."

            recent_context = " ".join(
                [m["content"] for m in conversation[-6:] if m["role"] == "user"]
            )

            conversation.append({
                "role": "user",
                "content": f"{frame_text}\nContent: {recent_context}"
            })

            print(f"\n Interpreting: {frame_text}\n")

            try:

            # Stream model response with context
                result = await client.chat(model=model_name, messages=conversation, stream=True)

                full_response = ""
                async for chunk in result:
                    if chunk.message and chunk.message.content:
                        print(chunk.message.content, end="", flush=True)
                        full_response += chunk.message.content

                # Store assistant reply in conversation
                conversation.append({"role": "assistant", "content": full_response})

                # Optional: keep memory bounded (avoid token overflow)
                if len(conversation) > 5:
                    conversation = [conversation[0]] + conversation[-10:]
            except:
                print("Ollama error", e)

asyncio.run(listen())
