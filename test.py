import asyncio
import websockets
import json
import os
from dotenv import load_dotenv

import pyaudio
import numpy as np
import base64
import time

load_dotenv()

# For openai
#key = os.getenv("OPENAI_API_KEY")
#url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# For azure openai
endpoint = os.getenv("https://deepchatbot.openai.azure.com/") # ex. https://my-eastus2-openai-resource.openai.azure.com/
deployment = os.getenv("gpt-3.5-turbo") # gpt-4o-realtime-preview
key = os.getenv("openai_api_key") # this is the API key for the Azure OpenAI resource

# strip the https:// from the endpoint
endpoint = endpoint.replace("https://", "")

url = f"wss://{endpoint}/openai/realtime?deployment={deployment}&api-version=2024-10-01-preview"

#wss://my-eastus2-openai-resource.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview-1001&api_key=....

print(key)
print(url)

async def connect():
    async with websockets.connect(url, extra_headers={
        "api-key": key , # if we provide a key , we don't need to provide the authorization header
#"Authorization": "Bearer " + credential.key ,
 #       "OpenAI-Beta": "realtime=v1",
    }) as websocket:
        print("Connected to server.")
        
        # Create tasks for sending and receiving messages
        receive_task = asyncio.create_task(receive_messages(websocket))
        send_task = asyncio.create_task(send_messages(websocket))

           # Call record_audio in a separate thread
        record_task= await asyncio.to_thread(record_audio,  websocket)


        # Wait for both tasks to complete
        await asyncio.gather(receive_task, send_task, record_task)

# New function to handle receiving messages
async def receive_messages(websocket):
    # play the delta audio chunk using pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    while True:
        message = await websocket.recv()
        message_data = json.loads(message)
        if message_data.get("type") == "response.done":  # Check for response.done type
            print(message_data)  # Print only if the type is response.done
        elif message_data.get("type") == "response.audio.delta":  # Check for response.audio.delta type
            delta = message_data.get("delta")
            # decode delta from base64
            delta = base64.b64decode(delta)
            stream.write(delta)
        else:
            print(f"Received message type: {message_data.get('type')}")  # Print the type if not response.done

    stream.stop_stream()
    stream.close()
    p.terminate()

# New function to handle audio recording in a separate thread
async def record_audio(websocket):

    # Set up PyAudio
    p = pyaudio.PyAudio()
    sample_rate = 24000
    duration_ms = 100
    samples_per_chunk = sample_rate * (duration_ms / 1000)
    bytes_per_sample = 2
    bytes_per_chunk = int(samples_per_chunk * bytes_per_sample)
    
    chunk_size = 2400  # 100ms chunks
    format = pyaudio.paInt16
    channels = 1  # Mono
    record_seconds = 500

    # Open the microphone stream
    stream = p.open(format=format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    await websocket.send(json.dumps({
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "server_vad", 
                "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 200
            },
            "input_audio_transcription": {
                "model": "whisper-1"
            }
        }
    }))

    print("Listening to microphone for 5 seconds...")
    start_time = time.time()

    chunk_counter = 0  # Initialize a counter for audio chunks

    while time.time() - start_time < record_seconds:
        # Read audio data from the microphone
        data = stream.read(chunk_size)
        
        # Convert to numpy array (already mono)
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Convert to bytes and encode in base64
        base64_audio = base64.b64encode(audio_data.tobytes()).decode('utf-8')
        
        chunk_counter += 1  # Increment the counter
        print(f"sending audio chunk {chunk_counter}")  # Print the counter
        # Send the audio chunk
        await websocket.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }))

        # Wait for the server to process the audio chunk
        # Needed to avoid buffer overflow
        await asyncio.sleep(0.1)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Finished recording.")

    # Not necessary as the server will detect the end of the audio stream
    # Send the audio buffer finalize message
    #await websocket.send(json.dumps({
    #    "type": "input_audio_buffer.commit",
    #}))

# Update send_messages function to call record_audio in a separate thread
async def send_messages(websocket):
    await websocket.send(json.dumps({
        "type": "response.create",
        "response": {
            "modalities": ["text"],
            "instructions": "Please assist the user.",
        }
    }))

 
if __name__ == "__main__":
    asyncio.run(connect())