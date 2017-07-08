import asyncio
import websockets
import json as json

async def service_callback():
    async with websockets.connect('ws://localhost:9090') as websocket:

        await websocket.send("{ \"op\": \"advertise_service\",\
                      \"type\": \"roboy_communication_cognition/GenerateAnswer\",\
                      \"service\": \"/roboy/cognition/generative_nlp/answer\"\
                    }")
        i = 1
        while True:
            greeting = await websocket.recv()
            print("< {}".format(greeting))
            response = {}
            response["op"] = "service_response"
            response["id"] = "service_request:/roboy/cognition/generative_nlp/answer:" + str(i)
            response["result"] = True
            response["values"] = {"text_output": "fuck"}
            response["service"] = "/roboy/cognition/generative_nlp/answer"
            i += 1
            print(json.dumps(response))
            await websocket.send(json.dumps(response))

asyncio.get_event_loop().run_until_complete(service_callback())