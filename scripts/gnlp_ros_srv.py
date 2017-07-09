from __future__ import print_function

import os
import sys
import tensorflow as tf
import logging

import asyncio
import websockets
import json as json

from config import params_setup
from lib import data_utils
from lib.seq2seq_model_utils import create_model, get_predicted_sentence

import pdb

async def service_callback():
    async with websockets.connect('ws://localhost:9090') as websocket:

        # advertise the service
        await websocket.send("{ \"op\": \"advertise_service\",\
                      \"type\": \"roboy_communication_cognition/GenerateAnswer\",\
                      \"service\": \"/roboy/cognition/generative_nlp/answer\"\
                    }")

        i = 1 # counter for the service request IDs
     
        with tf.Session() as sess:
            # Create model and load parameters.
            logging.info("Loading the model")
            args = params_setup()
            args.batch_size = 1  # We decode one sentence at a time.
            model = create_model(sess, args)

            # Load vocabularies.
            vocab_path = os.path.join(
                args.data_dir, "vocab%d.in" % args.vocab_size)
            vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

            logging.info("Service /roboy/cognition/generative_nlp/answer is ready")

            # wait for the service request, generate the answer, and send it back
            while True:
                try:
                    request = await websocket.recv()

                    sentence = json.loads(request)["args"]["text_input"]
                    model_response = get_predicted_sentence(args, sentence, vocab, rev_vocab, model, sess)

                    srv_response = {}
                    answer = {}

                    if isinstance(model_response, list):
                        text = model_response[0]['dec_inp']
                    else:
                        text = model_response['dec_inp']

                    answer["text_output"] = ''.join([i if ord(i) < 128 else '' for i in text]) # strip down unicode 

                    srv_response["values"] = answer
                    srv_response["op"] = "service_response"
                    srv_response["id"] = "service_request:/roboy/cognition/generative_nlp/answer:" + str(i)
                    srv_response["result"] = True
                    srv_response["service"] = "/roboy/cognition/generative_nlp/answer"
                    i += 1 

                    await websocket.send(json.dumps(srv_response))

                except Exception as e:
                    logging.exception("Oopsie! Got an exception in generative_nlp")

logging.basicConfig(level=logging.INFO)
asyncio.get_event_loop().run_until_complete(service_callback())