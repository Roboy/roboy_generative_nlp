from __future__ import unicode_literals

import sys
import json
from pprint import pprint

sys.stdout = open('squad.txt', 'w')

with open('dev-v1.1.json') as data_file:
    data= json.load(data_file)


# pprint(data)
for index, item in enumerate(data["data"]):
    for index2, item2 in enumerate(data["data"][index]['paragraphs']):    	
		q=(data["data"][index]['paragraphs'][index2]["qas"][0]["question"]).encode('utf8')
		a=(data["data"][index]['paragraphs'][index2]["qas"][0]["answers"][0]['text']).encode('utf8')
    		print q
    		print a

# print(len(data["data"][0]['paragraphs']))
# print(data["data"][20]['paragraphs'][22]["qas"][0]["question"])
# print(data["data"][0]['paragraphs'][53]["qas"][0]["answers"][0]['text'])

sys.stdout.close()        