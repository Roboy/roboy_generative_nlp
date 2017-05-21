from __future__ import unicode_literals

import sys
import json
from pprint import pprint

sys.stdout = open('squad_trained_100k.txt', 'w')

with open('train-v1.1.json') as data_file:
    data= json.load(data_file)


# pprint(data)
# len_data=(len(data["data"][0]));


for index, item in enumerate(data["data"]):
    for index2, item2 in enumerate(item['paragraphs']):
   		for index3, item3 in enumerate(item2["qas"]):    	    	
			print(item3["question"]).encode('utf8')
			print(item3["answers"][0]['text']).encode('utf8')
    		
    		
sys.stdout.close()        