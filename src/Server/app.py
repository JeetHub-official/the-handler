# SERVER SIDE 

import pickle
import flask as fl
from flask import Flask, request, jsonify
import zipfile
import io
import pathlib
from io import BytesIO
import base64
from main import get_fine_tuned
import torch
from datetime import datetime
import os
import sys
import time
import torch

##creating a flask app and naming it "app"
app = Flask('app')

@app.route('/getSquare', methods=['POST'])
def get_square():
    '''An REST API endpoint that accepts a HTTP POST request
    Takes a number and returns  it's square.
    A dummy method without database involvement for quick connection checkup'''
    if not request.json or 'number' not in request.json:
        abort(400)
    num = request.json['number']

    return jsonify({'answer': num ** 2})

@app.route('/trainForMe', methods=['POST'])
def train_for_me():
	id = datetime.now().microsecond
	#print(f"id allocated: {id}")
	#time.sleep(30)
	print(f"getting request body:{id}")
	#request_body = request.get_json()
	file_zip = request.files['file']
	file_zip.save('data'+str(id)+'.zip')
	"""
	# decoding the zip file from request body
	base64_bytes = request_body['ZipFile'].encode('ascii')
	message_bytes = base64.b64decode(base64_bytes)
	data = BytesIO(message_bytes)
	print("starting training")
	"""
	time.sleep(2)
	#backend
	model = get_fine_tuned(id)
	
	print('Training Finished, Saving model')
	torch.save(model.module.state_dict(), f'./FineTuned/output_{id}.pth')
	del model
	torch.cuda.empty_cache()
	print('Zipping Model')
	file_name = pathlib.Path(f'./FineTuned/output_{id}.pth')
	data = io.BytesIO()
	with zipfile.ZipFile(data, mode='w') as z:
		z.write(file_name)
	
	data.seek(0)
	
	# sending the zip to client
	print('Sending Model')
	return fl.send_file(data,mimetype='application/zip',as_attachment=True,attachment_filename='data.zip')
        
        
        
@app.route('/ping', methods=['POST'])
def ping():  
    request_body = request.get_json()
    return f"Pinging Model!! number was {request_body['number']}"


@app.route('/pingGET', methods=['GET'])
def pingGET():  
    #request_body = request.get_json()
    return "Pinging Model!!"



if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=10000)
