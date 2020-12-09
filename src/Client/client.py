# CLIENT SIDE

import json
import requests
import zipfile
import io
import pathlib
from io import BytesIO
import base64
import shutil
import os

# Endpoint of API defined
url = f'http://thehandler.hopto.org:4000/trainForMe'
save_path = './recorded_gestures'
#zipping 
print('Zipping recorded data as data.zip')
output_filename = 'data'
shutil.make_archive(output_filename, 'zip', save_path)
print('Data zipping complete')
"""
# encoding the zip file into base64 format
with open("data.zip", "rb") as f: #take a look
    bytes = f.read()
    encoded = base64.b64encode(bytes)


# defining request body of the API call

payload = {
	"ZipFile": encoded.decode('ascii')
}

headers = {
	'Content-Type': 'application/json',
}
"""
files = {'file':open(f'{output_filename}.zip','rb')}
# API call
outcome = requests.post(url=url, files=files)
print(outcome)
zipped_model = BytesIO(outcome.content)

print('Recieved Model, Extracting')
# extracting the zip from response and extracting
with zipfile.ZipFile(zipped_model, 'r') as zip_ref:
    zip_ref.extractall('./')
    
for filename in os.listdir('./FineTuned'):
    if filename.endswith(".pth"):
        os.rename(f'./FineTuned/{filename}',f'./FineTuned/output.pth')
print('Model Stored')
