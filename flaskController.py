#!/usr/bin/python3
from flask import Flask, flash, request, redirect, url_for

from nif.annotation import *

from flask_cors import CORS
import re
import os
import shutil
from werkzeug.utils import secure_filename
import zipfile
import sys
import dill as pickle
import codecs
from summarizer import AbstractiveSummarizer

"""
then to start run:
export FLASK_APP=flaskController.py
export FLASK_DEBUG=1 (optional, to reload upon changes automatically)
python -m flask run

example calls:
curl -X GET localhost:5000/welcome
"""

app = Flask(__name__)
app.secret_key = "super secret key"
service = None

CORS(app)

'''

Headers:

Accept: text/turtle
Content-Type: text/turtle OR text/plain
Language: en OR de
Method: bert OR conv
Extract: true OR false

'''
@app.route('/srv-summarize/analyzeText', methods=['POST'])
def summarize():
    if request.method == 'POST':
        cType = request.headers["Content-Type"]
        accept = request.headers["Accept"]
        language = request.headers["Language"]
        method = request.headers["Method"]
        extract = request.headers["Extract"]

        data=request.stream.read().decode("utf-8")
        if accept == 'text/turtle':
            pass
        else:
            return 'ERROR: the Accept header '+accept+' is not supported!'
        if cType == 'text/plain':
            uri_prefix="http://lynx-project.eu/res/"+str(uuid.uuid4())
            d = NIFDocument.from_text(data, uri_prefix)
        elif cType == 'text/turtle':
            d = NIFDocument.parse_rdf(data, format='turtle')
        else:
            return 'ERROR: the contentType header '+cType+' is not supported!'

        service = AbstractiveSummarizer(language = language, method = method, extract = extract)
        summ = service.analyzeNIF(d)
        return summ.serialize(format="ttl")
    else:
        return 'ERROR, only POST method allowed.'

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(host='localhost', port=port, debug=True)
