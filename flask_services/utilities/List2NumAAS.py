from flask import Flask, Response ,send_file , send_from_directory, request, current_app

import json

import sys

app = Flask(__name__)

@app.route("/utils/list2num", methods=['POST', 'GET'])
def StrList2Num():
    print(request.form.keys())
    if 'data' in request.form.keys():
        data_list = eval(request.form["data"])
        print(data_list)
        return str(len(data_list))

    return "countable object not found"
            
        

if __name__ == "__main__":
    print('Starting the API')
    app.run(host='0.0.0.0', port=5301)