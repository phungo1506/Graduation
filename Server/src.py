#from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from PaddleDetection.deploy.pptracking.python.mot_jde_infer import JDE_Detector
from PaddleDetection.deploy.pptracking.python.mot.utils import MOTTimer, write_mot_results, flow_statistic
from collections import defaultdict
import base64
import time
import operator
from datetime import datetime
import yaml
# Choose to use a config and initialize the detector

phu_config = '/home/object_detection/MOT/work/PaddleDetection/deploy/pptracking/python/tracker_config.yml'
phu_checkpoint = '/home/object_detection/MOT/work/PaddleDetection/output_inference/fairmot_dla34_30e_1088x608_visdrone_vehicle/'

import os
from flask import Flask, flash, request, redirect, url_for, render_template, Response, send_file, jsonify
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")
#from vietocr.tool.predictor import Predictor
#from vietocr.tool.config import Cfg
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import json
from flask_cors import CORS, cross_origin

phu_input_path = '/home/object_detection/Phu_Demo/ThesisDemo/demo_input/'
phu_output_path = '/home/object_detection/Phu_Demo/ThesisDemo/demo_output/'

UPLOAD_FOLDER = phu_input_path

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/phu/upload', methods=['POST'])
def web_phu():
    start_time = time.time()
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(phu_input_path, filename))
        flash('Image successfully uploaded')
        output_name_img = phu_output_path+str(filename) 
        detector = JDE_Detector(model_dir = phu_checkpoint, device='GPU', output_dir=phu_output_path) 
        out_file = detector.predict_image([phu_input_path + str(filename)],ouput_img=output_name_img)

        online_tlwhs, online_scores, online_ids = out_file[0]

        result_filename =  output_name_img[:-4] + '.txt'
        results = defaultdict(list) 
        #results.append(online_tlwhs,online_scores,online_ids)
        for cls_id in range(1):
          results[cls_id].append((online_tlwhs[cls_id], online_scores[cls_id], online_ids[cls_id]))
        write_mot_results(result_filename, results, data_type='mot', num_classes=1)

        print(str(filename[:-3]+'txt'))
        with open(output_name_img, "rb") as image_file:
          encoded_string = base64.b64encode(image_file.read())
        f = open(result_filename, "r", encoding='utf-8')
        json_text_str = f.read()
        test_time = time.time() - start_time
        return jsonify(image=encoded_string.decode("utf-8"), time=round(test_time, 2), name=str(filename[:-4]),textLocation = json_text_str, title=output_name_img, created = str(datetime.fromtimestamp(start_time)))
        #return jsonify(image=encoded_string.decode("utf-8"), time=test_time, name=out_file.split('/')[-1], textLocation = json_text, title=filename)

        
@app.route('/phu/history', methods=['GET'])
def get_history_phu():
    start_time = time.time()
    predicted_images = os.listdir(phu_output_path)
    images = []
    for img in predicted_images:
      if ("removed" not in img and "txt" not in img and "json" not in img):
        with open(phu_output_path + img, "rb") as image_file:
          encoded_string = base64.b64encode(image_file.read())
          result_filename = open(phu_output_path + img[:-4] + ".txt")
          print(result_filename)
          json_text = result_filename.read()
          result_filename.close()
          json_text_str = str(json_text)
          img_dict = {"name": img[:-4], "created": str(datetime.fromtimestamp(os.path.getctime(phu_output_path + img))), "image": encoded_string.decode("utf-8"), "textLocation": json_text_str}
          images.append(img_dict)
    test_time = time.time() - start_time
    num_history = 10
    return jsonify(images=sorted(images, key=operator.itemgetter('created'), reverse=True)[0 : num_history], total=num_history, time=test_time)

@app.route('/phu/remove', methods=['POST'])
def delete_phu():
    start_time = time.time()
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    #file = request.files['file']
    file_name = request.data['image_name']
    os.rename(os.path.join(phu_output_path, file_name), os.path.join(phu_output_path, file_name + "_removed"))
    return jsonify(noti="Success")
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)

