from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import base64
import time
import operator
from datetime import datetime

# Choose to use a config and initialize the detector
# config ='/home/object_detection/K18_Minh/demo/mmdetection/pth/cascade_double_heads_focal_loss.py'
config = '/home/object_detection/K18_SongThuan/ThesisDemo/model/cdersnet_ga/cdersnet_ga.py'
# Setup a checkpoint file to load
# checkpoint = '/home/object_detection/K18_Minh/demo/mmdetection/pth/best_bbox_mAP_epoch_12.pth'
checkpoint = '/home/object_detection/K18_SongThuan/ThesisDemo/model/cdersnet_ga/best_bbox_mAP.pth'
checkpoint_ocr = '/home/object_detection/K18_SongThuan/ThesisDemo/weights/transformerocr.pth'

import os
from flask import Flask, flash, request, redirect, url_for, render_template, Response, send_file, jsonify
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


input_path = '/home/object_detection/K18_SongThuan/ThesisDemo/demo_input/'
output_path = '/home/object_detection/K18_SongThuan/ThesisDemo/demo_output/'

def detect_func(model, img):
    dict_detection = {}
    score_thr = 0.5
    result = inference_detector(model, img)
    img = img.split('/')[-1] 
    dict_detection[img] = []
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]
    else:
        bbox_result, segm_result = result, None
    
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    
    for cls, bbox in zip(labels, bboxes):
        dict_detection[img].append([str(cls), str((bbox[0])), str((bbox[1])), str((bbox[2])), str((bbox[3])), str((bbox[4]))])
        
    return dict_detection

def infer_reg_func(det_results, img):

  font = ImageFont.truetype("DejaVuSans.ttf", 20)
  color = [(0,0,255), (0,128,0), (255,0,0)]
  image = cv2.imread(img)
  curr_img = Image.open(img)
  draw = ImageDraw.Draw(curr_img)
  
  for bbox in det_results[img.split('/')[-1]]:
      cls = int(bbox[0])
      score = float(bbox[-1])
      if score >= 0.5:
          x1,y1,x2,y2 =  int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3])), int(float(bbox[4]))
          draw.rectangle((x1,y1,x2,y2),outline=color[int(cls)],width = 5)
  
          if cls == 0:
              crop_img = image[y1:y2,x1:x2]
  
              w, h = crop_img.shape[1], crop_img.shape[0]
  
              cv2.imwrite("temp.png", crop_img)
              image_ = Image.open("temp.png")
  
  
              caption = detector.predict(image_)
              print(caption)
              w, h = draw.textsize(caption, font=font)
              if x2 < h:
                  draw.rectangle(
                      (x1 + x2, y1 - 20, x1 + x2 + w, y1 + h),
                      fill=(64, 64, 64, 255)
                  )
                  draw.text(
                      (x1 + x2, y1 - 20),
                      text=caption,
                      fill=(255, 255, 255, 255),
                      font=font
                  )
              else:
                  draw.rectangle(
                      (x1, y1 - 20, x1 + w, y1 - 20 + h),
                      fill=(64, 64, 64, 255)
                  )
                  draw.text(
                      (x1, y1 - 20),
                      text=caption,
                      fill=(255, 255, 255, 255),
                      font=font
                  )
  
  cv2.imwrite(os.path.join(output_path, img.split('/')[-1]), np.asarray(curr_img))

UPLOAD_FOLDER = "/home/object_detection/K18_Minh/demo/mmdetection/pth/web/uploads/"

app = Flask(__name__, template_folder="/home/object_detection/K18_Minh/demo/mmdetection/pth/web/templates/")
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def run_model():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		# print(filename)
        flash('Image successfully uploaded')
        model = init_detector(config, checkpoint, device='cuda:0')
        result = inference_detector(model, UPLOAD_FOLDER + str(filename))
        show_result_pyplot(model, UPLOAD_FOLDER + str(filename), result, score_thr=0.5, out_file="/home/object_detection/K18_Minh/demo/mmdetection/pth/web/static/out.jpg")
        return render_template('upload.html', filename=filename)
        


@app.route('/upload', methods=['POST'])
def mobile():
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
        file.save(os.path.join(input_path, filename))
        flash('Image successfully uploaded')
        print("start")
        # model = init_detector(config_detection, check_point_detection, device='cuda:0')
        model = init_detector(config, checkpoint, device='cuda:0')
        det_results = detect_func(model, input_path + str(filename))
        # result = inference_detector(model, UPLOAD_FOLDER + str(filename))
        print("complete detection")
        #Caption recognition
        config_ocr = Cfg.load_config_from_name('vgg_transformer')
        config_ocr['weights'] = checkpoint_ocr
        config_ocr['cnn']['pretrained']=False
        config_ocr['device'] = 'cuda:0'
        config_ocr['predictor']['beamsearch']=False
        detector = Predictor(config_ocr)
        infer_reg_func(det_results, input_path + str(filename))
        
        print(filename)
        idname = time.time()
        out_file = output_path + str(idname) +"-"+filename
        # show_result_pyplot(model, output_path + str(filename), det_results, score_thr=0.5, out_file=out_file)
        with open(out_file, "rb") as image_file:
          encoded_string = base64.b64encode(image_file.read())
        test_time = time.time() - start_time

        return jsonify(image=encoded_string.decode("utf-8"), time=test_time, name=str(idname) +"-"+filename)

@app.route('/history', methods=['GET'])
def get_history():
    saved_folder = "/home/object_detection/K18_Minh/demo/mmdetection/pth/web/static/"
    start_time = time.time()
    predicted_images = os.listdir(saved_folder)
    images = []
    for img in predicted_images:
      with open(saved_folder + img, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        img_dict = {"name": img, "created": str(datetime.fromtimestamp(os.path.getctime(saved_folder + img))), "image": encoded_string.decode("utf-8")}
        images.append(img_dict)
    test_time = time.time() - start_time
    num_history = 10
    # return jsonify(images=sorted(images, key=operator.itemgetter('created'), reverse=True), total=len(predicted_images), time=test_time)
    return jsonify(images=sorted(images, key=operator.itemgetter('created'), reverse=True)[0 : num_history], total=num_history, time=test_time)
   
if __name__ == "__main__":
    app.run(host='0.0.0.0')