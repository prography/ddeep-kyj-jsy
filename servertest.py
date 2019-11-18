#server 코드
from flask import Flask, render_template, jsonify, request
import os, sys, time
# os.chdir(os.path.dirname(__file__))
from PIL import Image
from PIL import ImageTk
import json,cv2,argparse,torch
import numpy as np
from PIL import Image
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from datetime import datetime
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from face_recognition import get_face_feature
import argparse

app = Flask(__name__)

server = "http://127.0.0.1:5000/"
parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
args = parser.parse_args()
conf = get_config(False)

register_list = []
learner = face_learner(conf, True)
learner.threshold = args.threshold

@app.route('/register',methods=["POST"])
def register():

    print("------------------------")

    register_face = request.json['face_image']

    register_np = np.array(register_face)
    #아래에 두 방법다 pil로 바껴유...

    pil = Image.fromarray(register_np,'I;16')
    #pil = Image.fromarray(register_np, 'RGB')

    print(type(pil))
    face_feature = get_face_feature(conf, learner.model, pil)
    #register_list.append(face_feature)

    return "register success!"

@app.route('/learn',methods=["POST"])
def register_check():
    print(">>>>>>>>>>>>>")
    face_name = request.json['name']
    #learn_np = np.array(learn_score)
    print(face_name)


    return "register success!"

if __name__ =='__main__':
   app.run(host='0.0.0.0', port=5000,debug=True)

