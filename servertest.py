#server 코드
from flask import Flask, render_template, jsonify, request
import os, sys, time
# os.chdir(os.path.dirname(__file__))
from PIL import Image,ImageTk
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


app = Flask(__name__)

server = "http://127.0.0.1:5000/"
mtcnn=MTCNN()

@app.route('/register',methods=["POST"])
def register():

    print("--------------")

    register_face = request.json['face_list']
    register_np = np.array(register_face)
    print(register_np)

    return "register success!"

if __name__ =='__main__':
   app.run(host='0.0.0.0', port=5000,debug=True)

