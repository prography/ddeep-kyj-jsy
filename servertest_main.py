#진서연 server test 코드
import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe, Value, Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from take_pic_module import get_pic
import requests
import os,sys,time
import numpy as np


parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name', '-n', default='unknown', type=str, help='input the name of the recording person')
args = parser.parse_args()
from pathlib import Path
server = "http://127.0.0.1:5000/"
data_path = Path('data')
save_path = data_path / 'facebank' / args.name
if not save_path.exists():
    save_path.mkdir()

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)
mtcnn = MTCNN()


def send_face():
    while cap.isOpened():
        img = get_pic()
        URL = server +"register"
        print(URL)
        json_feed = {'face_list':img}
        response = requests.post(URL,json=json_feed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
print('Start Recognition')
send_face()










