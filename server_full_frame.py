# server 코드

from PIL import Image
import argparse
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, prepare_facebank

import numpy as np

from flask import Flask, render_template, jsonify, request
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from face_recognition import get_face_feature, get_max_cos  # 11.16

app = Flask(__name__)

server = "http://127.0.0.1:5000/"
# 11.16
parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("-s", "--save", help="whether save", action="store_true")
parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
args = parser.parse_args()
conf = get_config(False)

register_list = []
mtcnn = MTCNN()
print("mtcnn loaded")
learner = face_learner(conf, True)

learner.threshold = args.threshold
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')
if args.update:
    targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
    print('facebank updated')
else:
    targets, names = load_facebank(conf)
    print('facebank loaded')


# 여기서 frame을 받아서 frame에서 얼굴 detection + blur칠수있도록 bboxes 데이터까지 보내주기.
@app.route('/getframe', methods=["POST"])
def getframe():
    # 여기서 frame받음. 얼굴 detection
    print("===============")

    get_frame = request.json['frame_to_server']
    # print(type(get_frame))

    np_frame = np.array(get_frame)
    print(np_frame)
    pil_frame = Image.fromarray(np_frame, mode='RGB')

    bboxes, faces = mtcnn.align_multi(pil_frame, conf.face_limit, conf.min_face_size)

    return 'success!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

