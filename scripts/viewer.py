"""
from 모듈명 import 함수명 
모듈안의 특정한 함수/ 클래스/ 변수만은 사용하고 싶을 때 

streamlit - Streamlit’s open-source app framework is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours! All in pure Python. All for free.
imuitils - 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
helpers - 반복되는 루틴을 자동화하는 필요한 재사용 가능한 함수들을 포함하고 있다. 
"""
import os
import time
import io
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from absl import flags, logging
from absl.flags import FLAGS

import sys
from absl import flags

from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.utils import draw_outputs


from imutils import opencv2matplotlib
from helpers import byte_array_to_pil_image, get_config, get_now_string
from mqtt import get_mqtt_client
from paho.mqtt import client as mqtt
from PIL import Image
from io import BytesIO

# send to arduino- publish
import paho.mqtt.publish as publish


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID',
                    'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


CONFIG_FILE_PATH = os.getenv("MQTT_CAMERA_CONFIG", "./config/config.yml")
# 환경변수의 키-값 매핑된 값을 가져올 때 getenv를 사용
# config file path에 해당 하는 configuraion를 가져온다.
# config에 broker, port , qos , topic 에 대한 정보를 넣는다.
CONFIG = get_config(CONFIG_FILE_PATH)

MQTT_BROKER = CONFIG["mqtt"]["broker"]
MQTT_PORT = CONFIG["mqtt"]["port"]
MQTT_QOS = CONFIG["mqtt"]["QOS"]

MQTT_TOPIC = CONFIG["save-captures"]["mqtt_topic"]


VIEWER_WIDTH = 800  # viewer  의 크기 지정

global yolo


def get_random_numpy():
    """Return a dummy frame."""
    return np.random.randint(0, 100, size=(32, 32))


title = st.title(MQTT_TOPIC)  # topic으로 st 의 title 정함
#streamlit.image(image, caption=None, width=None, use_column_width=False,clamp=False, channels='RGB', output_format='auto', **kwargs)

# viewer의 값을 초기화한다.  image 를 랜덤값으로 넣고, viewer width를 정한다.
viewer = st.image(get_random_numpy(), width=VIEWER_WIDTH)


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    st.write(
        f"Connected with result code {str(rc)} to MQTT broker on {MQTT_BROKER}"
    )


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    if msg.topic != MQTT_TOPIC:
        return
    # image = byte_array_to_pil_image(msg.payload) # bytearray 로 전달된 msg.payload 값을 pil 이미지로 변환한다.
    # image = image.convert("RGB") # 변환된 pil image를 rgb에 맞게 변환해준다.
    payload = BytesIO(msg.payload)
    snapshot = Image.open(payload)
    image = np.array(snapshot)

    img_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    times = []
    global yolo
    t1 = time.time()
    boxes, scores, classes, nums = yolo.predict(img_in)
    t2 = time.time()
    times.append(t2-t1)
    times = times[-20:]

    x_pos = -1
    image, x_pos = draw_outputs(
        image, (boxes, scores, classes, nums), class_names)
    image = cv2.putText(image, "Time: {:.2f}ms".format(sum(
        times)/len(times)*1000), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    if x_pos != -1:
        print('x_pos : ', x_pos)
    send_xpos_to_arduino(x_pos)
    viewer.image(image, width=VIEWER_WIDTH)  # viewer 객체에 값을 새로 넣어준다.


def send_xpos_to_arduino(x_pos):
    print(f'Send {x_pos}')
    publish.single("x_pos", x_pos, hostname=MQTT_BROKER)
    print(f'Send complete')


def main():
    client = get_mqtt_client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, port=MQTT_PORT)
    client.subscribe(MQTT_TOPIC)

    time.sleep(4)  # Wait for connection setup to complete

    client.loop_forever()


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    # Tensorflow code Starts
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    global yolo
    yolo = YoloV3(classes=80)

    yolo.load_weights(
        '/Users/mkrice/Desktop/REGALA/mqtt-camera-streamer-master/scripts/checkpoints/yolov3.tf')
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(
        '/Users/mkrice/Desktop/REGALA/mqtt-camera-streamer-master/scripts/data/coco.names').readlines()]
    logging.info('classes loaded')

    main()
