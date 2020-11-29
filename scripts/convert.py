from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import load_darknet_weights
import tensorflow as tf

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU') #gpu설정 
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny: # tiny 데이터일 경우 
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else: # yolov3일 경우 
        yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.summary() # 모델 로드 
    logging.info('model created')

    load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny) # 가중치 로드 
    logging.info('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img) #image에 욜로 분석 
    logging.info('sanity check passed')

    yolo.save_weights(FLAGS.output) # 가중치 저장 
    logging.info('weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
