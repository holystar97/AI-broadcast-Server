import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file') # coco data의 name 를 가져온다. 
flags.DEFINE_string('weights', './checkpoints/yolov3.tf', 
                    'path to weights file') # 가중치 
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny') #tiny 모델 
flags.DEFINE_integer('size', 416, 'resize images to') # size 
flags.DEFINE_string('image', './data/girl.png', 'path to input image') # fake image 
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')# tensorflow image
flags.DEFINE_string('output', './output.jpg', 'path to output image') # 결과 image 
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model') # model의 class 갯수 


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU') # gpu 사용하는 경우 
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny: # tiny 모델 사용하는 경우 
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else: # 그외 
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial() # 가중치 로드 
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()] 
    logging.info('classes loaded')

    if FLAGS.tfrecord: # tensorflow에 data넣기 
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0) 
    img = transform_images(img, FLAGS.size) #이미지 변환 

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR) # 원본 데이터를 rgb색상으로 바꿔주기 
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names) # 데이터 위에 box그리기 
    cv2.imwrite(FLAGS.output, img) 
    logging.info('output saved to: {}'.format(FLAGS.output)) 


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    
    
    
    