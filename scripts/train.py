"""
absl -  for building Python applications.
absl.flags - defines a distributed command line system, replacing systems like getopt(), optparse, and manual argument processing

# tensorflow 의 flag =  고정값으로 되어 있는 기본적인 데이터를 편리하게 사용할 수 있습니다.
int, float, boolean, string 의 값을 저장하고, 가져다 사용하기 쉽게 해주는 기능

### ReduceLROnPlateau - Reduce learning rate when a metric has stopped improving.

* scalar - a single number or value
* vector - an array of numbers, either in a row or in a column, identified by an index
* matrix - a 2-D array of numbers, where each element is identified by two indeces 

### EarlyStopping - 
너무 많은 Epoch 은 overfitting 을 일으킨다. 하지만 너무 적은 Epoch 은 underfitting 을 일으킨다. 
Epoch 을 정하는데 많이 사용되는 Early stopping 은 무조건 Epoch 을 많이 돌린 후, 특정 시점에서 멈추는 것이다. 
그 특정시점을 어떻게 정하느냐가 Early stopping 의 핵심이라고 할 수 있다. 일반적으로 hold-out validation set 에서의 성능이 더이상 증가하지 않을 때 학습을 중지시키게 된다. 


Earlystopping 클래스의 구성 요소

Performance measure: 어떤 성능을 monitoring 할 것인가?
Trigger: 언제 training 을 멈출 것인가?


hist = model.fit(train_x, train_y, nb_epoch=10,  

                 batch_size=10, verbose=2, validation_split=0.2,   

                 callbacks=[early_stopping])  

* verbose logging - In software, verbose logging is the practice of recording to a persistent medium as much information as you possibly can about events that occur while the software runs
if you enable verbose > 0, that printing to the screen is generally a very slow process. 
 The algorithm may run an order of magnitude slower, or more, with verbose enabled. 


### ModelCheckpoint- 

Early stopping 객체에 의해 트레이닝이 중지되었을 때, 
그 상태는 이전 모델에 비해 일반적으로 validation error 가 높은 상태일 것이다. 
따라서, Earlystopping 을 하는 것은 특정 시점에 모델의 트레이닝을 멈춤으로써, 
모델의 validation error 가 더 이상 낮아지지 않도록 조절할 수는 있겠지만, 중지된 상태가 최고의 모델은 아닐 것이다. 
따라서 가장 validation performance 가 좋은 모델을 저장하는 것이 필요한데, 
keras 에서는 이를 위해 ModelCheckpoint 라고 하는 객체를 존재한다. 이 객체는 validation error 를 모니터링하면서, 
이전 epoch 에 비해 validation performance 가 좋은 경우, 
무조건 이 때의 parameter 들을 저장한다. 이를 통해 트레이닝이 중지되었을 때,
가장 validation performance 가 높았던 모델을 반환할 수 있다. 


### TensorBoard- 텐서보드는 머신러닝 실험에 필요한 시각화 및 도구를 제공



### yolov3Tiny 

Tiny-yolov3 is a simplified version of YOLOv3, 
which has a much smaller number of convolution layers than YOLOv3, 
which means that tiny-yolov3 does not need to occupy a large amount of memory,
reducing the need for hardware. And it also greatly speeds up detection, 
but lost some of the detection accuracy.



### YoloLoss
YOLO는 각 grid cell마다 다수의 bounding boxes를 예측하지만
true positive에 대한 loss를 계산하기 위해 탐지된 객체를 가장 잘 포함하는 box 하나를 선택해야 한다. 
이를 위해 ground truth와 IOU를 계산하여 가장 높은 IOU를 가진 하나를 선택한다. 
이로써 크기나 가로, 세로 비율에 대해 더 좋은 예측 결과를 얻을 수 있다.

### train 에서 fake_data를 학습시키는 이유 

Even worse it may (and likely is) trying to generalize off of your incorrect examples,
which will lessen the effect your real examples have. 
Essentially, you are just dampening your training set with noise.
--> fake_data도 correct 학습의 일부이다. 


### train에서 suffle를 사용하는 이유 

순차적으로 학습을 시키지만 순서를 임의로 바꾸어서 무작위로 다시 학습을 시키는 과정이 필요하다 .


The process of training a neural network is to find the minimum value of a loss function ℒ𝑋(𝑊),
where 𝑊 represents a matrix (or several matrices) of weights between neurons 
and 𝑋 represents the training dataset. 
I use a subscript for 𝑋 to indicate that our minimization of ℒ occurs only over the weights 𝑊 
(that is, we are looking for 𝑊 such that ℒ is minimized) while 𝑋 is fixed.


### train에서 prefetch() 를 사용하는 이유 
tf.data 파이프라인을 사용해 HDFS에 있는 학습 데이터를 직접 읽어오는 방법을 제공한다. 
Dataset.prefetch() 메서드를 사용하면 학습 데이터를 나눠서 읽어오기 때문에 
첫 번째 데이터를 GPU에서 학습하는 동안 두 번째 데이터를 CPU에서 준비할 수 있어 리소스의 유휴 상태를 줄일 수 있다.


### machine learning 에서 validation set를 사용하는 이유 
모델의 성능을 평가하기 위해서
training을 한 후에 만들어진 모형이 잘 예측을 하는지 그 성능을 평가하기 위해서 사용합니다. 

데이터를 구하고 나서 분석을 시작할 때 대부분 처음 하는 작업은 데이터를 3등분으로 나누는 작업이다.
Train Data
분석 모델을 만들기 위한 학습용 데이터이다.
Validation Data
여러 분석 모델 중 어떤 모델이 적합한지 선택하기 위한 검증용 데이터이다.
Test Data
최종적으로 선택된 분석 모델이 얼마나 잘 작동하는지 확인하기 위한 결과용 데이터이다.

validation dataset = training dataset에서 추출된 가상의 dataset이다.



### Transfer Learning? 

- transfer learning 이란 머신 러닝 기술로서 한 task에서 model 이 train 것이 
다시 두번째로 관련된 task에서 재고 되는것을 말한다. 쉽게 말해, 기존의 만들어진 모델을 사용하여
새로운 모델을 만들시 학습을 빠르게 하여, 예측을 더 높이는 방법을 말한다. 

- 왜 사용하는지 ? 
1. 실질적으로 Convolution network을 처음부터 학습시키는 일은 많지 않다.
대부분의 문제는 이미 학습된 모델을 사용해서 문제를 해결할 수 있습니다.
2. 복잡한 모델일수록 학습시키기 어렵다.
어떤 모델은 2주정도 걸릴수 있으며, 비싼 GPU 여러대를 사용하기도 합니다.
3. layers의 갯수, activation, hyper parameters등등 고려해야 할 사항들이 많으며, 
실질적으로 처음부터 학습시키려면 많은 시도가 필요하다. 

결론적으로 이미 잘 훈련된 모델이 있고, 특히 해당 모델과 유사한 문제를 해결시 transfer learining을 사용한다. 

### Overfitting ? (과적합 )

학습 데이터를 과하게 잘 학습하는 것을 말한다. 
일반적으로 학습데이터는 실제 데이터의 부분집합인 경우가 대부분이다. 
학습 모델이 training set에 존재하는 noise까지 학습하여 test set에서는 정확도가 낮은 상황 

### Regularizaion  ? (일반화 )

overfitting 문제 해결을 위한 대책 

cost function 혹은 error function 이 작아지는 쪽으로 추론을 할 경우에,
단순하게 오차가 작아지는 쪽으로만 진행을 하다 보면 특정 가중치 값들이 커지면서 오히려 결과를 나쁘게 하는 경우도 있다. 
따라서 regularization을 사용하여 더 좋은 학습 결과를 가져 오도록 한다. 

refularization에는 l1, l2 가 있다. 


### 손실함수 ? 

손실함수 (Loss function) ,비용함수 (costfunction)는 실제값과 예측값이 차이가 났을 때, 그 오차가 얼마인지 계산해주는 함수이다. 
우리는 여기서 오차를 줄이도록 신경망을 계속해서 학습시켜나가는 것이 목적이다. 이 오차를 계산하도록 도와주는 함수가 손실함수이다. 
손실함수를 계산하는 방법에는 평균제곱오차, 교차엔트로피 오차 방법 등이 있다. 

"""


from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset



flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny: # yolov3 tiny 일 경우 
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else: # yolov3일 경우 model, anchor, anchor_mask 정하기 
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    if FLAGS.dataset: # dataset 일 경우 
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size) # 이미지, 라벨링, 이미지 사이즈 로드 
    else:
        train_dataset = dataset.load_fake_dataset() # fake_dataset 로드하여 학습
    train_dataset = train_dataset.shuffle(buffer_size=512)  # 데이터셋을 무작위로 shuffle 시키기 
    train_dataset = train_dataset.batch(FLAGS.batch_size) # 일정 batch 크기 만큼 학습 시키기 
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size))) # 데이터셋의 이미지와 target를 변형시키기 
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE) # prefetch를 사용하여 학습 데이터를 나누어서 전송 / 리소스의 유휴상태 줄이기 

    if FLAGS.val_dataset: # val_dataset일 경우(training dataset에서 추출된 가상의 dataset) 
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    else:
        val_dataset = dataset.load_fake_dataset() # dataset에서 fake data 가져온다. 
    val_dataset = val_dataset.batch(FLAGS.batch_size) # dataset에서 batch size 가져온다. 
    val_dataset = val_dataset.map(lambda x, y: ( # 이미지를 변형시킨다. 
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    # Configure the model for transfer learning
    # transfer 학슴을 진행하기 위해 모델 설정하기 
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # top layers 를 다시 설정하기 
        if FLAGS.tiny: # tiny 데이터를 학습 시킬 경우 
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else: # yolov3 데이터를 학습 시킬 경우 
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet': # transfer 이 darknet 일 경우 
            model.get_layer('yolo_darknet').set_weights( # yolo_darknet layer에 가중치 부여 
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet')) # freeze 시켜서 학습 효율 증대 

        elif FLAGS.transfer == 'no_output': # transfer 이 no_output 일 경우 
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)

    else:
        # 모든 다른 transfer 의 경우 matching classes 가 필요하다 
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune': # transfer 이 fine_tune일 경우 
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen': # transfer 이 frozen 일 경우 
            # freeze everything
            freeze_all(model) 

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate) # optimizer 설정 
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes) # 손실 함수 설정 
            for mask in anchor_masks]

    if FLAGS.mode == 'eager_tf':
        # Eager mode 디버깅에 적합하다. 
        # Non eager graph mode 실제 train과정에서 필요로한다. 
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32) # loss 의 평균값 을 rank가 3인 metrix, tf.float32 형으로  선언 
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32) # val_loss 을 rank가 3인 metrix, tf.float32 형으로 선언 

        for epoch in range(1, FLAGS.epochs + 1): # epoch를 설정하여 학습을 반복시킨다. 
            for batch, (images, labels) in enumerate(train_dataset): # batch 사이즈, image,label를 train_dataset 의 크기만큼 돌린다. 
                with tf.GradientTape() as tape: # GradientTape인 tape 를 사용하여 경사하강법 사용하여 손실함수의 최소값을 찾는다. 
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses) # reduce_sum -> 합을 구한다. cf) reduce_mean = 평균을 구한다. 
                    pred_loss = []  # predict box의 손실값을 넣고자 pred_loss 초기화 
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output)) # pred_loss 에 
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss 
                # gradient - 경사 하강 알고리즘 -> Convex 손실 함수에서 최소화 (W,b)값을 구한다. 미분을 사용하여 접선의 기울기를 구해 값을 조정하여 minimize값을 찾는다. 
                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                'checkpoints/yolov3_train_{}.tf'.format(epoch)) # model에 가중치를 저장한다 .
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
