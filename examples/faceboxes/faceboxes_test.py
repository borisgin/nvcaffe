import numpy as np
import sys, os
import cv2

sys.path.insert(0, '../../python')
import caffe
import time

net_file = 'SSD.prototxt'
caffe_model = 'SSD.caffemodel'
test_dir = "images"

if not os.path.exists(caffe_model):
    print("SSD.caffemodel does not exist, see https://github.com/sfzhang15/SFD")
    exit()
caffe.set_mode_gpu()
net = caffe.Net(net_file, caffe_model, caffe.TEST)

CLASSES = ('background',
           'face')

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel


def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0, 0, :, 3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0, 0, :, 1]
    conf = out['detection_out'][0, 0, :, 2]
    return (box.astype(np.int32), conf, cls)


def detect(imgfile):
    frame = cv2.imread(imgfile)
    transformed_image = transformer.preprocess('data', frame)
    net.blobs['data'].data[...] = transformed_image
    time_start = time.time()
    out = net.forward()
    time_end = time.time()
    print (time_end - time_start),
    print ("s")

    box, conf, cls = postprocess(frame, out)

    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(frame, p1, p2, (0, 255, 0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    cv2.imshow("SSD, %d boxes" % len(box), frame)
    cv2.waitKey()
    # if cv2.waitKey(100) & 0xFF == ord('q'):
    #     break


detect("pepper.jpg")
