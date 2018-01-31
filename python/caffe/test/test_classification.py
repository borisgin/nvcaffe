import unittest
import numpy as np
import tarfile
import tempfile
import zipfile
import shutil
import os
import time

from glob import glob
from google.protobuf import text_format
from PIL import Image
import scipy.misc

# os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
import caffe
from caffe.proto import caffe_pb2

def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
        inputs = {'data': dims}
    )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk

    Returns an np.ndarray (channels x width x height)

    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension

    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

def forward_pass(images, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 16

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in range(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[-1]]
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        print('Processed %s/%s images in %f seconds ...' %
              (len(scores), len(caffe_images), (end - start)))
    return scores


def read_labels(labels_file):
    """
    Returns a list of strings

    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print('WARNING: No labels file provided. Results will be difficult to interpret.')
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels

def unzip_archive(archive):
    """
    Unzips an archive into a temporary directory
    Returns a link to that directory

    Arguments:
    archive -- the path to an archive file
    """
    assert os.path.exists(archive), 'File not found - %s (current %s)' % (archive, os.getcwd())
    tmpdir = os.path.join(tempfile.gettempdir(), os.path.basename(archive))
    assert tmpdir != archive # That wouldn't work out

    if os.path.exists(tmpdir):
        # files are already extracted
        pass
    else:
        if tarfile.is_tarfile(archive):
            print('Extracting tarfile ...')
            with tarfile.open(archive) as tf:
                tf.extractall(path=tmpdir)
        elif zipfile.is_zipfile(archive):
            print('Extracting zipfile ...')
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(path=tmpdir)
        else:
            raise ValueError('Unknown file type for %s' % os.path.basename(archive))
    return tmpdir


class TestClassification(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super(TestClassification, self).setUpClass()
        print('Setting TestClassification...')
        caffe.set_device(0)
        cdir = glob('classification')
        if len(cdir) == 0:
            cdir = glob('*/classification')
            if len(cdir) == 0:
                cdir = glob('*/*/classification')
        cdir = cdir[0]
        imdir = os.path.join(cdir, 'images')
        self.model = os.path.join(cdir, 'model.tar.gz')
        self.img0 = os.path.join(imdir, 'zero.png')
        self.img1 = os.path.join(imdir, 'one.png')
        self.img2 = os.path.join(imdir, 'two.png')
        self.img3 = os.path.join(imdir, 'three.png')
        self.img4 = os.path.join(imdir, 'four.png')
        self.img5 = os.path.join(imdir, 'five.png')
        self.tmpdir = unzip_archive(self.model)

    @classmethod
    def tearDownClass(self):
        super(TestClassification, self).tearDownClass()
        print('TestClassification.tearDownClass')
        shutil.rmtree(self.tmpdir)

    def classify(self, caffemodel, deploy_file, image_files,
                 mean_file=None, labels_file=None, batch_size=None, use_gpu=True):
        """
        Classify some images against a Caffe model and print the results

        Arguments:
        caffemodel -- path to a .caffemodel
        deploy_file -- path to a .prototxt
        image_files -- list of paths to images

        Keyword arguments:
        mean_file -- path to a .binaryproto
        labels_file path to a .txt file
        use_gpu -- if True, run inference on the GPU
        """
        # Load the model and images
        self.net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
        self.transformer = get_transformer(deploy_file, mean_file)
        _, channels, height, width = self.transformer.inputs['data']
        if channels == 3:
            mode = 'RGB'
        elif channels == 1:
            mode = 'L'
        else:
            raise ValueError('Invalid number for channels: %s' % channels)
        images = [load_image(image_file, height, width, mode) for image_file in image_files]
        labels = read_labels(labels_file)

        # Classify the image
        scores = forward_pass(images, self.net, self.transformer, batch_size=batch_size)

        # Process the results
        indices = (-scores).argsort()[:, :5] # take top 5 results
        classifications = []
        for image_index, index_list in enumerate(indices):
            result = []
            for i in index_list:
                # 'i' is a category in labels and also an index into scores
                if labels is None:
                    label = 'Class #%s' % i
                else:
                    label = labels[i]
                result.append((label, round(100.0*scores[image_index, i],4)))
            classifications.append(result)
        return classifications

    def classify_with_archive(self, archive, image_files, batch_size=None):
        tmpdir = unzip_archive(archive)
        caffemodel = None
        deploy_file = None
        mean_file = None
        labels_file = None
        for filename in os.listdir(tmpdir):
            full_path = os.path.join(tmpdir, filename)
            if filename.endswith('.caffemodel'):
                caffemodel = full_path
            elif filename == 'deploy.prototxt':
                deploy_file = full_path
            elif filename.endswith('.binaryproto'):
                mean_file = full_path
            elif filename == 'labels.txt':
                labels_file = full_path
            else:
                pass

        assert caffemodel is not None, 'Caffe model file not found'
        assert deploy_file is not None, 'Deploy file not found'

        return self.classify(caffemodel, deploy_file, image_files,
                        mean_file=mean_file, labels_file=labels_file,
                        batch_size=batch_size)

    def test_all(self):
        classifications = self.classify_with_archive(self.model,
            [self.img0, self.img1, self.img2, self.img3, self.img4, self.img5])
        c0 = ord('0')
        for i in range(0, 6):
            self.assertEqual(classifications[i][0][0], chr(c0 + i))
            self.assertGreater(classifications[i][0][1], 99.9)
