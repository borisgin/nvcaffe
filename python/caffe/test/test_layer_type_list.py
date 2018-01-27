import unittest

import caffe

class TestLayerTypeList(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super(TestLayerTypeList, self).setUpClass()
        caffe.set_device(0)
        print('TestLayerTypeList.setUpClass')

    @classmethod
    def tearDownClass(self):
        super(TestLayerTypeList, self).tearDownClass()
        print('TestLayerTypeList.tearDownClass')

    def test_standard_types(self):
        #removing 'Data' from list 
        for type_name in ['Data', 'Convolution', 'InnerProduct']:
            self.assertIn(type_name, caffe.layer_type_list(),
                    '%s not in layer_type_list()' % type_name)
