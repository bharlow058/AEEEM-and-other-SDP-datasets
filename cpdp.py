#__author__ = 'WeiFu'
from __future__ import print_function, division
import pdb
import random
from utility import *
def common(test_src, train_src):
    if not jvm.started: jvm.start()
    loader = Loader(classname="weka.core.converters.ArffLoader")
    train_data = loader.load_file(train_src)
    test_data = loader.load_file(test_src)
    train_data.class_is_last()
    test_data.class_is_last()
    index_test, index_train = [],[]
    for i in range(test_data.class_index):
      for j in range(train_data.class_index):
        if str(test_data.attribute(i))== str(train_data.attribute(j)):
          index_test +=[i]
          index_train +=[j]
    if len(index_train) !=0:
      return [index_test, index_train]
    else:
      return 0

def cpdp(target_group,test_src):
  datasrc = readsrc()
  for source_group, srclst in datasrc.iteritems():
    if source_group == target_group:
      continue
    for one in srclst:
      # test_src = "./exp/test.arff"
      attr_index = common(test_src,one)
      if attr_index == 0:
        continue
      else:
        # pdb.set_trace()
        print(target_group,"<==",source_group,str(attr_index))

      #
      #
      # pdb.set_trace()
      # print(train_data)


