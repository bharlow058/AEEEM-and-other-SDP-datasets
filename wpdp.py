# __author__ = 'WeiFu'
from __future__ import print_function, division
import pdb
import random, math
from utility import *


def call(train, test):
  r = round(wekaCALL(train, test), 3)
  if not math.isnan(r):
    return r
  else:
    return 0


def wpdp(train, test):
  result_once = []
  result_once += [call(featureSelection(train, int((data.class_index) * 0.15)), test)]
  result_once += [call(featureSelection(test, int((data.class_index) * 0.15)), train)]
  return result_once


if __name__ == "__main__":
  wpdp()
  # wekaExp()