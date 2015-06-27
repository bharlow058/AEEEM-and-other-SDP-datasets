# __author__ = 'WeiFu'
from __future__ import division, print_function
import sys
from utility import *
from wpdp import *
from cpdp import *
from hdp import *


def process(match, target_src, result):
  total = []
  for i in match:
    one_source_result = None
    if i.target_src == target_src:
      one_source_result = [j.result[0] for j in result if
                           j.source_src == i.source_src and j.result != []]  # put all the results from one source
                           # together.
    if not one_source_result:
      continue
    ordered = sorted(one_source_result)
    one_median = ordered[int(len(ordered) * 0.5)]
    print(i.source_src, "===>", target_src, one_median)
    total += [one_median]
  if len(total) == 0:
    print("no results for ", target_src)
    return
  total_median = sorted(total)[int(len(total) * 0.5)]
  print("final ====>", target_src, total_median)
  return total_median


def run():
  datasrc = readsrc()
  source_target_match = KSanalyzer()
  pdb.set_trace()
  for group, srclst in datasrc.iteritems():
    for one in srclst:
      random.seed(1)
      data = loadWekaData(one)
      out_wpdp, out_cpdp, out_hdp = [], [], []  # store results for three methods
      for _ in xrange(500):
        randomized = filter(data, False, "", "weka.filters.unsupervised.instance.Randomize", ["-S", str(_)])
        train = filter(randomized, True, "train", "weka.filters.unsupervised.instance.RemoveFolds",
                       ["-N", "2", "-F", "1", "-S", "1"])
        test = filter(randomized, True, "test", "weka.filters.unsupervised.instance.RemoveFolds",
                      ["-N", "2", "-F", "2", "-S", "1"])
        # out_wpdp += wpdp(tarin, test)
        # cpdp(group,one)
        temp = hdp(one, source_target_match)
        if len(temp) == 0:
          continue
        else:
          out_hdp += temp
      process(source_target_match, one, out_hdp)
      # re_sorted = sorted(out_hdp)
      # print(one, "===>", re_sorted[int(len(re_sorted) * 0.5)])
      # pdb.set_trace()
      # print("next=======>")


if __name__ == "__main__":
  print("hshshhs")
  run()