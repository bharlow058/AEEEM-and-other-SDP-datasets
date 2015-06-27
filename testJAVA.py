from __future__ import print_function, division
__author__ = 'WeiFu'

import jnius_config
jnius_config.add_options('-Xrs', '-Xmx4096')
jnius_config.set_classpath('.', '/Users/WeiFu/Github/HDP/commons-math3-3.5/commons-math3-3.5.jar')
from jnius import autoclass

KStest = autoclass('org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest')
mytest = KStest()
X = [1,2,3,4,5,6,7,8]
Y = [1,2,3,4,5,6,7,8]
result = mytest.kolmogorovSmirnovTest(X,Y)
print(result)









