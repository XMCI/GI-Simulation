import time
from GI_SimDemo_phant_mod4 import *

if __name__ == '__main__':
    start_time = time.time()
    GI_SimDemo_phant_mod4('configTest')
    print
    print('---- %s seconds ----' % (time.time() - start_time))

