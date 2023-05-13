import scipy.io
import numpy as np
mat = scipy.io.loadmat('data/color150.mat')
mat['colors'][1][0] = 181
mat['colors'][1][1] = 0
mat['colors'][1][2] = 180
scipy.io.savemat('data/color_cgl_realtime.mat', mat)