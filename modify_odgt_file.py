f = open("data/training_demo_medium.odgt","w")
import os
for file in os.listdir("data/ADEChallengeData2016/images/training_medium/"):
  dt = '{"fpath_img": "ADEChallengeData2016/images/training_medium/' + file + '", "fpath_segm" : "ADEChallengeData2016/annotations/training_medium/' + file.split(".")[0] + '.png", "width" : 1920, "height": 1080}'
  f.write(dt + '\n')
  
f.close()