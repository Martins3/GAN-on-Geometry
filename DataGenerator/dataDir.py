# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf
import getpass

def clear(logdir):
    if tf.gfile.Exists(logdir):
        tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)

home_dir = None

user_name = getpass.getuser()
if(user_name == "martin"):
    home_dir = "/home/martin/X-Brain/abstraction"
elif(user_name == "zhangmengxiao"):
    home_dir = "/home/zhangmengxiao/hubachelar/data"
elif(user_name == "huxueshi"):
    home_dir = "/home/huxueshi/abstract/data"
else:
    home_dir = "/home/ubuntu/data"

show_data = home_dir + "/show_data"
show_data2 = home_dir + "/show_data2"


ploygons = os.path.join(home_dir, "polygon.npy")


# geometric object
five_star = os.path.join(home_dir, "five_star.npy")  # 所有的形状都是相同的,5格子的圆形
five_star_noise = os.path.join(home_dir, "five_star_noise.npy")
five_star_huge = os.path.join(home_dir, "five_star_huge.npy")
five_star_not_huge = os.path.join(home_dir, "five_star_not_huge.npy")

# lines object
lines = os.path.join(home_dir, "lines.npy")

# tile object
tiles = os.path.join(home_dir, "tiles.npy")


# data make for the VGG
vgg_pictures = os.path.join(home_dir, "vgg.npy")

