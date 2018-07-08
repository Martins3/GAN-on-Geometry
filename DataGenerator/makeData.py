# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import tensorflow as tf
import dataDir as dd
import utils
import math
import time
import matplotlib.pyplot as plt
from skimage import draw
from scipy import misc
from scipy import ndimage
import skimage



def make_polygons(dataset_size = 800, labels = 8, regular = True,
                  edge_style = "mosaic", centeral = False, cramming = False,
                  image_size = (60, 60), radius_range = (25, 29)):
    """
    默认的, 生成的位置都是在中间的,也就是多数的位置全部都是黑色的, 不采用填充的
    """
    picture_base = np.zeros((dataset_size, image_size[0], image_size[1]), dtype = np.float32)
    doubel_pi = 2 * math.pi
    assert dataset_size % labels == 0 # 必须保证所有的label的个数相同


    label_size = dataset_size // labels # label_size 表示为多边形生成的个数
    for i in range(label_size):
        center = (image_size[0] // 2, image_size[1] //2) # 初始化 center
        start_angle = doubel_pi * np.random.random() # 起始角度
        random_radius = np.random.randint(radius_range[0], radius_range[1])
        centeral_bias_x = 0
        centeral_bias_y = 0
        if(not centeral):
            centeral_bias_x = np.random.randint(-2, 3)
            centeral_bias_y = np.random.randint(-2, 3)
        center = (center[0] + centeral_bias_x, center[1] + centeral_bias_y)

        # 创建数据集合
        for p in range(labels):
            rotation = doubel_pi / (p + 3) # 旋转的角度
            vertex = []  # 收集顶点
            begin_vertex = ()
            for v in range(p + 3):
                angle = start_angle + v * rotation
                a, b = int(random_radius * math.cos(angle)), int(random_radius * math.sin(angle))
                pos = (center[0] + a, center[1] + b)
                if(v == 0):
                    begin_vertex = pos
                vertex.append(pos)
            vertex.append(begin_vertex)

            # 确定位置,
            if(not cramming):
                for j in range(p + 3):
                    a = vertex[j]
                    b = vertex[j + 1]
                    for x in range(image_size[0]):
                        for y in range(image_size[0]):
                            distance = utils.distance_to_segment_line((x, y), a, b)
                            if(edge_style == "sharp"):
                                if(distance <= math.sqrt(2) / 2 + 0.001):
                                    picture_base[i * labels + p][x][y] = 1
                            elif(edge_style == "mosaic"):
                                if(distance <= math.sqrt(2) / 2 + 0.001):
                                    picture_base[i * labels + p][x][y] = 1
                                else:
                                    if(picture_base[i * labels + p][x][y] !=1):
                                        picture_base[i * labels + p][x][y] = picture_base[i * labels + p][x][y] +  1 - math.tanh(distance)
                            else:
                                print("https://stackoverflow.com/\
                                      questions/1119627/how-to-test-if-a-point-is\
                                      -inside-of-a-convex-polygon-in-2d-integer-coordinates")
                                raise NotImplemented

            else:
                raise NotImplemented

    # 为数据添加标签, 并且转化为 [-1, 28 ** 2 + 1] 的形式
    # 将得到的数据进行存储
    signatures = np.asarray([x % labels for x in range(dataset_size)],dtype = np.float32)
    signatures = np.reshape(signatures, [dataset_size, 1])
    picture = np.reshape(picture_base, [dataset_size, image_size[0] * image_size[1]])
    data = np.append(picture, signatures, 1)
    np.save(dd.ploygons, data)


# 如何处理的随机的位置
# 半径大小的确定
# 填充的数目的确定的
# 如何放置碰撞
def make_geomeric_objects(dataset_size = 10000, hole = False,
                 edge_style = "sharp", num_range = (1, 11),
                 image_size = (28, 28), single_shape = False,
                 object_shape = "circle", locations = dd.five_star,
                 noise= False, guassian = False):
    """
    所有的圆形不应该和边界相切, circle 之间也是不应该相切的, 使用sharp 的边界,
    位置全部随机, 半径的大小随机,使用radius_range的位置
    暂时模式使用圆形, 所谓的圆形只有两种形式
    """
    picture = np.zeros((dataset_size, image_size[0], image_size[1]), dtype = np.float32)
    labels = num_range[1] - num_range[0]
    assert dataset_size % labels == 0 # 相同, 需要设置所有的数值都是相同的
    label_size = dataset_size // labels

    for epoch in range(label_size):
        # 首先创建最简单的内容, 全部都是 5 单元的

        for object_num in range(num_range[0], num_range[1]):
            pos = []
            while(len(pos) != object_num):
                x = np.random.randint(1, image_size[0]-2) # 边界位置不可以靠近
                y = np.random.randint(1, image_size[1] -2)
                pos = utils.collison_five_circle((x, y), pos)
            # 进行绘制图形
            for i in pos:
                picture_index = epoch * labels + object_num - num_range[0]
                picture[picture_index][i[0]][i[1]] = 1
                picture[picture_index][i[0] - 1][i[1]] = 1
                picture[picture_index][i[0]][i[1] - 1] = 1
                picture[picture_index][i[0] + 1][i[1]] = 1
                picture[picture_index][i[0]][i[1] + 1] = 1
            # 添加噪音和模糊
            if(guassian):
                picture[picture_index] = \
                ndimage.gaussian_filter(picture[picture_index], sigma=0.5)
            if(noise):
                picture[picture_index] = \
                 skimage.util.random_noise(picture[picture_index],
                 mode='gaussian', seed=None, clip=True, var = 0.02)

    # 毫无新意的部分, 注意标签的不代表数值
    signatures = np.asarray([x % labels for x in range(dataset_size)],dtype = np.float32)
    signatures = np.reshape(signatures, [dataset_size, 1])
    picture = np.reshape(picture, [dataset_size, image_size[0] * image_size[1]])
    data = np.append(picture, signatures, 1)
    np.save(locations, data)


def make_lines(dataset_size = 10000, num_range = (1, 11), width = 2,
               length_range = (14, 20) , parallel = False, noise = True,
               curving = False, average = True, image_size =(60, 60)):
    # more data style haven been implemented
    picture = np.zeros((dataset_size, image_size[0], image_size[1]), dtype = np.float32)
    labels = num_range[1] - num_range[0] # 默认的条数为 10 labels
    double_pi = math.pi * 2
    assert dataset_size % labels == 0 # 老规则

    # 找到整体赋值的方法是什么
    label_size = dataset_size // labels
    for iter_index in range(label_size):
        for line_num in range(*num_range):
            valid_angles = []
            anti_clock = 0

            while(len(valid_angles) != line_num):
                anti_clock = anti_clock + 1
                if(anti_clock == 100000):
                    valid_angles = []

                length = np.random.randint(*length_range)
                a, b = np.random.randint(0, 60), np.random.randint(0, 60)
                portion = np.random.random()
                thta = portion * double_pi
                a2 = a + int(length * math.sin(thta))
                b2 = b + int(length * math.cos(thta))

                while(utils.out_of_boundary((a2, b2), image_size= image_size)):
                    portion = np.random.random()
                    thta = portion * double_pi
                    a2 = a + int(length * math.sin(thta))
                    b2 = b + int(length * math.cos(thta))
                old_len = len(valid_angles)
                valid_angles = utils.check_angle(thta, valid_angles)
                new_len = len(valid_angles)
                if(old_len!=new_len):
                    rr, cc = draw.line(a, b, a2, b2)
                    pic_index = iter_index * labels + line_num - num_range[0]
                    picture[pic_index][rr, cc] = 1
                    picture[pic_index] = ndimage.gaussian_filter(picture[pic_index], sigma=0.5)

    signatures = np.asarray([x % labels for x in range(dataset_size)],dtype = np.float32)
    signatures = np.reshape(signatures, [dataset_size, 1])
    picture = np.reshape(picture, [dataset_size, image_size[0] * image_size[1]])
    data = np.append(picture, signatures, 1)
    np.save(dd.lines, data)



def make_tiles(dataset_size = 10000, size_range = (7, 20), blurry = False,
               noise = False, piece_num = 2, image_size = (28, 28),
               rotate = False, depth = 3):

    def in_range(point):
        if(point[0] < 0 or point[0] >= image_size[0] or point[1] < 0 or point[1] > image_size[1]):
            return False
        else:
            return True

    def rec_not_intersect(pointA, pointAP, rec_a, rec_b):
        a = (pointA[0], pointAP[1])
        b = (pointAP[0], pointA[1])
        def e_not_intersect(p):
            return not_intersect(p, rec_a, rec_b)
        if(e_not_intersect(a) and e_not_intersect(b) and e_not_intersect(pointA) and e_not_intersect(pointAP)):
            return True
        return False

    def not_intersect(point, rect_a, rect_b):
        width_range = (min(rect_a[0], rect_b[0]), max(rect_a[0], rect_b[0]))
        length_range = (min(rect_a[1], rect_b[1]), max(rect_a[1], rect_b[1]))
        return (point[0] < width_range[0] - 1 or point[0] > width_range[1] + 1) and  (point[1] < length_range[0] - 1 or point[1] > length_range[1] + 1)

    def random_direction(point, width, length, rd = None):
        if(rd == None):
            rd = np.random.randint(0, 4)

        if(rd == 0):
            return (point[0] + width, point[1] + length)
        elif(rd == 1):
            return (point[0] + width, point[1] - length)
        elif(rd == 2):
            return (point[0] - width, point[1] + length)
        else:
            return (point[0] - width, point[1] - length)

    def generate_random_point(corner = False):
        if(not corner):
            return  (np.random.randint(1,image_size[0] - 1), np.random.randint(1, image_size[1] - 1))
        else:
            seed = np.random.randint(0, 4)
            score = 10
            if(seed == 0):
                return  (np.random.randint(1, score), np.random.randint(1, score))
            elif(seed == 1):
                return  (np.random.randint(1, score), np.random.randint(image_size[1] - score, image_size[1] - 1))
            elif(seed == 2):
                return  (np.random.randint(image_size[0] - score, image_size[0] - 1), np.random.randint(1, score))
            else:
                return  (np.random.randint(image_size[0] - score, image_size[0] - 1), np.random.randint(image_size[1] - score, image_size[1] - 1))

    picture = np.zeros((dataset_size, image_size[0], image_size[1]), dtype = np.float32)
    # 由于只有两个类别, 偶数正确 计数的时候错误
    iter_num = 0
    while(iter_num < dataset_size):
        print("iter_num : ", iter_num)

        # right = True if(iter_num % 2 == 0) else False
        right = True # 需要全部的数据都是真实数据

        # 每一次创建的时候, 使用不相同的size
        width = np.random.randint(*size_range)
        length = np.random.randint(*size_range)

        # 初始化 pointA
        pointA = generate_random_point(corner = True)
        pointAP = random_direction(pointA, width, length)
        # 获取创建第一个点
        while(not in_range(pointAP)):
            pointA = generate_random_point(corner = True)
            pointAP = random_direction(pointA, width, length)

        pointB = ()
        pointBP = ()


        fail = 0
        while(fail < 10):
            fail = fail + 1
            if(fail == 10):
                break

            pointB = generate_random_point() # 生成第一个valid 的点
            while(not not_intersect(pointB, pointA, pointAP)): # 如果创建的位置本身在矩形中间
                pointB = generate_random_point()

            break_signal = False
            for i in range(4):
                pointBP = random_direction(pointB, width, length, rd = i)
                if(in_range(pointBP) and rec_not_intersect(pointB, pointBP, pointA, pointAP)):
                    break_signal = True
                    break

            if(break_signal):
                break

        if(fail == 10):
            continue

        # if(iter_num < 10):
            # print(pointA, pointAP, pointB, pointBP)
        # else:
            # return

        # 重做 point的表示, 使用左上方的点
        point_A = (min(pointA[0], pointAP[0]), max(pointA[1], pointAP[1]))
        point_AP = (max(pointA[0], pointAP[0]), min(pointA[1], pointAP[1]))
        point_B = (min(pointB[0], pointBP[0]),  max(pointB[1], pointBP[1]))
        point_BP = (max(pointB[0], pointBP[0]), min(pointB[1], pointBP[1]))

        if(abs(pointA[0] - pointAP[0]) == abs(pointB[0] - pointBP[0])):
            # 方向相同, step 需要保证为 大于0 的
            # 添加不同深度的基底
            base = np.random.randint(1, 4)
            choice_side = np.random.random() > 0.5
            if(np.random.random() > 0.5):
                # 从第一个维度
                step = 0
                for index in range(point_AP[0] - point_A[0]):
                    picture[iter_num, point_A[0] + index, point_AP[1]: point_A[1] - step] = 1
                    cor_step = step if(right) else np.random.randint(0, depth)

                    if(choice_side):
                        picture[iter_num, point_B[0] + index, point_B[1] - cor_step - base: point_B[1]] = 1
                    else:
                        picture[iter_num, point_B[0] + index, point_BP[1]: point_BP[1] + base + cor_step] = 1

                    goto = np.random.randint(-1, 2)
                    step = step + goto
                    if(step < 0 or step > depth):
                        step = depth // 2

            else:
                # 从第二个维度
                step = 0
                for index in range(point_A[1] - point_AP[1]):
                    picture[iter_num, point_A[0]:(point_AP[0] - step), point_AP[1] + index] = 1
                    cor_step = step if(right) else np.random.randint(0, depth)


                    if(choice_side):
                        picture[iter_num, point_B[0]: point_B[0] + cor_step + base, point_BP[1] + index] = 1
                    else:
                        picture[iter_num, point_BP[0] - cor_step - base: point_BP[0], point_BP[1] + index] = 1

                    goto = np.random.randint(-1, 2)
                    step = step + goto
                    if(step < 0 or step > depth):
                        step = depth // 2

        else:
            # 方向不同, 由于随机的关系, A 从 维度1 , B 从维度 2

            step = 0
            for index in range(point_A[1] - point_AP[1]):
                picture[iter_num, point_A[0]:point_AP[0] - step, point_AP[1] + index] = 1
                cor_step = step if(right) else np.random.randint(0, depth)


                if(choice_side):
                    picture[iter_num, point_B[0] + index, point_B[1] - cor_step - base: point_B[1]] = 1
                else:
                    picture[iter_num, point_B[0] + index, point_BP[1]: point_BP[1] + base + cor_step] = 1

                goto = np.random.randint(-1, 2)
                step = step + goto
                if(step < 0 or step > depth):
                    step = depth // 2

        iter_num = iter_num + 1
    labels = 2
    signatures = np.asarray([x % labels for x in range(dataset_size)],dtype = np.float32)
    signatures = np.reshape(signatures, [dataset_size, 1])
    picture = np.reshape(picture, [dataset_size, image_size[0] * image_size[1]])
    data = np.append(picture, signatures, 1)
    np.save(dd.tiles, data)


def make_data(data_dir):
    np.random.seed(41)
    if(data_dir == dd.five_star):
        make_geomeric_objects()
    elif(data_dir == dd.tiles):
        make_tiles()
    elif(data_dir == dd.lines):
        make_lines()
    elif(data_dir == dd.ploygons):
        make_polygons()
    elif (data_dir == dd.five_star_noise):
         make_geomeric_objects(dataset_size = 20000, guassian = True,
    noise= True, locations = dd.five_star_noise)
    elif(data_dir == dd.five_star_huge):
        make_geomeric_objects(dataset_size = 30000 ,image_size = (120, 120), locations = dd.five_star_huge,
        num_range = (1, 51))
    elif(data_dir == dd.five_star_not_huge):
        make_geomeric_objects(dataset_size = 30000 ,image_size = (120, 120), locations = dd.five_star_not_huge,
        num_range = (1, 31), guassian = True, noise = True)
    elif(data_dir == dd.vgg_pictures):
        vgg_make_data()
    else:
        raise NotImplemented



def test_tensor():
    num = 800
    size = 60
    data_dir = dd.ploygons
    show_dir = dd.show_data + "/polygon/"

    dd.clear(show_dir)
    tensor = tf.placeholder(tf.float32, [num, size * size], name = "t")
    tf.summary.image("x_generated", tf.reshape(tensor, [-1, size, size, 1]), max_outputs= 10)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(show_dir)

    a = np.load(data_dir)
    a = a[:, 0:-1]

    # mnist
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets('/home/martin/X-Brain/Martins3.github.io/source/_posts/nonsense/ML/project/MNIST_data', one_hot=True)
    # x_value, _ = mnist.train.next_batch(100)
    # a = x_value

    with tf.Session() as sess:
        summary = sess.run(merged, feed_dict={tensor:a})
        writer.add_summary(summary, 0)
    writer.close()


def test_Generators():
    num_range = (1, 31)
    data_size = 300 # 4

    make_geomeric_objects(dataset_size = 300 ,image_size = (120, 120), locations = dd.five_star_not_huge,
        num_range = (1, 31), guassian = True, noise = True)
    input_data = np.load(dd.five_star_not_huge) # 1
    print(input_data.shape)
    size = 120 #5
    epoch_size =  30 # 表示一波含有的图片

    dd.clear(dd.show_data) # !!

    image = tf.placeholder(tf.float32, [epoch_size, size, size, 1], name = "image")
    tf.summary.image('five_star_not_huge_20', image, max_outputs = epoch_size) # 2

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter(dd.show_data + "/five_star_not_huge_20") # 3
    with tf.Session() as sess:
        for i in range(data_size // epoch_size):
            data = input_data[i * epoch_size: (i + 1) * epoch_size,0: size * size]

            data = np.reshape(data, (epoch_size, size, size, 1))
            summary = sess.run(merged, feed_dict = {image:data})
            writer.add_summary(summary, i)
    writer.close()




def make_one_line():
    image_size = (60, 60)
    length = 18
    double_pi = math.pi * 2

    def out_of_boundary(point):
        if(point[0] < 0 or point[0] >= image_size[0] or point[1] < 0 or point[1] >= image_size[1]):
            return True
        return False

    def check_angle(target, valid):
        if(not valid):
            valid.append(target)
            return valid
        for i in valid:
            if(abs(i - target) < double_pi / 15):
                return valid
        valid.append(target)
        return valid


    picture = np.zeros(image_size, dtype = np.float32)

    valid_angles = []
    while(len(valid_angles) != 10):
        length = np.random.randint(*length_range)
        a, b = np.random.randint(0, 60), np.random.randint(0, 60)
        portion = np.random.random()
        thta = portion * double_pi
        a2 = a + int(length * math.sin(thta))
        b2 = b + int(length * math.cos(thta))

        while(out_of_boundary((a2, b2))):
            portion = np.random.random()
            thta = portion * double_pi
            a2 = a + int(length * math.sin(thta))
            b2 = b + int(length * math.cos(thta))
        old_len = len(valid_angles)
        valid_angles = check_angle(thta, valid_angles)
        new_len = len(valid_angles)
        if(old_len!=new_len):
            rr, cc = draw.line(a, b, a2, b2)
            picture[rr, cc] = 1
    valid_angles = [int(angles / double_pi * 360) for angles in valid_angles]
    list.sort(valid_angles)
    print(valid_angles)


    picture = ndimage.gaussian_filter(picture, sigma=0.8)
    # picture = ndimage.uniform_filter(picture, size=2) 放弃使用
    # picture = ndimage.median_filter(picture, 1)

    plt.imshow(picture, cmap = "gray")

if __name__ == "__main__":
    a = time.time()
    np.random.seed(41)
    # make_geomeric_objects()
    # make_tiles()
    # make_geomeric_objects()
    test_tensor()
    print(time.time() - a)
