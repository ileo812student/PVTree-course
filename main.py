import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from args import parse_args
import random
from tools import min_distance_point_to_line, \
    show_img_2d, draw_segment2d_rotate
from class_v_s import TPoint, Segment
from kamiya import get_random_position_c1, kamiya_optimal
from crop_veinline import crop
from data_tools import cross_folder, aug, cvt
import time
import json
import hashlib
from multiprocessing import Pool
from PIL import Image
import os


def create_finger_position(case):
    if case == 1:
        finger_x = [0, 3, 7, 20, 25, 37, 41, 58, 65, 70]
        finger_z = [52, 68, 71, 76, 80, 80, 80, 76, 71, 50]
    elif case == 2:
        finger_x = [0, 8, 13, 23, 30, 40, 47, 60, 65, 70]
        finger_z = [58, 70, 72, 76, 80, 80, 80, 76, 72, 48]
    elif case == 3:
        finger_x = [0, 4, 9, 18, 26, 36, 43, 58, 65, 70, 53, 65]
        finger_z = [56, 64, 66, 75, 77, 80, 80, 80, 72, 48, 42, 34]
    elif case == 4:
        finger_x = [0, 6, 10, 25, 30, 40, 48, 60, 66, 70]
        finger_z = [56, 68, 72, 75, 80, 80, 80, 75, 70, 50]


    finger_position_list = []
    for i in range(len(finger_x)): 
        bias_x = random.uniform(-1.5, 1.5)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-2, 2)
        finger_position = np.array([finger_x[i] + bias_x, 40 + bias_y, finger_z[i] + bias_z])      
        finger_position_list.append(finger_position)

    return finger_position_list

def create_main_tree_c1(args, segment):
# c1
    list_root_point = []
    
    position_r_x = np.array([33.0, 22.0, 18.0, 25.0, 30.0, 38.0, 46.0, 50.0, 45.0, 38.0])
    position_r_z = np.array([-3.0, 30.0, 42.0, 50.0, 55.0, 58.0, -3.0, 31.0, 45.0, 58.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=9)
        elif i == 2:
            point_r = TPoint(args, position_r, num_end=8)
        elif i == 3:
            point_r = TPoint(args, position_r, num_end=6)
        elif i == 4 or i==8:
            point_r = TPoint(args, position_r, num_end=4)
        elif i == 5 or i==9:
            point_r = TPoint(args, position_r, num_end=2)
        elif i == 6 or i == 7:
            point_r = TPoint(args, position_r, num_end=5)
        
        list_root_point.append(point_r)
        

   
    for i in range(1, 6):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(7, len(list_root_point)):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

   
    s_finger_point = list_root_point[1: 2] + list_root_point[2: 6] + list_root_point[8:9] + list_root_point[7:8]

   
    list_branch_finger_point = []
    
    position_bf_x = np.array([5, 6, 15, 28, 47, 58, 70])
    position_bf_z = np.array([23, 55, 66, 73, 72, 53, 30])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        if i == 0 or i == 6:
            point_bf = TPoint(args, position_bf, num_end=1)
        else:
            point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)


    finger_position = create_finger_position(case=1)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[1], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[2], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[3], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point

    for i in range(len(list_move)):
        list_move[i].parent = s_finger_point[i]
        seg = Segment(s_finger_point[i], list_move[i])
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE

def create_main_tree_c2(args, segment):
# c2
    list_root_point = []
    
    position_r_x = np.array([30.0, 25.0, 20.0, 22.0, 33.0, 41.0, 51.0, 50.0, 47.0])
    position_r_z = np.array([-5.0, 24.0, 35.0, 52.0, 58.0, -3.0, 27.0, 39.0, 58.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=7)
        elif i == 2:
            point_r = TPoint(args, position_r, num_end=6)
        elif i == 5 or i == 6:
            point_r = TPoint(args, position_r, num_end=5)    
        elif i == 3 or i == 7:
            point_r = TPoint(args, position_r, num_end=4)
        else:
            point_r = TPoint(args, position_r, num_end=2)
        
        list_root_point.append(point_r)
      
    for i in range(1, 5):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(6, len(list_root_point)):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    s_finger_point = list_root_point[1: 5] + list_root_point[8:9] + list_root_point[7:8] + list_root_point[6:7]

    list_branch_finger_point = []
    position_bf_x = np.array([5, 6, 18, 34, 48, 60, 70])
    position_bf_z = np.array([13, 56, 68, 70, 70, 45, 30])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        if i == 0 or i == 6:
            point_bf = TPoint(args, position_bf, num_end=1)
        else:
            point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)

    finger_position = create_finger_position(case=2)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[1], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[2], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[3], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point

    for i in range(len(list_move)):
        list_move[i].parent = s_finger_point[i]
        seg = Segment(s_finger_point[i], list_move[i])
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE

def create_main_tree_c3(args, segment):
# c3
    list_root_point = []
    position_r_x = np.array([32.0, 20.0, 28.0, 36.0, 46.0, 60.0, 42.0, 45.0])
    position_r_z = np.array([-3.0, 40.0, 50.0, 57.0, 60.0, 65.0, -3.0, 25.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=10)
        elif i == 2:
            point_r = TPoint(args, position_r, num_end=8)
        elif i == 3:
            point_r = TPoint(args, position_r, num_end=6)    
        elif i == 4:
            point_r = TPoint(args, position_r, num_end=4)
        else:
            point_r = TPoint(args, position_r, num_end=2)
        
        list_root_point.append(point_r)
       
    for i in range(1, 6):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(7, len(list_root_point)):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    s_finger_point = list_root_point[1: 6] + list_root_point[7:8]

    list_branch_finger_point = []
    position_bf_x = np.array([5, 16, 33, 50])
    position_bf_z = np.array([57, 64, 72, 72])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)
    list_branch_finger_point.append(s_finger_point[-2])
    list_branch_finger_point.append(s_finger_point[-1])
    finger_position = create_finger_position(case=3)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[0]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[0], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[1], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[2], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[3], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 10 or i == 11:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point[0:4]

    for i in range(len(list_move)):
        list_move[i].parent = s_finger_point[i]
        seg = Segment(s_finger_point[i], list_move[i])
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE

def create_main_tree_c4(args, segment):
# c4
    list_root_point = []
    position_r_x = np.array([30.0, 22.0, 42.0, 45.0, 40.0, 30.0, 42.0])
    position_r_z = np.array([-3.0, 30.0, -3.0, 25.0, 45.0, 50.0, 54.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=1)
        elif i == 2 or i ==3:
            point_r = TPoint(args, position_r, num_end=10)
        elif i == 4:
            point_r = TPoint(args, position_r, num_end=8)    
        else:
            point_r = TPoint(args, position_r, num_end=4)
        
        list_root_point.append(point_r)
       
    for i in range(1, 2):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(3, len(list_root_point)-1):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)
    list_root_point[-1].parent = list_root_point[-3]
    segg = Segment(list_root_point[-3], list_root_point[-1])
    segment.append(segg)

    s_finger_point = list_root_point[1: 2] + list_root_point[5:7] + list_root_point[3: 4]

    list_branch_finger_point = []
    position_bf_x = np.array([0, 19, 20, 36, 50, 57])
    position_bf_z = np.array([38, 60, 70, 74, 72, 53])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        if i == 0:
            point_bf = TPoint(args, position_bf)
        else:
            point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)
   
    finger_position = create_finger_position(case=4)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[1], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[2], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[3], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point

    for i in range(len(list_move)):
        if i == 0:
            list_move[i].parent = s_finger_point[0]
            seg = Segment(s_finger_point[i], list_move[i])
        elif i == 1 or i == 2:
            list_move[i].parent = s_finger_point[1]
            seg = Segment(s_finger_point[1], list_move[i])
        elif i == 3 or i == 4:
            list_move[i].parent = s_finger_point[2]
            seg = Segment(s_finger_point[2], list_move[i])
        elif i == 5:
            list_move[i].parent = s_finger_point[3]
            seg = Segment(s_finger_point[3], list_move[i])
        
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE


def save_subfig(fig, ax, save_path):
    ax.axis('tight')
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path, bbox_inches=bbox, pad_inches=0)

def flip_image(image_path, output_path):
    
    img = Image.open(image_path)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(output_path)

def _identity_seed(seed, case, index):
    payload = f'{seed}|{case}|{index}'.encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little')


def create_3dtree(args, case=1, identity_index=0, fullpath='', croppath='', s=1,
                  num_sams=7, resume=False, force=False):
    expected = [fullpath + '_sample{}.png'.format(i) for i in range(num_sams)]
    expected_crop = [croppath + '_sample{}.png'.format(i) for i in range(num_sams)]
    if resume and all(os.path.isfile(p) for p in expected + expected_crop):
        return 'skipped'
    existing = [p for p in expected + expected_crop if os.path.exists(p)]
    if existing and not (resume or force):
        raise FileExistsError('Refusing to overwrite existing identity output: {}'.format(existing[0]))
    identity_seed = _identity_seed(args.seed, case, identity_index)
    random.seed(identity_seed)
    np.random.seed(identity_seed % (2 ** 32))
    segment = []
    randnum = random.randint(-10, 10)
    num = 80
    # num = 100 + randnum
    # num = 70 + randnum
    
    if case == 1:
        create_main_tree_c1(args, segment)
    elif case == 2:
        create_main_tree_c2(args, segment)
    elif case == 3:
        create_main_tree_c3(args, segment)
    elif case == 4:
        create_main_tree_c4(args, segment)

    for i in range(num):
        position = get_random_position_c1(args)
        new_point = TPoint(args, position, args.r_ori)
        min_seg = min_distance_point_to_line(position, segment)
        kamiya_optimal(args, new_point, min_seg, segment)

    for sam in range(num_sams):
        full_sample = fullpath+'_sample{}.png'.format(sam)
        crop_sample = croppath+'_sample{}.png'.format(sam)
        # Consume one projection angle for every sample, including resume skips,
        # so a partially resumed identity matches an uninterrupted run exactly.
        ang = np.radians(random.randint(-2, 2))
        if resume and os.path.isfile(full_sample) and os.path.isfile(crop_sample):
            continue
        if resume and os.path.isfile(full_sample) and not os.path.isfile(crop_sample):
            crop(full_sample, 'c{}'.format(case), crop_sample)
            continue
        fig = plt.figure(figsize=(7, 8))
        ax = fig.add_subplot(111)
        draw_segment2d_rotate(args, ax, segment, angle=ang, s=s) 
        show_img_2d(ax)
        fig.savefig(full_sample, bbox_inches='tight', pad_inches=0)
        plt.close()
        if not (resume and os.path.isfile(crop_sample)):
            crop(full_sample, 'c{}'.format(case), crop_sample)
    return 'generated'


def _generate_identity_task(task):
    args, case, index, fullpath, croppath, sams, resume, force = task
    return create_3dtree(args, case=case, identity_index=index, fullpath=fullpath,
                         croppath=croppath, num_sams=sams, resume=resume, force=force)


def get_large_pv(args, fp, cp, num_id, sams, resume=False, force=False):
    num = num_id // 4
    os.makedirs(fp, exist_ok=True)
    os.makedirs(cp, exist_ok=True)
    tasks = []
    for case in range(1, 5):
        for i in range(num):
            tasks.append((args, case, i, os.path.join(fp, f'c{case}_{i}'),
                          os.path.join(cp, f'c{case}_{i}'), sams, resume, force))
    if args.workers <= 1:
        for task in tqdm(tasks, desc='3D vascular identities'):
            _generate_identity_task(task)
    else:
        with Pool(processes=args.workers) as pool:
            list(tqdm(pool.imap(_generate_identity_task, tasks), total=len(tasks),
                      desc='3D vascular identities'))
    

if __name__ == '__main__':
    args = parse_args()
    num_case = 4
    if args.num_identities <= 0 or args.num_identities % num_case:
        raise ValueError('--num-identities must be positive and divisible by 4')
    if args.samples_per_identity <= 0:
        raise ValueError('--samples-per-identity must be positive')
    num_percase = args.num_identities // num_case
    num_ids = num_case * num_percase # num of ids
    num_sams = args.samples_per_identity
    output_root = os.path.abspath(args.output_root)

    # path to full palm vein patterns
    full_path = os.path.join(output_root, 'full')
    # path to crop palm vein patterns
    crop_path = os.path.join(output_root, 'crop')
    # path to images of palmprint and palm vein blended together
    pv_path = os.path.join(output_root, 'pv')
    # path to enhanced images
    aug_path = os.path.join(output_root, 'palmvein')

    started = time.time()
    os.makedirs(output_root, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update({'num_case': num_case, 'num_percase': num_percase})
    with open(os.path.join(output_root, 'generation_config.json'), 'w', encoding='utf-8') as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)

    # generate 3d tree
    get_large_pv(args, full_path, crop_path, num_ids, num_sams, resume=args.resume, force=args.force)
    
    # blend palm vein patterns with palmprint modeled by bezier creases
    cross_folder(vein_path=crop_path, palmprint_path=args.bezier_root,
                  pv_path=pv_path, sams=num_sams, num_percase=num_percase, 
                  num_case=num_case, resume=args.resume, force=args.force)
    
    # image enhancement
    aug(pv_path, aug_path, seed=args.seed, resume=args.resume, force=args.force)

    cvt(aug_path)
    elapsed = time.time() - started
    with open(os.path.join(output_root, 'generation_status.json'), 'w', encoding='utf-8') as handle:
        json.dump({'complete': True, 'identities': num_ids, 'samples_per_identity': num_sams,
                   'images': num_ids * num_sams, 'elapsed_seconds': elapsed}, handle,
                  indent=2, sort_keys=True)
   
    
