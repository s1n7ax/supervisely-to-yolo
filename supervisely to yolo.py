import json
import os
from os import listdir
from os import path
from shutil import copyfile



# current directory
__location__ = path.realpath(
    path.join(os.getcwd(), path.dirname(__file__)))

#supervisely
SUP_DATASET_NAME = 'website_dataset_1'
SUP_META = path.join(__location__, 'meta.json')
SUP_ANN_DIR = path.join(__location__, SUP_DATASET_NAME, 'ann')
SUP_IMG_DIR = path.join(__location__, SUP_DATASET_NAME, 'img')

# yolo
YOLO_DATA_DIR_NAME = 'training'
YOLO_DATA_DIR = path.join(__location__, YOLO_DATA_DIR_NAME)
YOLO_DATA_OBJ_DIR = path.join(YOLO_DATA_DIR, 'obj')
YOLO_NAMES = path.join(YOLO_DATA_DIR, 'obj.names')
YOLO_DATA  = path.join(YOLO_DATA_DIR, 'obj.data')
YOLO_TRAIN  = path.join(YOLO_DATA_DIR, 'obj.train.txt')

'''
Global variables
'''
classes = None




def get_classes():
    with open(SUP_META) as meta:
        meta_json = json.load(meta)
        classes = [item['title'] for item in meta_json['classes']]
        return classes

def create_output_dir():
    if not path.exists(YOLO_DATA_DIR):
        os.makedirs(YOLO_DATA_DIR)

    if not path.exists(YOLO_DATA_OBJ_DIR):
        os.makedirs(YOLO_DATA_OBJ_DIR)

def write_names_file(classes):
    create_output_dir()

    with open(YOLO_NAMES, 'w') as names:
        for cls in classes:
            names.write(cls + '\n')

def get_yolo_obj_rec(size, class_name, points):
    class_index = classes.index(class_name)

    img_width = size['width']
    img_height = size['height']

    min_x, min_y = points[0]
    max_x, max_y = points[1]

    width = max_x - min_x
    height = max_y - min_y

    x = min_x + (width / 2)
    y = min_y + (height / 2)

    yolo_x = x / img_width
    yolo_y = y / img_height
    yolo_width = width / img_width
    yolo_height = height / img_height

    return '{cls} {x} {y} {width} {height}'.format(
        cls = class_index,
        x = yolo_x, 
        y = yolo_y, 
        width = yolo_width,
        height = yolo_height
    )

def add_training_image(img_name):
    with open(YOLO_TRAIN, 'a') as file:
        file.write(YOLO_DATA_DIR_NAME + '/obj/' + img_name + '\n')

def generate_files():
    index = 1
    for fname in listdir(SUP_ANN_DIR):
        filepath = path.join(SUP_ANN_DIR, fname)
        img_file_name, img_file_ext = path.splitext(path.splitext(filepath)[0])
        img_file_name = path.basename(img_file_name)

        with open(filepath) as file:
            json_obj = json.load(file)
            objects = json_obj['objects']
            size = json_obj['size']

            yolo_object_record = ''

            if len(objects) < 1:
                print('no objects in ' + fname)
                continue

            for obj in objects:
                class_name = obj['classTitle']
                if class_name not in classes:
                    continue
                points = obj['points']['exterior']
                yolo_object_record += get_yolo_obj_rec(size, class_name, points) + '\n'
        

        # ignore if there are no matching objects
        if yolo_object_record == '':
            continue

        source_img_file_path = path.join(SUP_IMG_DIR, img_file_name + img_file_ext) 
        new_img_data_file_path = path.join(YOLO_DATA_OBJ_DIR, 'img_' + str(index) + '.txt') 
        new_img_file_path = path.join(YOLO_DATA_OBJ_DIR, 'img_' + str(index) + img_file_ext) 
        
        print(source_img_file_path)
        print(new_img_file_path)
        print(new_img_data_file_path)

        with open(new_img_data_file_path, 'w') as file:
            file.write(yolo_object_record)

        copyfile(source_img_file_path, new_img_file_path)
        add_training_image('img_' + str(index) + img_file_ext)

        index += 1

    with open(YOLO_DATA, 'w') as file:
        content = 'classes = {}\n'.format(len(classes))
        content += 'train  = {}/obj.train.txt\n'.format(YOLO_DATA_DIR_NAME)
        content += 'valid  = {}/obj.test.txt\n'.format(YOLO_DATA_DIR_NAME)
        content += 'names = {}/obj.names\n'.format(YOLO_DATA_DIR_NAME)
        content += 'backup = backup/'
        file.write(content)



classes = get_classes()
#classes = ['text', 'button', 'link', 'field']
write_names_file(classes)
generate_files()