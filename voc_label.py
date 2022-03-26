import xml.etree.ElementTree as ET
import pickle
import os,shutil
from os import listdir, getcwd
from os.path import join
sets = ['train', 'test','val']
classes = ['fire']

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def convert_annotation(image_id,image_set):
    in_file = open('data/Annotations/%s.xml' % (image_id))
    out_file = open('data/labels/%s/'%(image_set)+'%s.txt' %(image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
wd = getcwd()
print(wd)
if not os.path.exists('data/labels/'):
    os.makedirs('data/labels/')
for image_set in sets:
    if not os.path.exists('data/labels/%s'%(image_set)):
        os.makedirs('data/labels/%s'%(image_set))
    image_ids = open('data/ImageSets/%s.txt' % (image_set)).read().split('\n')
    if not os.path.exists('data/images/%s'%(image_set)):
        os.makedirs('data/images/%s'%(image_set))
    for image_id in image_ids:
        srcfile='data/JPEGImages/%s.jpg'%(image_id)
        dstfile='data/images/%s'%(image_set)
        shutil.copy(srcfile, dstfile)
        convert_annotation(image_id,image_set)