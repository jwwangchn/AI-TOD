import mmcv
import os
import json
import numpy as np
import cv2
import csv
import shutil
import inspect
from tqdm import tqdm
from PIL import Image
from skimage.io import imread

from wwtool.datasets import Convert2COCO
import wwtool



Image.MAX_IMAGE_PIXELS = int(2048 * 2048 * 2048 // 4 // 3)

class XVIEW2COCO(Convert2COCO):
    def __generate_coco_annotation__(self, annotpath, imgpath):
        """
        docstring here
            :param self: 
            :param annotpath: the path of each annotation
            :param return: dict()  
        """
        objects = self.__xview_parse__(annotpath, imgpath)
        
        coco_annotations = []

        if generate_small_dataset and len(objects) > 0:
            wwtool.generate_same_dataset(imgpath, 
                                        annotpath,
                                        dst_image_path,
                                        dst_label_path,
                                        src_img_format='.png',
                                        src_anno_format='.txt',
                                        dst_img_format='.png',
                                        dst_anno_format='.txt',
                                        parse_fun=wwtool.simpletxt_parse,
                                        dump_fun=wwtool.simpletxt_dump,
                                        save_image=True)

        for object_struct in objects:
            bbox = object_struct['bbox']
            segmentation = object_struct['segmentation']
            label = object_struct['label']

            width = bbox[2]
            height = bbox[3]
            area = height * width

            if area <= self.small_object_area and self.groundtruth:
                self.small_object_idx += 1
                continue

            coco_annotation = {}
            coco_annotation['bbox'] = bbox
            coco_annotation['segmentation'] = [segmentation]
            coco_annotation['category_id'] = label
            coco_annotation['area'] = np.float(area)

            coco_annotations.append(coco_annotation)
            
        return coco_annotations
    
    def __xview_parse__(self, label_file, image_file):
        """
        (xmin, ymin, xmax, ymax)
        """
        with open(label_file, 'r') as f:
            lines = f.readlines()
    
        objects = []
        
        total_object_num = len(lines)
        small_object_num = 0
        large_object_num = 0
        total_object_num = 0

        basic_label_str = " "
        for line in lines:
            object_struct = {}
            line = line.rstrip().split(' ')
            label = basic_label_str.join(line[4:])
            bbox = [float(_) for _ in line[0:4]]

            xmin, ymin, xmax, ymax = bbox
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            if bbox_w * bbox_h <= self.small_object_area:
                continue

            total_object_num += 1
            if bbox_h * bbox_w <= small_size:
                small_object_num += 1
            if bbox_h * bbox_w >= large_object_size:
                large_object_num += 1

            object_struct['bbox'] = [xmin, ymin, bbox_w, bbox_h]
            object_struct['segmentation'] = wwtool.bbox2pointobb([xmin, ymin, xmax, ymax])
            object_struct['label'] = original_class[label]
            
            objects.append(object_struct)
        
        if total_object_num > self.max_object_num_per_image:
            self.max_object_num_per_image = total_object_num

        if just_keep_small or generate_small_dataset:
            if small_object_num >= total_object_num * small_object_rate and large_object_num < 1:
                return objects
            else:
                return []
        else:
            return objects



def coco_merge(
    input_extend: str, input_add: str, output_file: str, 
) -> str:
    """Merge COCO annotation files.

    Args:
        input_extend: Path to input file to be extended.
        input_add: Path to input file to be added.
        output_file : Path to output file with merged annotations.
        indent: Argument passed to `json.dump`. See https://docs.python.org/3/library/json.html#json.dump.
    """
    with open(input_extend, "r") as f:
        data_extend = json.load(f)
    with open(input_add, "r") as f:
        data_add = json.load(f)

    output: Dict[str, Any] = {
        k: data_extend[k] for k in data_extend if k not in ("images", "annotations")
    }

    output["images"], output["annotations"] = [], []

    for i, data in enumerate([data_extend, data_add]):

        cat_id_map = {}
        for new_cat in data["categories"]:
            new_id = None
            for output_cat in output["categories"]:
                if new_cat["name"] == output_cat["name"]:
                    new_id = output_cat["id"]
                    break

            if new_id is not None:
                cat_id_map[new_cat["id"]] = new_id
            else:
                new_cat_id = max(c["id"] for c in output["categories"]) + 1
                cat_id_map[new_cat["id"]] = new_cat_id
                new_cat["id"] = new_cat_id
                output["categories"].append(new_cat)

        img_id_map = {}
        for image in data["images"]:
            n_imgs = len(output["images"])
            img_id_map[image["id"]] = n_imgs
            image["id"] = n_imgs

            output["images"].append(image)

        for annotation in data["annotations"]:
            n_anns = len(output["annotations"])
            annotation["id"] = n_anns
            annotation["image_id"] = img_id_map[annotation["image_id"]]
            annotation["category_id"] = cat_id_map[annotation["category_id"]]

            output["annotations"].append(annotation)


    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    #########################################
    # split xview, get the annotations and labels of the split xview
    
    xview_class_labels_file = 'aitod_xview/xview_class_labels.txt'
    json_file = 'xview/xView_train.geojson'
    xview_parse = wwtool.XVIEW_PARSE(json_file, xview_class_labels_file)

    image_format1 = '.tif'

    subimage_size = 800
    gap = 200


    image_path = 'xview/ori/train_images'

    image_save_path = 'xview/split/images'
    wwtool.mkdir_or_exist(image_save_path)
    label_save_path = 'xview/split/labels'
    wwtool.mkdir_or_exist(label_save_path)

    for idx, image_name in enumerate(os.listdir(image_path)):
        print(idx, image_name)
        file_name = image_name.split(image_format1)[0]
        image_file = os.path.join(image_path, file_name + image_format1)
        
        img = imread(image_file)

        objects = xview_parse.xview_parse(image_name)
        bboxes = np.array([wwtool.xyxy2cxcywh(obj['bbox']) for obj in objects])
        labels = np.array([obj['label'] for obj in objects])

        subimages = wwtool.split_image(img, subsize=subimage_size, gap=gap)
        subimage_coordinates = list(subimages.keys())
        bboxes_ = bboxes.copy()
        labels_ = labels.copy()

        if bboxes_.shape[0] == 0:
            continue
        
        for subimage_coordinate in subimage_coordinates:
            objects = []
            
            bboxes_[:, 0] = bboxes[:, 0] - subimage_coordinate[0]
            bboxes_[:, 1] = bboxes[:, 1] - subimage_coordinate[1]
            cx_bool = np.logical_and(bboxes_[:, 0] >= 0, bboxes_[:, 0] < subimage_size)
            cy_bool = np.logical_and(bboxes_[:, 1] >= 0, bboxes_[:, 1] < subimage_size)
            subimage_bboxes = bboxes_[np.logical_and(cx_bool, cy_bool)]
            subimage_labels = labels_[np.logical_and(cx_bool, cy_bool)]
            
            if len(subimage_bboxes) == 0:
                continue
            img = subimages[subimage_coordinate]
            if np.mean(img) == 0:
                continue

            label_save_file = os.path.join(label_save_path, '{}__{}_{}.txt'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]))
            image_save_file = os.path.join(image_save_path, '{}__{}_{}.png'.format(file_name, subimage_coordinate[0], subimage_coordinate[1]))
            cv2.imwrite(image_save_file, img)
            
            for subimage_bbox, subimage_label in zip(subimage_bboxes, subimage_labels):
                subimage_objects = dict()
                subimage_objects['bbox'] = wwtool.cxcywh2xyxy(subimage_bbox.tolist())
                subimage_objects['label'] = subimage_label
                objects.append(subimage_objects)
            wwtool.simpletxt_dump(objects, label_save_file)
    
    
    ########################
    # filter out irrelevant classes in xView, get the filtered_xview
    convert_classes = {}

    with open('aitod_xview/converted_class.txt') as f:
        for row in csv.reader(f):
            if row[0].split(":")[1] == 'None':
                converted_class = None
            else:
                converted_class = row[0].split(":")[1]
            convert_classes[row[0].split(":")[0]] = converted_class
    
    image_format2 = '.png'

    origin_image_path = 'xview/split/images'
    origin_label_path = 'xview/split/labels'

    filtered_image_path = 'xview/filtered/images'
    filtered_label_path = 'xview/filtered/labels'

    wwtool.mkdir_or_exist(filtered_image_path)
    wwtool.mkdir_or_exist(filtered_label_path)

    filter_count = 1
    progress_bar = mmcv.ProgressBar(len(os.listdir(origin_label_path)))
    for label_name in os.listdir(origin_label_path):
        image_objects = wwtool.simpletxt_parse(os.path.join(origin_label_path, label_name))
        filtered_objects = []
        for image_object in image_objects:
            if convert_classes[image_object['label']] == None:
                filter_count += 1
                continue
            else:
                image_object['label'] = convert_classes[image_object['label']]
                filtered_objects.append(image_object)

        if len(filtered_objects) > 0:
            img = cv2.imread(os.path.join(origin_image_path, os.path.splitext(label_name)[0] + image_format2))
            save_image_file = os.path.join(filtered_image_path, os.path.splitext(label_name)[0] + '.png')
            # print("Save image file: ", save_image_file)
            cv2.imwrite(save_image_file, img)
            wwtool.simpletxt_dump(filtered_objects, os.path.join(filtered_label_path, os.path.splitext(label_name)[0] + '.txt'))
        
        progress_bar.update()

    print("Filter object counter: {}".format(filter_count))
    
    #########################
    # select the xview images included in ai-tod, then merge these xview images into json. 
    sets = ['val','train','trainval','test']

    path = inspect.getfile(inspect.currentframe())
    abspath = os.path.abspath(path) # get the abs path of current file
    pre_abspath = abspath.split('generate_aitod.py')

    for set in sets:
        dst_image_path = 'xview/xview_aitod_sets/{}/images'.format(set)
        dst_label_path = 'xview/xview_aitod_sets/{}/labels'.format(set)
        wwtool.mkdir_or_exist(dst_image_path)
        wwtool.mkdir_or_exist(dst_label_path)
        abs_dst_image_path = os.path.join(pre_abspath[0], dst_image_path)
        abs_dst_label_path = os.path.join(pre_abspath[0], dst_label_path)
        
        
        xview_aitod_path = 'aitod_xview/aitod_xview_{}.txt'.format(set)
        xview_aitod_path = open(xview_aitod_path, 'r')
        xview_aitod = xview_aitod_path.read()

        xview_aitod = xview_aitod.replace('[\'','')
        xview_aitod = xview_aitod.replace('\']','')
        xview_aitod = xview_aitod.split('\', \'')
        print(len(xview_aitod))
        for item in tqdm(xview_aitod):
            src_img_path = os.path.join('xview/filtered/images', item)
            src_label_path = os.path.join('xview/filtered/labels', item.replace('.png','.txt'))
            abs_src_img_path = os.path.join(pre_abspath[0], src_img_path)
            abs_src_label_path = os.path.join(pre_abspath[0], src_label_path)
            final_dst_image_path = os.path.join(abs_dst_image_path, item)
            final_dst_label_path = os.path.join(abs_dst_label_path, item.replace('.png','.txt'))
            
            shutil.copy(abs_src_img_path, final_dst_image_path)
            shutil.copy(abs_src_label_path, final_dst_label_path)
                  

    
    #########################
    # basic dataset information
    info = {"year" : 2019,
                "version" : "1.0",
                "description" : "XVIEW-COCO",
                "contributor" : "Jinwang Wang",
                "url" : "jwwangchn.cn",
                "date_created" : "2019"
            }
    
    licenses = [{"id": 1,
                    "name": "Attribution-NonCommercial",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                }]

    # dataset's information
    image_format3='.png'
    anno_format='.txt'

    original_class = { 'airplane':                     1, 
                        'bridge':                       2,
                        'storage-tank':                 3, 
                        'ship':                         4, 
                        'swimming-pool':                5, 
                        'vehicle':                      6, 
                        'person':                       7, 
                        'wind-mill':                    8}

    converted_class = [{'supercategory': 'none', 'id': 1,  'name': 'airplane',                 },
                        {'supercategory': 'none', 'id': 2,  'name': 'bridge',                   },
                        {'supercategory': 'none', 'id': 3,  'name': 'storage-tank',             },
                        {'supercategory': 'none', 'id': 4,  'name': 'ship',                     },
                        {'supercategory': 'none', 'id': 5,  'name': 'swimming-pool',            },
                        {'supercategory': 'none', 'id': 6,  'name': 'vehicle',                  },
                        {'supercategory': 'none', 'id': 7,  'name': 'person',                  },
                        {'supercategory': 'none', 'id': 8,  'name': 'wind-mill',               }]

    core_dataset_name = 'xview'
    #imagesets = ['train_filtered']
    release_version = 'v1'
    #rate = '1.0'
    groundtruth = True
    keypoint = False
    
    just_keep_small = False
    generate_small_dataset = False
    small_size = 16 * 16
    small_object_rate = 0.5
    large_object_size = 64 * 64

    for set in sets:

        anno_name = [core_dataset_name, set]
        
        anno_name.append('small')



        if keypoint:
            for idx in range(len(converted_class)):
                converted_class[idx]["keypoints"] = ['top', 'right', 'bottom', 'left']
                converted_class[idx]["skeleton"] = [[1,2], [2,3], [3,4], [4,1]]
            anno_name.append('keypoint')
        
        if groundtruth == False:
            anno_name.append('no_ground_truth')


        imgpath = 'xview/xview_aitod_sets/{}/images'.format(set)
        annopath = 'xview/xview_aitod_sets/{}/labels'.format(set)
        save_path = 'xview/annotations'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        xview = XVIEW2COCO(imgpath=imgpath,
                        annopath=annopath,
                        image_format=image_format3,
                        anno_format=anno_format,
                        data_categories=converted_class,
                        data_info=info,
                        data_licenses=licenses,
                        data_type="instances",
                        groundtruth=groundtruth,
                        small_object_area=0)

        images, annotations = xview.get_image_annotation_pairs()

        json_data = {"info" : xview.info,
                    "images" : images,
                    "licenses" : xview.licenses,
                    "type" : xview.type,
                    "annotations" : annotations,
                    "categories" : xview.categories}

        #anno_name.insert(1, 'train')
        with open(os.path.join(save_path, "_".join(anno_name) + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
    
    sets = ['val','train','trainval','test']

    path = inspect.getfile(inspect.currentframe())
    abspath = os.path.abspath(path) # get the abs path of current file
    pre_abspath = abspath.split('generate_aitod.py')

    ###############################
    # merge the result of aitod_wo_xview and xview
    for set in sets:

        src_ann_file = 'aitod/annotations/aitod_wo_xview_{}.json'.format(set)
        ext_ann_file = 'xview/annotations/xview_{}_small.json'.format(set)
        dst_ann_file = 'aitod/annotations/aitod_{}.json'.format(set)

        coco_merge(src_ann_file, ext_ann_file, dst_ann_file)


    # move xview-aitod files into 
    for set in sets:

        xview_aitoid = os.listdir('xview/xview_aitod_sets/{}/images'.format(set))
        abs_src_dir = os.path.join(pre_abspath[0], 'xview/xview_aitod_sets/{}/images'.format(set))
        abs_dst_dir = os.path.join(pre_abspath[0], 'aitod/images/{}'.format(set))
        for item in xview_aitoid:
            abs_src_img = os.path.join(abs_src_dir, item)
            abs_dst_img = os.path.join(abs_dst_dir, item)
            shutil.copy(abs_src_img, abs_dst_img)

    # delete irrelevant temp files 
    # to do
