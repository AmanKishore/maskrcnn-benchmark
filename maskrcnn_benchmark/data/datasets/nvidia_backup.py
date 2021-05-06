import os
from PIL import Image
import torch
import cv2
import json
import numpy as np
import sqlite3
import uuid
from PIL import Image
import glob
import shutil
import pandas as pd


from maskrcnn_benchmark.structures.bounding_box import BoxList

def fp16tojpeg(img_path, width=960, height=604):
    data = np.fromfile(img_path, dtype=np.float16, count=-1)
    data = data.reshape(3, height, width)
    data *= 255.
    data[data > 255] = 255.0
    data[data < 0] = 0.0
    data = np.asarray(data, dtype=np.uint8)
    data = data.transpose(1, 2, 0)
    img = Image.fromarray(data, mode="RGB")

    # Save JPG image.
    # img_path = os.path.join('/tmp', os.path.splitext(img_path)[0] + '.jpeg')
    # img.save(img_path)
    return img

def is_valid_uuid(val):
    try:
        uuid.UUID(str(val))
        return True
    except ValueError:
        return False

def v8tov9(dataset):
    # Iterate through the datasets
    v8_sqlite = os.path.join(os.path.dirname(dataset), 'dataset' + '.sqlite')

    # Make a copy of the sqlite and connect to edit.
    shutil.copyfile(src=dataset, dst=v8_sqlite)
    print('Editing the following path: ' + str(dataset))
    conn = sqlite3.connect(dataset)
    cur = conn.cursor()

    try: cur.execute("UPDATE export_info SET schema_version = 9")
    except: pass
    try: cur.execute("UPDATE export_info SET encoding = 'STRUCTURED_JSON'")
    except: pass
    try: cur.execute("ALTER TABLE export_info ADD schema_version_minor TEXT NOT NULL DEFAULT('0')")
    except: pass
    try: cur.execute("ALTER TABLE export_info ADD encoding TEXT NOT NULL DEFAULT('STRUCTURED_JSON')")
    except: pass
    try: cur.execute("ALTER TABLE features ADD label_id TEXT NOT NULL DEFAULT('NULL')")
    except: pass
    try: cur.execute("ALTER TABLE features ADD trackline_id TEXT NOT NULL DEFAULT('NULL')")
    except: pass


    # Get the features dataframe.
    features_df = pd.read_sql_query('SELECT * FROM features', conn)

    # Iterate over the rows and edit the data json.
    for ind, row in features_df.iterrows():
        data = row['data']
        data = json.loads(data)
        
        features_df.at[ind, 'label_id'] = "jackalope:ffffffff:fffffffffffffffffff:fffff:fffff:ffffffff"
        '''
        if 'vertices' not in data:
            features_df.at[ind, 'data'] = json.dumps(data)
            continue
        else:
            data["occlusion"] = "unknown"
            data["truncation"] = "unknown"
            left = data['vertices'][0][0]
            top = data['vertices'][0][1]
            right = data['vertices'][1][0]
            bottom = data['vertices'][1][1]
            data['vertices'] = [[left, top], [right, bottom]]
            features_df.at[ind, 'data'] = json.dumps(data)
        '''

        # coordinates = {"occlusion": "width", "front": 1, "vertices": [[1072, 596], [1236, 698]], "back": 0, "truncation": "full", "attributes": []}
        if 'vertices' not in data:
            datas = {'emptyLabel':True}
            features_df.at[ind, 'label_data_type'] = 'EMPTY'
            features_df.at[ind, 'label_family'] = 'EMPTY'
        else:
            left = data['vertices'][0][0]
            top = data['vertices'][0][1]
            right = data['vertices'][1][0]
            bottom = data['vertices'][1][1]
            data['vertices'] = [{"x":left,"y":top},{"x":right,"y":bottom}]
            datas = {"shape2d":
                {"box2d":
                    {"vertices":[
                        {"x":left,"y":top},{"x":right,"y":bottom}]},
                "attributes":[
                    {"name":"tags","enumsList":{}},
                    {"name":"label_name","enum":row["label_name"]},
                    {"name":"front","numerical":0.0},
                    {"name":"back","numerical":0.0},
                    {"name":"truncated","enum":"UNKNOWN"},
                    {"name":"occluded","enum":"UNKNOWN"}]}}
        datas = json.dumps(datas)
        # datas = datas.replace(" ", "") 

        features_df.at[ind, 'data'] = datas
        features_df.at[ind, 'label_id'] = "jackalope:ffffffff:fffffffffffffffffff:fffff:fffff:ffffffff"
    

    # Replace features table with pandas dataframe.
    # features_df = features_df[['data', 'frame_id', 'id', 'label_class_identifier', 'label_class_namespace', 'label_class_version', 'label_data_type', 'label_family', 'label_id', 'label_name', 'trackline_id']]

    features_df.to_sql('features', con=conn, if_exists='replace')

    # Create the removed indices for the features table.
    try: cur.execute("CREATE UNIQUE INDEX export_info_uniq_idx on export_info(1)")
    except: pass
    try: cur.execute("CREATE INDEX features_frame_id_index ON features(frame_id)")
    except: pass
    try: cur.execute("CREATE INDEX features_label_id_index ON features(label_id)")
    except: pass
    try: cur.execute("CREATE INDEX features_trackline_id_index ON features(trackline_id)")
    except: pass
    try: cur.execute("CREATE INDEX frames_frame_number_index ON frames(frame_number)")
    except: pass
    try: cur.execute("CREATE INDEX frames_sequence_id_index ON frames(sequence_id)")
    except: pass
    try: cur.execute("CREATE INDEX label_class_identifier_index ON features(label_class_identifier)")
    except: pass
    try: cur.execute("CREATE INDEX sequences_camera_name_index ON sequences(camera_name)")
    except: pass
    try: cur.execute("CREATE INDEX sequences_dataset_name_index ON sequences(dataset_name)")
    except: pass
    try: cur.execute("CREATE INDEX sequences_session_uuid_camera_name_index ON sequences(session_uuid, camera_name)")
    except: pass
    try: cur.execute("CREATE INDEX sequences_session_uuid_index ON sequences(session_uuid)")
    except: pass
    # Commit and close the connection.
    conn.commit()
    conn.close()

class NVIDIADataset(object):
    CLASSES = {
        "automobile" : 0,
        "bicycle" : 1,
        "person" : 2,
        "heavy_truck" : 3 
    }
    def __init__(self, image_dir, dataset, use_difficult=False, transforms=None):
        # as you would do normally
        self.image_dir = image_dir
        self.dataset = dataset
        self.images = []
        self.labels = []
        self.bboxs = []
        self.widths = []
        self.heights = []
        self.categories = {0 : "automobile",
                           1 : "bicycle",
                           2 : "person",
                           3 : "heavy_truck"}
        datasetv9 = False

        for img_path in glob.glob(self.image_dir + '/**', recursive=True):
            if os.path.splitext(img_path)[1] in {'.fp16', '.jpeg', 'jpg', '.png'}:
                path = os.path.normpath(img_path)
                path = path.split(os.sep)
                
                try:
                    while not is_valid_uuid(path[0]):
                        path.pop(0)
                    session_uuid = path[0]
                except:
                    raise ValueError("Could not find valid uuid in file path please specify with the argument --session_uuid.")
                frame_number = os.path.splitext(path[-1])[0]


                try: conn = sqlite3.connect(self.dataset)
                except: raise ValueError("Please specify valid path to a v9 sqlite.")
                cursor = conn.cursor()
                query = (
                    "SELECT fe.data, s.width, s.height, fo.width, fo.height "
                    "FROM sequences s INNER JOIN frames fr ON fr.sequence_id = s.id "
                    "INNER JOIN features fe ON fe.frame_id = fr.id LEFT JOIN formats fo "
                    "WHERE s.session_uuid='{0}' and fr.frame_number='{1}'").format(str(session_uuid), str(frame_number))
                cursor.execute(query)
                n = cursor.fetchall()
                if len(n) < 1:
                    self.labels.append([])
                    continue
                
                if not datasetv9:
                    try: 
                        gt = json.loads(n[0][0])
                        # print(gt)
                        # print(gt["shape2d"]["attributes"])
                        attributes = gt["shape2d"]["attributes"] 
                    except Exception as e: 
                        # print(e)
                        # print(gt)
                        # print("HERE")
                        v8tov9(self.dataset)
                        conn = sqlite3.connect(self.dataset)
                        cursor.execute(query)
                        datasetv9 = True
                        n = cursor.fetchall()
                

                width = int(n[0][1] * n[0][3])
                height = int(n[0][2] * n[0][4])
                gt_2d_bbox = []
                gt_labels = []

                # Get the GT data per object
                for gt in n:
                    gt = json.loads(gt[0])
                    attributes = gt["shape2d"]["attributes"]
                    attributes = {attr["name"]:value for attr in attributes for _, value in attr.items()}
                    if len(gt["shape2d"]["box2d"]["vertices"]) == 2:
                        bbox = [list(x.values()) for x in gt["shape2d"]["box2d"]["vertices"]]
                    else:
                        del gt["shape2d"]["box2d"]["vertices"][1]
                        del gt["shape2d"]["box2d"]["vertices"][1]
                        bbox = [list(x.values()) for x in gt["shape2d"]["box2d"]["vertices"]]

                    # Scale the values accordingly
                    bbox = [[x*n[0][3] for x in bbox[0]], [x*n[0][3] for x in bbox[1]]]
                    # Flatten the list
                    bbox = [x for sublist in bbox for x in sublist]
                    gt_2d_bbox.append(bbox)

                    label = attributes["label_name"]
                    gt_labels.append(NVIDIADataset.CLASSES[label])
                
                self.images.append(img_path)
                self.widths.append(width)
                self.heights.append(height)
                self.bboxs.append(gt_2d_bbox)
                self.labels.append(gt_labels)



    def __getitem__(self, idx):
        # load the image as a PIL Image
        img_path = self.images[idx]
        # print(img_path)

        if img_path.endswith(".fp16"):
            image = fp16tojpeg(img_path, self.widths[idx], self.heights[idx])
        else:
            image = Image.open(self.images[idx]).convert("RGB")
        image = np.array(image)

        # print(image.shape)
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        # boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        boxes = self.bboxs[idx]
        # and labels
        # labels = torch.tensor([10, 20])
        labels = torch.tensor(self.labels[idx])

        # if img_path.endswith(".fp16"):
        #     image = fp16tojpeg(img_path, width, height)
        # else:
        #     image = Image.open(self._imgpath % img_id).convert("RGB")
        # img_width, img_height = image.sizer(self.labels[idx])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        # if self.transforms:
        #     image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        image = torch.as_tensor(image)

        return image, boxlist, idx

    def __len__(self):
        return len(self.images)

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        # img_path = self.images[idx]

        # if img_path.endswith(".fp16"):
        #     image = fp16tojpeg(img_path, width, height)
        # else:
        #     image = Image.open(self._imgpath % img_id).convert("RGB")
        # img_width, img_height = image.size
        return {"height": self.heights[idx], "width": self.widths[idx]}

def via2coco(via_json, output_path):

    # load original via data
    # loaded_json = load_annotations(via_json)

    # create header info starting coco file
    # first: 'categories' data
    categories = []
    categories.append(dict(supercategory='automobile', id=1, name='automobile'))
    categories.append(dict(supercategory='bicycle', id=2, name='bicycle'))
    categories.append(dict(supercategory='person', id=3, name='person'))
    categories.append(dict(supercategory='heavy_truck', id=4, name='heavy_truck'))
    categories.append(category)

    # second: 'annotations' data
    annotations = []
    filenames = []
    area = 0
    via_annotations = loaded_json['_via_img_metadata']
    for an_item in via_annotations.values():
        filename = an_item['filename']
        filenames.append(filename)
        image_id = filename.split('_')
        image_id = image_id.pop()
        image_id = int(image_id[:-4])
        # generate keypoint vector
        # keypoint_vector = np.zeros(len(coco_parts.keys())*3)  # x-y-visible, make it json serializable later
        bbox = []  # bounding baby box
        regions = an_item['regions']
        for a_region in regions:
            shape = a_region['shape_attributes']
            region = a_region['region_attributes']
            # if region['id'] in coco_parts.keys():  # recognize mscoco part
            #     index = coco_parts[region['id']]*3
            #     keypoint_vector[index] = shape['cx']
            #     keypoint_vector[index + 1] = shape['cy']
            #     keypoint_vector[index + 2] = int(region['visible']) + 1  # visible in coco is 1 false 2 true
            if region['id'] == 'body':
                area = shape['width'] * shape['height']
                x0 = shape['x']
                w = shape['width']
                y0 = shape['y']
                h = shape['height']

                # save bbox in this weird cocotools notation
                bbox.append(int(x0))
                bbox.append(int(y0))
                bbox.append(int(w))
                bbox.append(int(h))
        # TODO: consolidate this parameters appart from keypoints
        annotation = dict(image_id=image_id, category_id=1, iscrowd=0, num_keypoints=len(keypoint_vector)/3,
                          id=image_id, segmentation=[], area=area, keypoints=keypoint_vector.tolist(), bbox=bbox)
        # TODO: id=image_id OR filename?
        annotations.append(annotation)

    # third: 'info' data
    info = dict(url='https://github.com/harrisonford', contributor='harrisonford', year=2019, description='alpha',
                date_created='2019', version=1.0)

    # fourth: 'images' data
    images = []
    for a_file in filenames:
        subname = a_file.split('_')
        file_id = subname[1]
        file_id = int(file_id[:-4])  # take ".png" out
        # TODO: We add a lot of dummy numbers to image dict (check wid-hei most importantly)
        an_image = dict(date_captured='secret_hehexd', id=file_id, coco_url='no-coco', height=0, width=0, license=0,
                        file_name=a_file, flickr_url='who_uses_flicker?')
        images.append(an_image)

    # fifth: 'licenses' data
    licenses = ['private']

    # put data in final dictionary
    data = dict(categories=categories, annotations=annotations, info=info, images=images, licenses=licenses)

    with open(output_path, 'w') as outfile:
        json.dump(data, outfile)