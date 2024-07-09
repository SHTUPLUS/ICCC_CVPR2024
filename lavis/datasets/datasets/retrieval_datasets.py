"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from easydict import EasyDict as edict
import json
import pandas as pd
import ast
import random


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        visual_key = "image" if "image" in ann else "video"

        return OrderedDict(
            {
                "file": ann[visual_key],
                "caption": ann["caption"],
                visual_key: sample[visual_key],
            }
        )


class __DisplMixinARO:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        visual_key = "image_path"

        return OrderedDict(
            {
                "file": ann[visual_key],
                "true_caption": ann["true_caption"],
                "false_caption": ann["false_caption"],
            }
        )


class RetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instance_id": ann["instance_id"],
        }


class RetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.annotation = self.annotation

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):

        image_path = os.path.join(self.vis_root, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {"image": image, "index": index}


class VideoRetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["video"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        vpath = os.path.join(self.vis_root, ann["video"])

        video = self.vis_processor(vpath)
        caption = self.text_processor(ann["caption"])

        # return image, caption, self.img_ids[ann['image_id']]
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["video"]],
        }


class VideoRetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["video"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        vpath = os.path.join(self.vis_root, ann["video"])
        video = self.vis_processor(vpath)

        return {"video": video, "index": index}


class VG_Relation(BaseDataset, __DisplMixinARO):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.dataset = self.annotation
        image_dir = vis_root

        self.all_relations = list()
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
            self.all_relations.append(item["relation_name"])

        self.image_preprocess = vis_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        relation = self.all_relations[index]
        item = edict({"image_options": [image],
                      "caption_options": [false_caption, true_caption],
                      "type": relation})
        return item


class VG_Attribution(BaseDataset, __DisplMixinARO):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        '''
        image_preprocess: a function that takes in a PIL image and returns a tensor.
        root_dir: Directory for the VG-A dataset.
        download: Whether to download the dataset if it does not exist.
        '''
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.dataset = self.annotation
        image_dir = vis_root

        self.all_attributes = [f"{item['attributes'][0]}_{item['attributes'][1]}" for item in self.dataset]
        for item in self.dataset:
            item["image_path"] = os.path.join(image_dir, item["image_path"])
            # for display
            # item["caption"] = item["true_caption"] + " | " + item["false_caption"]

        self.image_preprocess = vis_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image = Image.open(test_case["image_path"]).convert('RGB')
        # Get the bounding box that contains the relation. This is to remove the irrelevant details in the scene.
        image = image.crop((test_case["bbox_x"], test_case["bbox_y"], test_case["bbox_x"] + test_case["bbox_w"], test_case["bbox_y"] + test_case["bbox_h"]))

        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        # Each test case has a correct and incorrect caption.
        true_caption = test_case["true_caption"]
        false_caption = test_case["false_caption"]
        attributes = self.all_attributes[index]
        item = edict({"image_options": [image],
                      "caption_options": [false_caption, true_caption],
                      "type": attributes})
        return item


class Winoground(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root=None, ann_paths=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # from datasets import load_dataset
        # self.dataset = load_dataset('facebook/winoground', use_auth_token='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')['test']
        self.data_dir = '../winoground/data'
        path = os.path.join(self.data_dir, 'examples.jsonl')
        with open(path, 'r') as f:
            self.dataset = [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        sample = self.dataset[i]
        idx = sample['id']
        image_0 = sample['image_0']
        image_1 = sample['image_1']
        caption_0 = sample['caption_0']
        caption_1 = sample['caption_1']

        # import ipdb; ipdb.set_trace()
        image_0 = Image.open(os.path.join(self.data_dir, 'images', image_0+'.png')).convert('RGB')
        image_1 = Image.open(os.path.join(self.data_dir, 'images', image_1+'.png')).convert('RGB')
        image_0 = self.vis_processor(image_0)
        image_1 = self.vis_processor(image_1)
        caption_0 = self.text_processor(caption_0)
        caption_1 = self.text_processor(caption_1)

        item = edict({"image_options": [image_0, image_1],
                      "caption_options": [caption_0, caption_1],
                      "id": idx})

        return item


class NegCLIPDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 neg_file_path):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        backup_neg_path = "data/neg_coco/train_neg_clip.tsv"

        df = pd.read_csv(neg_file_path, sep="\t", converters={"neg_caption":ast.literal_eval, "neg_image":ast.literal_eval, "neg_constituency":ast.literal_eval})
        backup_df = pd.read_csv(backup_neg_path, sep="\t", converters={"neg_caption":ast.literal_eval, "neg_image":ast.literal_eval})

        self.images = df["filepath"].tolist()
        for i in range(len(self.images)):
            image_path_split = self.images[i].split("/")
            self.images[i] = os.path.join(vis_root, image_path_split[1], image_path_split[2])

        self.captions = df["title"].tolist() if "constituency" not in df else df["constituency"].tolist()
        self.hard_captions = df["neg_caption"].tolist() if "neg_constituency" not in df else df["neg_constituency"].tolist()
        self.hard_images = df["neg_image"].tolist()
        self.hard_captions_backup = backup_df["neg_caption"].tolist()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        while not self.hard_captions[idx]: idx = random.choice(range(len(self.captions))) # FIXME: change the approximate way

        image = self.vis_processor(Image.open(str(self.images[idx])).convert("RGB"))
        text = self.text_processor(str(self.captions[idx]))

        chose_image_index = random.choice(self.hard_images[idx])
        chosen_new_image = self.images[chose_image_index]
        chosen_new_text = self.captions[chose_image_index]

        chosen_caption = random.choice(self.hard_captions[idx]) \
            if self.hard_captions[idx] else random.choice(self.hard_captions_backup[idx])
        hard_caption = self.text_processor(str(chosen_caption))

        new_image = self.vis_processor(Image.open(str(chosen_new_image)).convert("RGB"))
        new_text = self.text_processor(str(chosen_new_text))

        chosen_caption = random.choice(self.hard_captions[chose_image_index]) \
            if self.hard_captions[chose_image_index] else random.choice(self.hard_captions_backup[chose_image_index])
        new_hard = self.text_processor(str(chosen_caption))

        return {
            "image_id": self.images[idx],
            "instance_id": idx,
            "image": image,
            "text_input": text,
            "new_image": new_image,
            "new_text": new_text,
            "hard_caption": hard_caption,
            "new_hard": new_hard
        }


class NegCLIPEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 neg_file_path):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        df = pd.read_csv(neg_file_path, sep="\t", converters={"neg_caption":ast.literal_eval, "neg_image":ast.literal_eval, "neg_constituency":ast.literal_eval})

        self.images = df["filepath"].tolist() # raw image list
        for i in range(len(self.images)):
            image_path_split = self.images[i].split("/")
            self.images[i] = os.path.join(vis_root, image_path_split[1], image_path_split[2])

        self.captions = df["title"].tolist() if "constituency" not in df else df["constituency"].tolist()
        self.hard_captions = df["neg_caption"].tolist() if "neg_constituency" not in df else df["neg_constituency"].tolist()
        self.hard_images = df["neg_image"].tolist()

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for idx, img_path in enumerate(self.images):
            caption = self.captions[idx]
            if img_path not in self.image:
                self.image.append(img_path) # rearranged images
                self.text.append(self.text_processor(caption))
                img_id = len(self.image) - 1
                self.img2txt[img_id] = [txt_id]
                self.txt2img[txt_id] = img_id
                txt_id += 1
                for neg_caption in self.hard_captions[idx]:
                    self.text.append(self.text_processor(neg_caption))
                    txt_id += 1
            else:
                img_id = self.image.index(img_path)
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                for neg_caption in self.hard_captions[idx]:
                    self.text.append(self.text_processor(neg_caption))
                    txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.vis_processor(Image.open(str(self.image[idx])).convert("RGB"))

        return {"image": image, "index": idx}


class CREPEDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 neg_file_path):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        df = pd.read_csv(neg_file_path, converters={"image_id":lambda x:str(x)+'.jpg', "n":str, "hard_negs":ast.literal_eval})

        self.captions = df['caption'].to_list()
        self.images = df['image_id'].to_list()
        self.vis_root = vis_root
        self.hard_captions = df['hard_negs'].to_list()
        self.complexities = df['n'].to_list()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path = os.path.join(self.vis_root, self.images[idx])
        image = self.vis_processor(Image.open(image_path).convert('RGB'))

        # Each test case has a correct and several wrong captions.
        true_caption = self.captions[idx]
        false_caption_list = self.hard_captions[idx]
        complexity = self.complexities[idx]
        item = edict({"image_options": [image],
                      "caption_options": [true_caption] + false_caption_list,
                      "type": complexity})
        return item