"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict
import random
import msgspec

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class ImageTextPairDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {"image": image, 
                "prompt": "This image has the caption: ",
                "text_input": caption,
                "text_output": caption, 
                "image_id": index}



class ImageTextRoleReplaceSwap(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,
                 aug_ratio=0.3, swap_ratio=0.3, roles='all',prompt_set=None, aug_prompt_set=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        anno_path = ann_paths[0]
        replace_word_path = ann_paths[1]

        with open(anno_path, "rb") as f:
            anno_all = msgspec.json.decode(f.read())

        with open(replace_word_path, "rb") as f:
            token_dict = msgspec.json.decode(f.read())

        if roles == 'nva':
            self.role_types = ['noun', 'verb', 'attr']
        elif roles == 'npa':
            self.role_types = ['noun', 'pred', 'attr']
        elif roles == 'npae':
            self.role_types = ['noun', 'pred', 'attr', 'ent']
        elif roles == 'pvane':
            self.role_types = ['pred', 'verb', 'attr', 'noun', 'ent']
        elif roles == 'ne':
            self.role_types = ['noun', 'ent']
        elif roles == 'ep':
            self.role_types = ['ent', 'pred']
        elif roles == 'nv':
            self.role_types = ['noun', 'verb']
        else:
            self.role_types = list(token_dict.keys())

        self.image_id = []
        for idx in range(len(anno_all)):
            parsed_res = anno_all[idx].get('parse_res')
            if parsed_res is None: continue

            parsed_pos = parsed_res.get('parsed_pos')
            if parsed_pos is None: continue

            valid_ent_cnt = 0
            for r in self.role_types:
                type_pos_min = 1
                if len(parsed_pos[r]) >= type_pos_min:
                    valid_ent_cnt += 1
            if valid_ent_cnt < 1:
                continue

            self.image_id.append(idx)

        print(f'valid img_txt pair {len(anno_all)} -> {len(self.image_id)}')

        self.annotation = anno_all

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.all_roles_dict = {k:[] for k in self.role_types}
        for role_type in self.role_types:
            freq_thres = len(self.annotation) // len(token_dict[role_type].keys())
            for name, cnt in token_dict[role_type].items():
                if cnt > freq_thres:
                    self.all_roles_dict[role_type].append(name)

        self.no_swap_indicators = ['next to', 'near', 'beside', 'adjacent', 'across', 'with', 'and']

        self.aug_ratio = aug_ratio
        self.swap_ratio = swap_ratio

        print(f'aug_ratio {self.aug_ratio} \nswap_ratio {self.swap_ratio}')

        if prompt_set is None:
            prompt_set = ['A short image caption:',
            'A short image description:',
            'A photo of',
            'An image that shows',
            'Write a short description for the image:',
            'Write a description for the photo:',
            'Provide a description of what is presented in the photo:',
            'Briefly describe the content of the image:',
            'Can you briefly explain what you see in the image?',
            'Could you use a few words to describe what you perceive in the photo?',
            'Please provide a short depiction of the picture.',
            'Using language, provide a short account of the image.',
            'Use a few words to illustrate what is happening in the picture.']

        if aug_prompt_set is None:
            aug_prompt_set = ['Check the caption',
            'Check the caption according to the image',
            'Based on the image, please correct the caption',
            'Given the image, please correct the wrong description',
            'Based on the image, refine the description',
            'Use the provided image to correct the caption']

        self.prompt_set = prompt_set
        self.aug_prompt_set = aug_prompt_set

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, index):

        index = self.image_id[index]

        each_anno = self.annotation[index]

        if each_anno["image"][0] == '/':
            image_path = os.path.join(self.vis_root, each_anno["image"].split('/')[-1])
        else:
            image_path = os.path.join(self.vis_root, each_anno["image"])

        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        each_cap = each_anno['caption']

        head = ''
        if 'there is' in each_cap:
            head = "there is "
        elif 'there are' in each_cap:
            head = "there are "

        parsed_res = each_anno['parse_res']
        parsed_pos = parsed_res['parsed_pos']
        parsed_text = parsed_res['parsed_text']

        swapped = False

        role_list = []
        role_idx_list = []
        swappable_type_list = []
        for role_type in self.role_types:
            role_poses = parsed_pos[role_type]
            if len(role_poses) > 0:
                role_idx = random.choice(range(len(role_poses)))
                role_list.append(role_poses[role_idx])
                role_idx_list.append(role_idx)
                if len(role_poses) > 1:
                    swapped = True
                    swappable_type_list.append(role_type)
            else:
                role_list.append(None)
                role_idx_list.append(-1)

        if swapped:
            if random.random() > self.swap_ratio:
                swapped = False
            else:
                for indicator in self.no_swap_indicators:
                    if indicator in each_cap:
                        swapped = False
                        break

        role = None
        if swapped:
            role_type_chosen = random.choice(swappable_type_list)
            role_id = self.role_types.index(role_type_chosen)
            role = role_list[role_id]
        else:
            while role is None:
                role_id = int(len(role_list) * random.random() - 0.1)  # idx is intra-type, id is inter-type: range(len(self.role_types))
                role = role_list[role_id]

        role_type_chosen = self.role_types[role_id]
        role_poses_chosen = parsed_pos[role_type_chosen]
        role_idx_chosen = role_idx_list[role_id]

        do_replace = True
        if random.random() > self.aug_ratio:
            do_replace = False

        start_idx = role[0]
        end_idx = role[-1]

        init_replace_word = ' '.join([parsed_text[each] for each in range(start_idx, end_idx)]).lower()

        if swapped:
            id_left = list(range(len(role_poses_chosen)))
            id_left.remove(role_idx_chosen)
            swap_id = random.choice(id_left)
            swap_role = role_poses_chosen[swap_id]

            replace_word = ' '.join([parsed_text[each] for each in range(swap_role[0], swap_role[1])]).lower()
            init_cap = ' '.join(parsed_text).lower()
            new_caption = init_cap.replace(replace_word, chr(0)).replace(init_replace_word, replace_word).replace(
                chr(0), init_replace_word)
            new_caption = head + new_caption
        else:
            replace_name_selected = random.choice(self.all_roles_dict[role_type_chosen])
            replace_word = replace_name_selected.lower()

            min_idx = 0
            max_idx = len(parsed_text)

            new_token_seq_head = [parsed_text[each] for each in range(min_idx, start_idx)]
            new_token_seq_tail = [parsed_text[each] for each in range(end_idx, max_idx)]

            new_caption = ' '.join(new_token_seq_head) + f" {replace_word} " + ' '.join(new_token_seq_tail)
            new_caption = new_caption.lower().strip(' ')
            new_caption = head + new_caption

        ori_caption = f'{each_cap}'
        hard_caption = f'{new_caption}'

        if do_replace:

            prompt = f'{random.choice(self.aug_prompt_set)}: "{hard_caption}",'
            prompt = self.text_processor(prompt)

            if not swapped:
                inter_word = ['should be', 'could be', 'is', 'actually is']
                inter_word_selected = int(len(inter_word) * random.random() - 0.1)
                answer = f'"{replace_word}" {inter_word[inter_word_selected]} "{init_replace_word}".'
                new_caption = f'{prompt} {answer}'
                new_caption = new_caption.strip(' ')
            else:
                inter_word = ['are swapped', 'need to be switched', 'should exchange their positions', 'need to be swapped']
                inter_word_selected = random.choice(inter_word)
                answer = f'"{replace_word}" and "{init_replace_word}" {inter_word_selected}.'

                new_caption = f'{prompt} {answer}'
                new_caption = new_caption.strip(' ')
        else:
            prompt = random.choice(self.prompt_set)
            new_caption = f'{each_cap}'
            answer = new_caption

        ori_caption = self.text_processor(ori_caption)
        hard_caption = self.text_processor(hard_caption)
        text_output = self.text_processor(answer)

        return {"image": image, "image_id": index, "prompt": prompt,
                "text_input": prompt, "text_output": text_output,
                "caption": ori_caption, "hard_caption": hard_caption,
                "replace": do_replace, "swap": swapped, "role_type": role_type_chosen,
                "init_phrase": init_replace_word, "replace_phrase": replace_word}
