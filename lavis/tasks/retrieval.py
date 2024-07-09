"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import os

import numpy as np
import torch
from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask

from pathlib import Path
import pandas as pd


logger = logging.getLogger(__name__)

@registry.register_task("retrieval")
class RetrievalTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls(cfg=run_cfg)

    def evaluation(self, model, data_loader, **kwargs):
        # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
        score_i2t, score_t2i = model.compute_sim_matrix(data_loader, task_cfg=self.cfg)

        if is_main_process():
            eval_result = self._report_metrics(
                score_i2t,
                score_t2i,
                data_loader.dataset.txt2img,
                data_loader.dataset.img2txt,
            )
            logger.info(eval_result)
        else:
            eval_result = None

        return eval_result

    def after_evaluation(self, val_result, **kwargs):
        return val_result

    @staticmethod
    @torch.no_grad()
    def _report_metrics(scores_i2t, scores_t2i, txt2img, img2txt):

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = -np.ones(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            if index not in txt2img: continue  # text may not have gt image
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        ranks = ranks[ranks!=-1]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(eval_result) + "\n")
        return eval_result

@registry.register_task("retrieval_vgrel")
class VGRelTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        # run_cfg = cfg.run_cfg

        return cls(cfg=cfg)

    def evaluation(self, model, data_loader, **kwargs):
        # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
        scores, all_keys = model.compute_grouped_simmat(data_loader, k_test=self.cfg.run_cfg.k_test)

        if is_main_process():
            eval_result = self._report_metrics(
                scores, all_keys
            )
            logger.info(eval_result)
        else:
            eval_result = None

        return eval_result

    def after_evaluation(self, val_result, **kwargs):
        for record in val_result:
            record.update({"Model": self.cfg.model_cfg.arch})

        output_file = os.path.join(registry.get_path("output_dir"),
                                   "VG_Relation.csv")
        df = pd.DataFrame(val_result)
        print(f"Saving results to {output_file}")
        if os.path.exists(output_file):
            all_df = pd.read_csv(output_file, index_col=0)
            all_df = pd.concat([all_df, df])
            all_df.to_csv(output_file)
        else:
            df.to_csv(output_file)
        return val_result

    @torch.no_grad()
    def _report_metrics(self, scores, all_keys):
        """
        Scores: N x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
        else:
            scores_i2t = scores

        # import ipdb; ipdb.set_trace()
        metrics = {"Accuracy": None}
        preds = np.argmax(scores_i2t, axis=-1) # no image dim included because always i2t
        correct_mask = (preds == 1)
        metrics["Accuracy"] = np.mean(correct_mask)

        all_relations = np.array(all_keys)

        result_records = []
        result_records.append({
            "Relation": "ALL",
            "Accuracy": correct_mask.mean(),
            "Count": len(correct_mask),
            "Dataset": "Visual Genome Relation"
        })

        # Log the accuracy of all relations
        acc_all = []
        for relation in np.unique(all_relations):
            relation_mask = (all_relations == relation)
            if relation_mask.sum() == 0:
                continue
            acc_type = correct_mask[relation_mask].mean()
            acc_all.append(acc_type)
            result_records.append({
                "Relation": relation,
                "Accuracy": acc_type,
                "Count": relation_mask.sum(),
                "Dataset": "Visual Genome Relation"
            })

        result_records.append({
            "Relation": "MEAN",
            "Accuracy": np.mean(acc_all),
            "Count": len(acc_all),
            "Dataset": "Visual Genome Relation"
        })

        return result_records


@registry.register_task("retrieval_vgattr")
class VGAttrTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        # run_cfg = cfg.run_cfg

        return cls(cfg=cfg)

    def evaluation(self, model, data_loader, **kwargs):
        # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
        scores, all_keys = model.compute_grouped_simmat(data_loader, k_test=self.cfg.run_cfg.k_test)

        if is_main_process():
            eval_result = self._report_metrics(
                scores, all_keys
            )
            logger.info(eval_result)
        else:
            eval_result = None

        return eval_result

    def after_evaluation(self, val_result, **kwargs):
        for record in val_result:
            record.update({"Model": self.cfg.model_cfg.arch})

        output_file = os.path.join(registry.get_path("output_dir"),
                                   "VG_Attribution.csv")
        df = pd.DataFrame(val_result)

        if os.path.exists(output_file):
            all_df = pd.read_csv(output_file, index_col=0)
            all_df = pd.concat([all_df, df])
            all_df.to_csv(output_file)
        else:
            df.to_csv(output_file)
        return val_result

    @torch.no_grad()
    def _report_metrics(self, scores, all_keys):
        """
        Scores: N x 2, i.e. first caption is the perturbed one, second is the positive one
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
        else:
            scores_i2t = scores

        preds = np.argmax(scores_i2t, axis=-1)
        correct_mask = (preds == 1)
        result_records = []
        result_records.append({
            "Attributes": "ALL",
            "Accuracy": correct_mask.mean(),
            "Count": len(correct_mask),
            "Dataset": "Visual Genome Attribution"
        })

        acc_all = []
        all_attributes = np.array(all_keys)
        for attr in np.unique(all_attributes):
            attr_mask = (all_attributes == attr)
            if attr_mask.sum() < 25:
                continue
            acc_type = correct_mask[attr_mask].mean()
            acc_all.append(acc_type)
            result_records.append({
                "Attributes": attr,
                "Accuracy": acc_type,
                "Count": attr_mask.sum(),
                "Dataset": "Visual Genome Attribution"
            })

        result_records.append({
            "Attributes": "MEAN",
            "Accuracy": np.mean(acc_all),
            "Count": len(acc_all),
            "Dataset": "Visual Genome Attribution"
        })
        return result_records


@registry.register_task("retrieval_winoground")
class WinogroundTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        # run_cfg = cfg.run_cfg

        return cls(cfg=cfg)

    def evaluation(self, model, data_loader, **kwargs):
        # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
        scores, _ = model.compute_grouped_simmat(data_loader, k_test=self.cfg.run_cfg.k_test)

        new_tag_path = "../winoground/new_tag_assignments.json"
        with open(new_tag_path, 'r') as f:
            new_tags = json.load(f)
        all_keys = list(new_tags.values())

        data_path = '../winoground/data/examples.jsonl'
        with open(data_path, 'r') as f:
            tags = [json.loads(line.strip())['tag']+', '+json.loads(line.strip())['collapsed_tag'] for line in f]

        if is_main_process():
            eval_result = self._report_metrics(
                scores, all_keys, tags
            )
            logger.info(eval_result)
        else:
            eval_result = None

        return eval_result

    def after_evaluation(self, val_result, **kwargs):
        output_file = os.path.join(registry.get_path("output_dir"),
                                   "result.json")
        with open(output_file, 'w') as f:
            json.dump(val_result, f, indent=4)

        output_file = os.path.join(registry.get_path("output_dir"),
                                   "result.csv")
        df = pd.DataFrame(val_result)
        df.to_csv(output_file)

        return val_result

    @torch.no_grad()
    def _report_metrics(self, scores, all_keys, tags):
        total = 0
        i_correct = 0
        t_correct = 0
        all_correct = 0

        ans_list = []
        tagged_stat = {}
        result_records = []
        for i in range(0, len(scores), 2):
            score = scores[i:i+2, :]

            t2i_pred_0 = score[0][0] > score[1][0]
            t2i_pred_1 = score[1][1] > score[0][1]

            i2t_pred_0 = score[0][0] > score[0][1]
            i2t_pred_1 = score[1][1] > score[1][0]

            ans_list.append({"id": i//2,
                             "t2i_pred": [t2i_pred_0.item(), t2i_pred_1.item()],
                             "i2t_pred": [i2t_pred_0.item(), i2t_pred_1.item()],
                             "scores": [[score[0][0].item(), score[0][1].item()],
                                        [score[1][0].item(), score[1][1].item()]]
                             })

            i_pred = [t2i_pred_0, t2i_pred_1]
            t_pred = [i2t_pred_0, i2t_pred_1]
            # t_res = t_pred == ['0','1'] or t_pred == [0,1]
            # i_res = i_pred == ['0','1'] or i_pred == [0,1]
            i_res = i_pred == [True, True]
            t_res = t_pred == [True, True]
            t_correct += t_res
            i_correct += i_res
            all_correct += t_res and i_res
            total += 1

            tag_list = tags[i//2].split(', ')
            new_tag_list = all_keys[i//2]
            if len(new_tag_list) == 0:
                new_tag_list.append('Vanilla')
            for new_tag in tag_list + new_tag_list:
                if new_tag not in tagged_stat:
                    tagged_stat[new_tag] = {'t': t_res, 'i': i_res, 'g':t_res and i_res, 'all': 1}
                else:
                    tagged_stat[new_tag]['t'] += t_res
                    tagged_stat[new_tag]['i'] += i_res
                    tagged_stat[new_tag]['g'] += t_res and i_res
                    tagged_stat[new_tag]['all'] += 1

        result_records.append({
            'Type': 'ALL',
            'Text': t_correct/total*100,
            'Image': i_correct/total*100,
            'Group': all_correct/total*100,
            'Count': total
        })

        for tag in tagged_stat:
            stat = tagged_stat[tag]
            result_records.append({
                'Type': tag,
                'Text': stat['t'] / stat['all'] * 100,
                'Image': stat['i'] / stat['all'] * 100,
                'Group': stat['g'] / stat['all'] * 100,
                'Count': stat['all']
            })

        with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.json"), "w"
        ) as f:
            json.dump(ans_list, f)

        return result_records


@registry.register_task("retrieval_crepe")
class CREPETask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        # run_cfg = cfg.run_cfg

        return cls(cfg=cfg)

    def evaluation(self, model, data_loader, **kwargs):
        # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
        scores, all_keys = model.compute_grouped_simmat(data_loader, k_test=self.cfg.run_cfg.k_test)

        if is_main_process():
            eval_result = self._report_metrics(
                scores, all_keys
            )
            logger.info(eval_result)
        else:
            eval_result = None

        return eval_result

    def after_evaluation(self, val_result, **kwargs):
        for record in val_result:
            record.update({"Model": self.cfg.model_cfg.arch})

        output_file = os.path.join(registry.get_path("output_dir"),
                                   "results.csv")
        df = pd.DataFrame(val_result)
        print(f"Saving results to {output_file}")
        if os.path.exists(output_file):
            all_df = pd.read_csv(output_file, index_col=0)
            all_df = pd.concat([all_df, df])
            all_df.to_csv(output_file)
        else:
            df.to_csv(output_file)
        return val_result

    @torch.no_grad()
    def _report_metrics(self, scores, all_keys):
        """
        Scores: N x k, i.e. first caption is the true one, the others are wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
        else:
            scores_i2t = scores

        # import ipdb; ipdb.set_trace()
        metrics = {"R@1": None, "R@3": None}
        # preds = np.argmax(scores_i2t, axis=-1) # no image dim included because always i2t
        # correct_mask = (preds == 0)
        # metrics["R@1"] = np.mean(correct_mask)

        ranks = -np.ones(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == 0)[0][0]
        ranks = ranks[ranks!=-1]

        # Compute metrics
        metrics["R@1"] = len(np.where(ranks < 1)[0]) / len(ranks)
        metrics["R@3"] = len(np.where(ranks < 3)[0]) / len(ranks)
        # metrics["R@5"] = len(np.where(ranks < 5)[0]) / len(ranks) # sanity check

        all_complexities = np.array(all_keys)

        result_records = []
        result_records.append({
            "Complexity": "ALL",
            "R@1": metrics["R@1"] * 100,
            "R@3": metrics["R@3"] * 100,
            "Count": len(scores_i2t),
        })

        # Log the recall of all complexities
        acc_all = []
        R3_all = []
        for complexity in np.unique(all_complexities):
            type_mask = (all_complexities == complexity)
            if type_mask.sum() == 0:
                continue
            acc_type = len(np.where(ranks[type_mask] < 1)[0]) / type_mask.sum()
            R3_type = len(np.where(ranks[type_mask] < 3)[0]) / type_mask.sum()
            acc_all.append(acc_type)
            R3_all.append(R3_type)
            result_records.append({
                "Complexity": complexity,
                "R@1": acc_type * 100,
                "R@3": R3_type * 100,
                "Count": type_mask.sum(),
            })

        result_records.append({
            "Complexity": "MEAN",
            "R@1": np.mean(acc_all) * 100,
            "R@3": np.mean(R3_all) * 100,
            "Count": len(acc_all),
        })

        return result_records