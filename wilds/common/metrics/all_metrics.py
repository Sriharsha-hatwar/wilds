import contextlib
import torch
import torch.nn as nn
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
from torchvision.ops import nms, box_convert
import numpy as np
import torch.nn.functional as F
from wilds.common.metrics.detection_utils import evaluate_det
from wilds.common.metrics.metric import Metric, ElementwiseMetric, MultiTaskMetric
from wilds.common.metrics.loss import ElementwiseLoss
from wilds.common.utils import avg_over_groups, minimum, maximum, get_counts
import sklearn.metrics
from scipy.stats import pearsonr

def binary_logits_to_score(logits):
    assert logits.dim() in (1,2)
    if logits.dim()==2: #multi-class logits
        assert logits.size(1)==2, "Only binary classification"
        score = F.softmax(logits, dim=1)[:,1]
    else:
        score = logits
    return score

def multiclass_logits_to_pred(logits):
    """
    Takes multi-class logits of size (batch_size, ..., n_classes) and returns predictions
    by taking an argmax at the last dimension
    """
    assert logits.dim() > 1
    return logits.argmax(-1)

def binary_logits_to_pred(logits):
    return (logits>0).long()

class Accuracy(ElementwiseMetric):
    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'acc'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        return (y_pred==y_true).float()

    def worst(self, metrics):
        return minimum(metrics)

class MultiTaskAccuracy(MultiTaskMetric):
    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn  # should work on flattened inputs
        if name is None:
            name = 'acc'
        super().__init__(name=name)

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if self.prediction_fn is not None:
            flattened_y_pred = self.prediction_fn(flattened_y_pred)
        return (flattened_y_pred==flattened_y_true).float()

    def worst(self, metrics):
        return minimum(metrics)

class Recall(Metric):
    def __init__(self, prediction_fn=None, name=None, average='binary'):
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'recall'
            if average is not None:
                name+=f'-{average}'
        self.average = average
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred, average=self.average, labels=torch.unique(y_true))
        return torch.tensor(recall)

    def worst(self, metrics):
        return minimum(metrics)

class F1(Metric):
    def __init__(self, prediction_fn=None, name=None, average='binary'):
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'F1'
            if average is not None:
                name+=f'-{average}'
        self.average = average
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        score = sklearn.metrics.f1_score(y_true, y_pred, average=self.average, labels=torch.unique(y_true))
        return torch.tensor(score)

    def worst(self, metrics):
        return minimum(metrics)

class PearsonCorrelation(Metric):
    def __init__(self, name=None):
        if name is None:
            name = 'r'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        r = pearsonr(y_pred.squeeze().detach().cpu().numpy(), y_true.squeeze().detach().cpu().numpy())[0]
        return torch.tensor(r)

    def worst(self, metrics):
        return minimum(metrics)

def mse_loss(out, targets):
    assert out.size()==targets.size()
    if out.numel()==0:
        return torch.Tensor()
    else:
        assert out.dim()>1, 'MSE loss currently supports Tensors of dimensions > 1'
        losses = (out - targets)**2
        reduce_dims = tuple(list(range(1, len(targets.shape))))
        losses = torch.mean(losses, dim=reduce_dims)
        return losses

class MSE(ElementwiseLoss):
    def __init__(self, name=None):
        if name is None:
            name = 'mse'
        super().__init__(name=name, loss_fn=mse_loss)

class PrecisionAtRecall(Metric):
    """Given a specific model threshold, determine the precision score achieved"""
    def __init__(self, threshold, score_fn=None, name=None):
        self.score_fn = score_fn
        self.threshold = threshold
        if name is None:
            name = "precision_at_global_recall"
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        score = self.score_fn(y_pred)
        predictions = (score > self.threshold)
        return torch.tensor(sklearn.metrics.precision_score(y_true, predictions))

    def worst(self, metrics):
        return minimum(metrics)

class DummyMetric(Metric):
    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'dummy'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        return torch.tensor(-1)

    def _compute_group_wise(self, y_pred, y_true, g, n_groups):
        group_metrics = torch.ones(n_groups, device=g.device) * -1
        group_counts = get_counts(g, n_groups)
        worst_group_metric = self.worst(group_metrics)
        return group_metrics, group_counts, worst_group_metric

    def worst(self, metrics):
        return minimum(metrics)

class DetectionAccuracy(ElementwiseMetric):
    """
    Given a specific Intersection over union threshold,
    determine the accuracy achieved for a one-class detector
    """
    def __init__(self, prediction_fn=None, iou_threshold=0.5,score_threshold=0.5, name=None):
        self.prediction_fn = prediction_fn
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        if name is None:
            name = "detection_accuracy"
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred ,y_true ):


        batch_results = []
        for src_boxes, target in zip( y_true, y_pred):
            target_boxes = target["boxes"]
            target_scores = target["scores"]
            # Here should be prediction_fn ?

            #target_scores =  F.softmax(target_logits, dim=1)[..., 0]
            pred_boxes = target_boxes[target_scores > self.score_threshold]

            det_accuracy = torch.mean(torch.stack([ self._accuracy(src_boxes["boxes"],pred_boxes,iou_thr) for iou_thr in np.arange(0.5,0.76,0.05)]))
            batch_results.append(det_accuracy)

        return torch.tensor(batch_results)


    def _accuracy(self, src_boxes,pred_boxes ,  iou_threshold):
        total_gt = len(src_boxes)
        total_pred = len(pred_boxes)


        if total_gt > 0 and total_pred > 0:

            # Define the matcher and distance matrix based on iou
            matcher = Matcher(iou_threshold,iou_threshold,allow_low_quality_matches=False)

            #src_boxes = box_convert(src_boxes , "cxcywh" ,"xyxy")
            #pred_boxes = box_convert(pred_boxes , "cxcywh" ,"xyxy")


            match_quality_matrix = box_iou(src_boxes,pred_boxes)

            results = matcher(match_quality_matrix)

            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]

            #in Matcher, a pred element can be matched only twice
            false_positive = torch.count_nonzero(results == -1) + ( len(matched_elements) - len(matched_elements.unique()))
            false_negative = total_gt - true_positive
            acc= true_positive / ( true_positive + false_positive + false_negative )


            return  true_positive / ( true_positive + false_positive + false_negative )            

        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.)
            else:
                return torch.tensor(1.)
        elif total_gt > 0 and total_pred == 0:
            return torch.tensor(0.)



    def worst(self, metrics):
        return minimum(metrics)


class MulticlassDetectionAccuracy(Metric):
    def __init__(self, id_to_cat, name=None):
        self.id_to_cat = id_to_cat
        if name is None:
            name = 'multiclass_detection_accuracy'
        super(MulticlassDetectionAccuracy, self).__init__(name=name)

    def _compute(self, y_pred, y_true, separate_classes=False):
        labels, preds = [], []
        pred_id = 1

        for src_boxes, target in zip(y_true, y_pred):
            name = src_boxes['name']
            frame = {'name': name, 'labels': None, 'attributes': {}}

            for label_id in range(len(src_boxes['boxes'])):
                label = {'box2d': src_boxes['boxes'][label_id].tolist(),
                         'category': self.id_to_cat[src_boxes['labels'][label_id].item()],
                         'id': label_id,
                         'attributes': {}}
                label['box2d'] = {k: v for k, v in zip(['x1', 'y1', 'x2', 'y2'],
                                                       label['box2d'])}
                if frame['labels'] is None:
                    frame['labels'] = []
                frame['labels'].append(label)
            labels.append(frame)

            for pred_id in range(len(target['boxes'])):
                preds.append({'category': self.id_to_cat[target['labels'][pred_id].item()],
                              'name': name,
                              'id': pred_id,
                              'score': target['scores'][pred_id].item(),
                              'bbox': target['boxes'][pred_id].tolist()})
        with contextlib.redirect_stdout(None):
            return torch.tensor(evaluate_det(labels, preds, separate_classes=separate_classes))

    def _compute_group_wise(self, y_pred, y_true, g, n_groups, n_classes):
        group_metrics = []
        group_counts = get_counts(g, n_groups)
        for group_idx in range(n_groups):
            y_pred_group, y_true_group = [], []
            if group_counts[group_idx]==0:
                group_metrics.append(torch.zeros(n_classes, device=g.device))
            else:
                for i in range(len(g)):
                    if g[i] == group_idx:
                        y_pred_group.append(y_pred[i])
                        y_true_group.append(y_true[i])
                group_metrics.append(
                    self._compute(y_pred_group, y_true_group, separate_classes=True)
                )

        group_metrics = torch.stack(group_metrics)
        worst_group_metric = self.worst(group_metrics[group_counts>0])

        return group_metrics, group_counts, worst_group_metric

    def compute_group_wise(self, y_pred, y_true, g, n_groups, n_classes, return_dict=True):
        group_metrics, group_counts, worst_group_metric = \
                self._compute_group_wise(y_pred, y_true, g, n_groups, n_classes)
        if return_dict:
            results = {}
            for group_idx in range(n_groups):
                for cls in range(n_classes):
                    results[self.group_metric_field(group_idx, cls)] = group_metrics[group_idx, cls].item()
                results[self.group_count_field(group_idx)] = group_counts[group_idx].item()
            results[self.worst_group_metric_field] = worst_group_metric.item()
            return results
        else:
            return group_metrics, group_counts, worst_group_metric

    def group_metric_field(self, group_idx, cls):
        return f'{self.name}_group:{group_idx}_y:{cls}'

    def worst(self, metrics):
        return minimum(metrics)
