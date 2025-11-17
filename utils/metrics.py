# Adapted from
# https://github.com/BellyBeauty/MDSAM/blob/master/py_sod_metrics/sod_metrics.py


# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist

import numpy as np

# the different implementation of epsilon (extreme min value) between numpy and matlab
EPS = np.spacing(1)
TYPE = np.float64

def prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """
    A numpy-based function for preparing ``pred`` and ``gt``.

    - for ``pred``, it looks like ``mapminmax(im2double(...))`` of matlab;
    - ``gt`` will be binarized by 128.

    :param pred: prediction
    :param gt: mask
    :return: pred, gt
    """
    gt = gt > 128
    # im2double, mapminmax
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    """
    Return an adaptive threshold, which is equal to twice the mean of ``matrix``.

    :param matrix: a data array
    :param max_value: the upper limit of the threshold
    :return: min(2 * matrix.mean(), max_value)
    """
    return min(2 * matrix.mean(), max_value)


class Fmeasure:
    """
    F-measure calculator for Salient Object Detection.
    
    Implementation based on:
    @inproceedings{Fmeasure,
        title={Frequency-tuned salient region detection},
        author={Achanta, Radhakrishna and Hemami, Sheila and Estrada, Francisco and Süsstrunk, Sabine},
        booktitle=CVPR,
        pages={1597--1604},
        year={2009}
    }
    
    Args:
        beta (float): Weight parameter for precision in F-measure calculation. 
                     Defaults to 0.3 as in the original paper.
    """
    
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.beta_squared = beta ** 2
        self._reset_accumulators()
    
    def _reset_accumulators(self) -> None:
        """Reset all accumulators for new evaluation."""
        self.precisions: List[np.ndarray] = []
        self.recalls: List[np.ndarray] = []
        self.adaptive_fms: List[float] = []
        self.changeable_fms: List[np.ndarray] = []
        self.max_fms: List[float] = []
    
    def step(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """
        Process a single prediction-ground truth pair.
        
        Args:
            pred: Prediction array with values in [0, 1]
            gt: Ground truth array with values in {0, 1}
        """
        pred, gt = prepare_data(pred, gt)
        
        # Calculate adaptive F-measure
        adaptive_fm = self._calculate_adaptive_fm(pred, gt)
        self.adaptive_fms.append(adaptive_fm)
        
        # Calculate precision-recall curves and F-measures
        precisions, recalls, changeable_fms, max_fm = self._calculate_pr_curves(pred, gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)
        self.max_fms.append(max_fm)
    
    def _calculate_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate F-measure using adaptive threshold.
        
        Args:
            pred: Normalized prediction in [0, 1]
            gt: Binary ground truth
            
        Returns:
            Adaptive F-measure value
        """
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        binary_prediction = pred >= adaptive_threshold
        
        # Calculate intersection and cardinalities
        intersection = np.count_nonzero(binary_prediction & gt)
        pred_positives = np.count_nonzero(binary_prediction)
        gt_positives = np.count_nonzero(gt)
        
        # Handle edge cases
        if intersection == 0:
            return 0.0
        
        precision = intersection / pred_positives
        recall = intersection / gt_positives
        
        # F-measure formula: (1 + β²) * precision * recall / (β² * precision + recall)
        return (1 + self.beta_squared) * precision * recall / (self.beta_squared * precision + recall)
    
    def _calculate_pr_curves(self, pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Calculate precision-recall curves for all thresholds (0-255).
        
        Args:
            pred: Normalized prediction in [0, 1]
            gt: Binary ground truth
            
        Returns:
            Tuple of (precisions, recalls, fmeasures, max_fmeasure)
        """
        # Convert to 8-bit for histogram calculation
        pred_uint8 = (pred * 255).astype(np.uint8)
        
        # Calculate histograms for foreground and background
        bins = np.arange(257)  # 0 to 256
        fg_hist, _ = np.histogram(pred_uint8[gt], bins=bins)
        bg_hist, _ = np.histogram(pred_uint8[~gt], bins=bins)
        
        # Cumulative histograms for thresholds (from high to low)
        # flip + cumsum gives counts for thresholds >= value
        fg_cumulative = np.cumsum(np.flip(fg_hist))
        bg_cumulative = np.cumsum(np.flip(bg_hist))
        
        # Calculate TP, FP, FN for all thresholds
        true_positives = fg_cumulative
        false_positives = bg_cumulative
        total_predictions = true_positives + false_positives
        gt_positives = max(np.count_nonzero(gt), 1)  # Avoid division by zero
        
        # Calculate precision and recall
        precisions = np.divide(true_positives, total_predictions, 
                              out=np.zeros_like(true_positives, dtype=float),
                              where=total_predictions != 0)
        
        recalls = true_positives / gt_positives
        
        # Calculate F-measures for all thresholds
        fmeasures = self._compute_fmeasure(precisions, recalls)
        max_fmeasure = np.max(fmeasures)
        
        return precisions, recalls, fmeasures, max_fmeasure
    
    def _compute_fmeasure(self, precisions: np.ndarray, recalls: np.ndarray) -> np.ndarray:
        """
        Compute F-measure from precision and recall arrays.
        
        Args:
            precisions: Precision values
            recalls: Recall values
            
        Returns:
            F-measure values
        """
        numerator = (1 + self.beta_squared) * precisions * recalls
        denominator = self.beta_squared * precisions + recalls
        
        # Handle division by zero - when both precision and recall are zero
        valid_mask = denominator > 0
        fmeasures = np.zeros_like(numerator)
        fmeasures[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        
        return fmeasures
    
    def get_results(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get aggregated results across all processed samples.
        
        Returns:
            Dictionary containing:
            - fm: F-measure results (adaptive, curve, max)
            - pr: Precision-recall curves
            - mf: Maximum F-measure
        """
        if not self.adaptive_fms:
            raise ValueError("No data processed. Call step() method first.")
        
        # Calculate mean values across all samples
        adaptive_fm = np.mean(self.adaptive_fms).astype(TYPE)
        changeable_fm = np.mean(self.changeable_fms, axis=0).astype(TYPE)
        precision_curve = np.mean(self.precisions, axis=0).astype(TYPE)
        recall_curve = np.mean(self.recalls, axis=0).astype(TYPE)
        max_f = np.mean(self.max_fms).astype(TYPE)
        
        return {
            'fm': {
                'adp': adaptive_fm,      # Adaptive F-measure
                'curve': changeable_fm,  # F-measure curve across thresholds
                'max': max_f             # Maximum F-measure
            },
            'pr': {
                'p': precision_curve,    # Precision curve
                'r': recall_curve        # Recall curve
            },
            'mf': max_f                  # Maximum F-measure (convenience alias)
        }
    
    def reset(self) -> None:
        """Reset all accumulators for a new evaluation run."""
        self._reset_accumulators()
    
    @property
    def sample_count(self) -> int:
        """Get the number of processed samples."""
        return len(self.adaptive_fms)


class MAE(object):
    def __init__(self):
        """
        MAE(mean absolute error) for SOD.

        ::

            @inproceedings{MAE,
                title={Saliency filters: Contrast based filtering for salient region detection},
                author={Perazzi, Federico and Kr{\"a}henb{\"u}hl, Philipp and Pritch, Yael and Hornung, Alexander},
                booktitle=CVPR,
                pages={733--740},
                year={2012}
            }
        """
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = prepare_data(pred, gt)

        mae = self.cal_mae(pred, gt)
        self.maes.append(mae)

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the mean absolute error.

        :return: mae
        """
        mae = np.mean(np.abs(pred - gt))
        return mae

    def get_results(self) -> dict:
        """
        Return the results about MAE.

        :return: dict(mae=mae)
        """
        mae = np.mean(np.array(self.maes, TYPE))
        return dict(mae=mae)


class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        """
        S-measure(Structure-measure) of SOD.

        ::

            @inproceedings{Smeasure,
                title={Structure-measure: A new way to eval foreground maps},
                author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
                booktitle=ICCV,
                pages={4548--4557},
                year={2017}
            }

        :param alpha: the weight for balancing the object score and the region score
        """
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the S-measure.

        :return: s-measure
        """
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the object score.
        """
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1], ddof=1)
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the region score.
        """
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info["weight"]
        # assert np.isclose(w1 + w2 + w3 + w4, 1), (w1 + w2 + w3 + w4, pred.mean(), gt.mean())

        pred1, pred2, pred3, pred4 = part_info["pred"]
        gt1, gt2, gt3, gt4 = part_info["gt"]
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        """
        To ensure consistency with the matlab code, one is added to the centroid coordinate,
        so there is no need to use the redundant addition operation when dividing the region later,
        because the sequence generated by ``1:X`` in matlab will contain ``X``.

        :param matrix: a bool data array
        :return: the centroid coordinate
        """
        h, w = matrix.shape
        area_object = np.count_nonzero(matrix)
        if area_object == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            y, x = np.argwhere(matrix).mean(axis=0).round()
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x: int, y: int) -> dict:
        """
        Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
        """
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(
            gt=(gt_LT, gt_RT, gt_LB, gt_RB),
            pred=(pred_LT, pred_RT, pred_LB, pred_RB),
            weight=(w1, w2, w3, w4),
        )

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the ssim score.
        """
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x ** 2 + y ** 2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        """
        Return the results about S-measure.

        :return: dict(sm=sm)
        """
        sm = np.mean(np.array(self.sms, dtype=TYPE))
        return dict(sm=sm)


class Emeasure(object):
    def __init__(self):
        """
        E-measure(Enhanced-alignment Measure) for SOD.

        More details about the implementation can be found in https://www.yuque.com/lart/blog/lwgt38

        ::

            @inproceedings{Emeasure,
                title="Enhanced-alignment Measure for Binary Foreground Map Evaluation",
                author="Deng-Ping {Fan} and Cheng {Gong} and Yang {Cao} and Bo {Ren} and Ming-Ming {Cheng} and Ali {Borji}",
                booktitle=IJCAI,
                pages="698--704",
                year={2018}
            }
        """
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the adaptive E-measure.

        :return: adaptive_em
        """
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the changeable E-measure, which can be used to obtain the mean E-measure,
        the maximum E-measure and the E-measure-threshold curve.

        :return: changeable_ems
        """
        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        """
        Calculate the E-measure corresponding to the specific threshold.

        Variable naming rules within the function:
        ``[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]``

        If only ``pred`` or ``gt`` is considered, another corresponding attribute location is replaced with '``_``'.
        """
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel,
                pred_bg_numel=bg___numel,
            )

            results_parts = []
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = (
                    2
                    * (combination[0] * combination[1])
                    / (combination[0] ** 2 + combination[1] ** 2 + EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        em = enhanced_matrix_sum / (self.gt_size - 1 + EPS)
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the E-measure corresponding to the threshold that varies from 0 to 255..

        Variable naming rules within the function:
        ``[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]``

        If only ``pred`` or ``gt`` is considered, another corresponding attribute location is replaced with '``_``'.
        """
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs,
                pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = (
                    2
                    * (combination[0] * combination[1])
                    / (combination[0] ** 2 + combination[1] ** 2 + EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        em = enhanced_matrix_sum / (self.gt_size - 1 + EPS)
        return em

    def generate_parts_numel_combinations(
        self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel
    ):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        """
        Return the results about E-measure.

        :return: dict(em=dict(adp=adaptive_em, curve=changeable_em))
        """
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=TYPE), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))


class WeightedFmeasure(object):
    def __init__(self, beta: float = 1):
        """
        Weighted F-measure for SOD.

        ::

            @inproceedings{wFmeasure,
                title={How to eval foreground maps?},
                author={Margolin, Ran and Zelnik-Manor, Lihi and Tal, Ayellet},
                booktitle=CVPR,
                pages={248--255},
                year={2014}
            }

        :param beta: the weight of the precision
        """
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = prepare_data(pred=pred, gt=gt)

        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the weighted F-measure.
        """
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        # 注意这里使用mask索引矩阵的时候不可使用Ew[gt]，这实际上仅在索引Ew的0维度
        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + EPS)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        """
        Return the results about weighted F-measure.

        :return: dict(wfm=weighted_fm)
        """
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=TYPE))
        return dict(wfm=weighted_fm)
