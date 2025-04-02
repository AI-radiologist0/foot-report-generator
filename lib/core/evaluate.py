# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred


def clean_special_tokens(text, eos_token="[EOS]", pad_token="[PAD]"):
    """
    Remove text after the [EOS] token and remove all [PAD] tokens.
    
    Args:
        text (str): Input text to process.
        eos_token (str): The [EOS] token to look for.
        pad_token (str): The [PAD] token to remove.
    
    Returns:
        str: Cleaned text with [EOS] and [PAD] tokens handled.
    """
    # Remove text after [EOS]
    if eos_token in text:
        text = text.split(eos_token)[0].strip()
    # Remove all [PAD] tokens
    text = text.replace(pad_token, "").strip()
    return text

# 단일 쌍에 대해 BLEU, ROUGE-L 점수 계산
def compute_bleu_rouge(reference, prediction):
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()

    bleu = sentence_bleu([reference_tokens], prediction_tokens)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(reference, prediction)['rougeL'].fmeasure

    return bleu, rouge_l

# 여러 쌍에 대해 BERTScore 계산
def compute_bertscore(predictions, references, lang="en"):
    P, R, F1 = bert_score.score(predictions, references, lang=lang, verbose=False)
    return F1.tolist()  # 각 쌍별 F1 점수 리스트 반환