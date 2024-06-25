import numpy as np
import os
import glob
import json
import pandas as pd
import ipdb

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, f_oneway, mannwhitneyu

from config import N_MARKERS

class Evaluation():
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.condition = None

    def rmse(self):
        rmse = mean_squared_error(self.y_true, self.y_pred, squared=False)
        return rmse

    def r2_score(self):
        return r2_score(self.y_true, self.y_pred)
    
    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def pearsonr(self):
        ''' looking for positive correlation p < 0.01 '''
        return pearsonr(self.y_true, self.y_pred)

    def err_std(self):
        return np.std(np.abs(self.y_true - self.y_pred))

    def get_stats(self, **kwargs):
        lbls_mu = np.mean(self.y_true, **kwargs)
        lbls_sd = np.std( self.y_true, **kwargs)

        pred_mu = np.mean(self.y_pred, **kwargs)
        pred_sd = np.std(self.y_pred, **kwargs)

        _, pval = f_oneway(self.y_pred, self.y_true)
        _, mpval = mannwhitneyu(self.y_pred, self.y_true)

        stats = {
            'lbls_mu' : lbls_mu,
            'lbls_sd' : lbls_sd,
            'pred_mu' : pred_mu,
            'pred_sd' : pred_sd,
            'anova'   : pval,
            'mannwu'  : mpval,
        }
        return stats

    def bland_altman(self):
        samples = np.array([self.y_true, self.y_pred]).T
        avgs = np.mean(samples, axis=-1)
        diffs = samples[:,0] - samples[:,1]
        return avgs, diffs

    def get_evals(self):
        coeff, pval = self.pearsonr()
        my_evals = {
            'mae': self.mae(),
            'std': self.err_std(),
            'rmse': self.rmse(),
            'r2': self.r2_score(),
            'pearsonr_coeff': coeff,
            'pearsonr_pval': pval,
        }
        return my_evals

class MarkerEvaluation():
    def __init__(self, y_true, y_pred):
        if len(y_pred.shape) > len(y_true.shape):
            n_elem = y_pred.shape[-1]
            y_true = np.repeat(y_true.reshape(-1,1), n_elem, axis=1)
        self.y_true = y_true
        self.y_pred = y_pred
        self.n_elem = 1
        try:
            self.n_elem = y_pred.shape[1]
        except: pass

    def get_stats(self, ind=None):
        if ind is not None:
            evals = Evaluation(self.y_true[:, ind], self.y_pred[:, ind])
        else:
            evals = Evaluation(self.y_true, self.y_pred)
        stats = evals.get_stats(axis=0)
        return stats

    def get_evals(self):
        marker_evals_dict = []
        for n in range(self.n_elem):
            evals = Evaluation(self.y_true[:,n], self.y_pred[:,n])
            marker_evals_dict.append(evals.get_evals())
        return marker_evals_dict

    def bland_altman(self):
        avgs, diff = [], []
        for n in range(self.n_elem):
            evals = Evaluation(self.y_true[:,n], self.y_pred[:,n])
            avg_vec, dif_vec = evals.bland_altman()
            avgs.append(avg_vec)
            diff.append(dif_vec)
        return np.array(avgs), np.array(diff)
