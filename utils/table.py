# ------------------------------------------------------------------------------
# Project:     anime-score-analysis-ranking
# Name:        table
# Purpose:
#
# Author:      Atomic
#
# Created:     2020/3/27
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os


class Matrix(object):

    def __init__(self, csv_file, logger):
        self.matrix_s_u = pd.read_csv(csv_file, index_col=0)
        self.logger = logger

    def matrix_initial(self):
        """将条目-用户评分矩阵转化为胜负场数矩阵。该步骤比较耗时"""
        # 矩阵原始定义
        A = self.matrix_s_u.T.to_numpy().astype(np.float)
        B = A - A
        ext = A[:, np.newaxis]

        wins = np.array([])
        score = np.array([])

        self.logger.info("开始计算中间矩阵")
        for i in range(ext.shape[0]):
            _wins = (ext[i] > A).astype(np.float) + (ext[i] == A) / 2
            _score = ext[i] - B
            wins = np.append(wins, np.nansum(_wins, axis=1))
            score = np.append(score, np.nansum(_score, axis=1))
            if (i + 1) % 10 == 0:
                self.logger.debug("已完成{}/{}行".format(i + 1, ext.shape[0]))
        self.logger.info("中间矩阵计算完成，共计{}行".format(ext.shape[0]))

        wins = wins.reshape(ext.shape[0], ext.shape[0])
        score = score.reshape(ext.shape[0], ext.shape[0])

        wins = pd.DataFrame(wins, index=self.matrix_s_u.columns, columns=self.matrix_s_u.columns)
        score = pd.DataFrame(score, index=self.matrix_s_u.columns, columns=self.matrix_s_u.columns)
        keener = pd.DataFrame(keener, index=self.matrix_s_u.columns, columns=self.matrix_s_u.columns)
        wins.to_csv('utils/algorithm_matrix/matrix_wins.csv')
        score.to_csv('utils/algorithm_matrix/matrix_score.csv')
        keener.to_csv('utils/algorithm_matrix/matrix_keener.csv')
