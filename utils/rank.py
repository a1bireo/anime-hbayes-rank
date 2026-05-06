# ------------------------------------------------------------------------------
# Project:     anime-score-analysis-ranking
# Name:        base_rank
# Purpose:
#
# Author:      Atomic
#
# Created:     2020/3/25
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os


class MasseyRank(object):

    def __init__(self, csv_file, logger, update=False):
        self.matrix_s_u = pd.read_csv(csv_file, index_col=0)
        self.logger = logger
        if not os.path.exists('data/algorithm_matrix/massey_matrix.csv') or update:
            self.matrix_initial()
        self.solution_diff = pd.Series(1)
        self.solution_wins = pd.Series(1)

    def solve(self):
        massey_matrix = pd.read_csv('data/algorithm_matrix/massey_matrix.csv', index_col=0)
        matrix_diff = pd.read_csv('data/algorithm_matrix/massey_diff_matrix.csv', index_col=0)
        matrix_wins = pd.read_csv('data/algorithm_matrix/massey_wins_matrix.csv', index_col=0)
        massey_matrix.columns = [int(_) for _ in massey_matrix.columns]
        matrix_diff.columns = [int(_) for _ in matrix_diff.columns]
        matrix_wins.columns = [int(_) for _ in matrix_wins.columns]
        massey_matrix.iloc[-1] = 1
        vector_diff = matrix_diff.sum(axis=1)
        matrix_wins -= matrix_wins.T
        vector_wins = 1 + matrix_wins.sum(axis=1)/2
        self.solution_diff = pd.Series(np.linalg.solve(massey_matrix, vector_diff), index=massey_matrix.index)
        self.solution_wins = pd.Series(np.linalg.solve(massey_matrix, vector_wins), index=massey_matrix.index)
        self.solution_diff.sort_values(ascending=False).to_csv('solution/massey_diff.csv')
        self.solution_wins.sort_values(ascending=False).to_csv('solution/massey_wins.csv')

    def matrix_initial(self):
        """将条目-用户评分矩阵转化为梅西矩阵。该步骤比较耗时"""
        # 矩阵原始定义
        A = self.matrix_s_u.T.to_numpy()
        ext = A[:, np.newaxis]

        matrix = np.array([])
        diff_matrix = np.array([])
        wins_matrix = np.array([])
        self.logger.info("开始计算梅西矩阵")
        for i in range(ext.shape[0]):
            diff = ext[i] - A
            b = (ext[i] == A)/2
            wins = (ext[i] > A).astype(float) + (ext[i] == A)/2
            matrix = np.append(matrix, -(1 - np.isnan(diff)).sum(axis=1))
            diff_matrix = np.append(diff_matrix, np.nansum(diff, axis=1))
            wins_matrix = np.append(wins_matrix, np.nansum(wins, axis=1))
            if (i + 1) % 10 == 0:
                self.logger.debug("已完成{}/{}行".format(i + 1, ext.shape[0]))
        self.logger.info("梅西矩阵计算完成，共计{}行".format(ext.shape[0]))

        matrix = matrix.reshape(ext.shape[0], ext.shape[0])
        diff_matrix = diff_matrix.reshape(ext.shape[0], ext.shape[0])
        wins_matrix = wins_matrix.reshape(ext.shape[0], ext.shape[0])

        # 按定义处理矩阵
        for i in range(ext.shape[0]):
            matrix[i][i] += - matrix[i].sum()
            wins_matrix[i][i] = 0

        matrix = pd.DataFrame(matrix, index=self.matrix_s_u.columns, columns=self.matrix_s_u.columns)
        diff_matrix = pd.DataFrame(diff_matrix, index=self.matrix_s_u.columns, columns=self.matrix_s_u.columns)
        wins_matrix = pd.DataFrame(wins_matrix, index=self.matrix_s_u.columns, columns=self.matrix_s_u.columns)

        matrix.to_csv('data/algorithm_matrix/massey_matrix.csv')
        diff_matrix.to_csv('data/algorithm_matrix/massey_diff_matrix.csv')
        wins_matrix.to_csv('data/algorithm_matrix/massey_wins_matrix.csv')


class OffenseDefenseRank(object):
    """攻防评分法"""

    def __init__(self, csv_file):
        self.matrix_s_u = pd.read_csv(csv_file, index_col=0)
        self.matrix_s_u_0 = self.matrix_s_u.copy().fillna(0)
        self.mask = self.matrix_s_u.notnull()
        self.weight = pd.Series(1)
        self.solution = pd.Series(1)

    def solve(self):
        self.iteration()

    def iteration(self):
        """迭代求解"""
        vector_user = pd.Series(1, index=self.matrix_s_u.index)
        vector_subject = self.weighting_score(vector_user)
        i = 0
        while True:
            i += 1
            next_u = self.similarity(vector_subject)
            next_s = self.weighting_score(next_u)
            diff = abs(next_s - vector_subject)
            if diff.min() < 10e-7:
                break
            vector_subject = next_s
        self.weight = next_u
        self.solution = next_s
        self.weight.sort_values(ascending=False).to_csv('solution/od_weight.csv')
        self.solution.sort_values(ascending=False).to_csv('solution/od_rate.csv')

    def similarity(self, vector):
        """计算相似度作为评分权重"""
        similar = np.exp(-0.2 * np.power(self.matrix_s_u - vector, 2)).fillna(0)
        sum_similar = similar.sum(axis=1)
        sum_num = self.mask.sum(axis=1)
        return sum_similar / sum_num

    def weighting_score(self, vector):
        """计算加权评分"""
        sum_score = np.matmul(self.matrix_s_u_0.T, vector)
        sum_weight = np.matmul(self.mask.T, vector)
        return sum_score / sum_weight


class KeenerRank(object):
    """基纳评分法"""

    def __init__(self, csv_file, logger, update=False):
        self.matrix_s_u = pd.read_csv(csv_file, index_col=0)
        self.logger = logger
        if not os.path.exists('data/algorithm_matrix/keener_matrix.csv') or update:
            self.matrix_initial()
        self.solution = pd.Series(1)

    def solve(self):
        self.iteration()

    def iteration(self):
        """迭代求解"""
        keener_matrix = pd.read_csv('data/algorithm_matrix/keener_matrix.csv', index_col=0)
        keener_matrix.columns = [int(_) for _ in keener_matrix.columns]
        bm_x = pd.Series(1, index=keener_matrix.index)
        i = 0
        while True:
            i += 1
            bm_y = np.matmul(keener_matrix, bm_x)
            next_x = bm_y / bm_y.sum()
            diff = abs(next_x - bm_x)
            if diff.min() < 10e-7:
                break
            bm_x = next_x
        self.solution = bm_x
        self.solution.sort_values(ascending=False).to_csv('solution/keener_rate.csv')

    def matrix_initial(self):
        """将条目-用户评分矩阵转化为基纳矩阵。该步骤比较耗时"""
        A = self.matrix_s_u.T.to_numpy()
        matrix = np.array([])
        ext = A[:, np.newaxis]
        self.logger.info("开始计算基纳矩阵")
        for i in range(ext.shape[0]):
            matrix = np.append(matrix, np.nanmean(((1 + ext[i]) / (2 + ext[i] + A)), axis=1))
            if (i + 1) % 10 == 0:
                self.logger.debug("已完成{}/{}行".format(i + 1, ext.shape[0]))
        self.logger.info("基纳矩阵计算完成，共计{}行".format(ext.shape[0]))
        matrix = matrix.reshape(ext.shape[0], ext.shape[0])
        keener_matrix = pd.DataFrame(matrix, index=self.matrix_s_u.columns, columns=self.matrix_s_u.columns).fillna(0.5)
        keener_matrix.to_csv('data/algorithm_matrix/keener_matrix.csv')


class Markov(object):

    def __init__(self):
        pass


# if __name__ == "__main__":
    # od = OffenseDefenseRank('../data/algorithm_matrix/_subject_user_matrix_31w.csv')
    # od.iteration()
    # print(od.solution.sort_values(ascending=False).head(20))

    # kn = KeenerRank('data/algorithm_matrix/_subject_user_matrix.csv')
    # kn.solve()
    # print(kn.solution.sort_values(ascending=False).head(20))

    # ms = MasseyRank('data/algorithm_matrix/_subject_user_matrix.csv')
    # ms.solve()
    # print(ms.solution_diff.sort_values(ascending=False).head(20))
    # print(ms.solution_wins.sort_values(ascending=False).head(20))
