# ------------------------------------------------------------------------------
# Project:     anime-score-analysis-ranking
# Name:        t
# Purpose:
#
# Author:      Atomic
#
# Created:     2020/3/24
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import time


def func_1(A):
    return np.matmul(A.T, A)


def func_2(A):
    C = np.zeros((A.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        C[i, i] = 0
        for j in range(i+1, A.shape[1]):
            tmp = (A[:, i] > A[:, j]).sum()
            C[i, j] = tmp
            C[j, i] = A.shape[0] - tmp
    return C


def func(A):
    print(A)
    B = A[:, np.newaxis]
    print(B)
    D = B > A
    print(D)
    C = D.sum(axis=2)
    print(C)
    return C

# def func_3(A):
#     print(A)
#     C = A.T
#     _3d_matrix = np.array(np.roll(C, -1, axis=0))
#     for i in range(1, C.shape[0]):
#         shift = np.roll(C, -i-1, axis=0)
#         _3d_matrix = np.append(_3d_matrix, shift)
#     _3d_matrix = _3d_matrix.reshape((C.shape[0], C.shape[1], C.shape[0]))
#
#     diff = C - _3d_matrix
#     a = 0


class Compare(object):
    def __init__(self):
        self.matrix_s_u = pd.read_csv('data/algorithm_matrix/_test_matrix.csv', index_col=0)

    def matrix_initial_1(self):
        """
        将条目-用户评分矩阵转化为梅西矩阵: 三个矩阵分别记录：梅西矩阵，(右向量推导)分差矩阵，(右向量推导)胜场数矩阵
        该步骤非常耗时！！！O(n^2), n为条目数
        """
        keener_matrix = pd.DataFrame(np.nan, index=self.matrix_s_u.columns, columns=self.matrix_s_u.columns)
        length = len(self.matrix_s_u.columns)
        for i in range(length):
            keener_matrix.iat[i, i] = 0
            for j in range(i + 1, length):
                df_2c = self.matrix_s_u.iloc[:, [i, j]]
                df_2c = df_2c.dropna()
                if df_2c.index.size == 0:
                    keener_matrix.iat[i, j] = 0.5
                    keener_matrix.iat[j, i] = 0.5
                else:
                    column_1, column_2 = df_2c.iloc[:, 0], df_2c.iloc[:, 1]
                    s1 = 1 + column_1
                    s2 = 1 / (2 + column_1 + column_2)
                    _sum = np.multiply(s1, s2)
                    keener_matrix.iat[i, j] = _sum.mean()
                    keener_matrix.iat[j, i] = 1 - _sum.mean()
        return keener_matrix

    def matrix_initial_2(self):
        A = self.matrix_s_u.T.to_numpy()
        # massey_matrix = (1 - np.isnan(A[:, np.newaxis] - A)).sum(axis=2)
        # massey_diff_matrix = np.nansum((A[:, np.newaxis] - A), axis=2)
        # massey_wins_matrix = np.nansum((A[:, np.newaxis] > A), axis=2)

        matrix = np.nanmean(((1 + A[:, np.newaxis]) / (2 + A[:, np.newaxis] + A)), axis=2)
        keener_matrix = pd.DataFrame(matrix, index=self.matrix_s_u.columns, columns=self.matrix_s_u.columns).fillna(0.5)
        return keener_matrix

    # def matrix_initial_3(self):
    #     A = self.matrix_s_u.T.to_numpy()
    #     matrix = np.array([])
    #     ext = A[:, np.newaxis]
    #     for i in range(ext.shape[0]):
    #         matrix = np.append(matrix, np.nanmean(((1 + ext[i]) / (2 + ext[i] + A)), axis=1))
    #     matrix = matrix.reshape(ext.shape[0], ext.shape[0])
    #     keener_matrix = pd.DataFrame(matrix, index=self.matrix_s_u.columns, columns=self.matrix_s_u.columns).fillna(0.5)
    #     return keener_matrix


if __name__ == "__main__":
    c = Compare()

    kn_1 = c.matrix_initial_1()
    kn_2 = c.matrix_initial_2()
    print(kn_1)
    print(kn_2)
    print(kn_1 - kn_2)
    kn_3 = c.matrix_initial_3()
    print(kn_3 - kn_2)
    # a = np.array([1,2,3,4,5])
    # b = np.array([2,4,6,8,10])
    # print(a/b)
