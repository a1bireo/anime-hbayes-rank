# ------------------------------------------------------------------------------
# Project:     anime-score-analysis-ranking
# Name:        load
# Purpose:
#
# Author:      Atomic
#
# Created:     2020/3/23
# ------------------------------------------------------------------------------
import sqlite3
import pandas as pd
import numpy as np


def load_sql_to_csv():
    conn = sqlite3.connect('data/anime_score_new.db')
    cur = conn.cursor()

    sql = "SELECT subject_id, COUNT(*) as count FROM bangumi WHERE score > 0 AND status = 'collect' " \
          "GROUP BY subject_id HAVING count >= 50"
    cur.execute(sql)
    subject_list = [_[0] for _ in cur.fetchall()]

    user_set = set()
    for i in range(0, len(subject_list), 100):
        sql = "SELECT num_id FROM bangumi WHERE subject_id IN ({}) AND score > 0 GROUP BY num_id". \
            format(', '.join(['?'] * len(subject_list[i:i+100])))
        cur.execute(sql, subject_list[i:i+100])
        user_set |= {_[0] for _ in cur.fetchall()}

    user_list = list(user_set)
    user_list.sort()
    subject_list.sort()
    ori_score = pd.DataFrame(np.nan, index=user_list, columns=subject_list)

    sql = "SELECT num_id, subject_id, score FROM bangumi WHERE score > 0 AND status = 'collect'"
    cur.execute(sql)

    i = 0
    while True:
        i += 1
        line = cur.fetchone()
        if not line:
            break
        else:
            num_id, subject_id, score = line

        if subject_id in subject_list:
            ori_score.at[num_id, subject_id] = score

        if i % 1000 == 0:
            print("已完成{}条记录导出".format(i))

    ori_score.to_csv('data/_subject_user_matrix.csv')


if __name__ == '__main__':
    load_sql_to_csv()
