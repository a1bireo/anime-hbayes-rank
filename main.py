# ------------------------------------------------------------------------------
# Project:     anime-score-analysis-ranking
# Name:        main
# Purpose:
#
# Author:      Atomic
#
# Created:     2020/3/20
# ------------------------------------------------------------------------------
from utils.rank import *
from utils.log import Log
from utils.table import Matrix


class RateSummary(object):

    def __init__(self, update=False):
        log = Log()
        csv_file = 'data/algorithm_matrix/_te st_matrix.csv'
        self.ms = MasseyRank(csv_file, log.get_log(), update=update)
        self.kn = KeenerRank(csv_file, log.get_log(), update=update)
        self.od = OffenseDefenseRank(csv_file)

    def solve(self):
        self.ms.solve()
        self.kn.solve()
        self.od.solve()


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(description="动漫评分排名——融合算法")
    parser.add_argument("csv", nargs="?", default="data/_subject_user_matrix.csv",
                        help="条目-用户矩阵 CSV 路径")
    parser.add_argument("--compare", action="store_true",
                        help="运行对比评估（含所有基线方法）")
    parser.add_argument("--update", action="store_true",
                        help="强制重新计算中间矩阵")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print("错误: 数据文件不存在: {}".format(args.csv))
        print("请先运行 load.py 从数据库导出 CSV。")
        exit(1)

    if args.compare:
        # 横向对比模式
        from fusion.evaluate import compare_all
        compare_all(args.csv)
    else:
        # 仅运行融合算法
        from fusion.pipeline import FusionRank
        fusion = FusionRank(args.csv)
        fusion.fit()
        fusion.save()
        print("\nTop 20:")
        print(fusion.rank(top_n=20))
