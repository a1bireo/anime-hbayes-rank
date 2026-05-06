# ------------------------------------------------------------------------------
# Project:     anime-score-analysis-ranking
# Name:        log
# Purpose:
#
# Author:      Atomic
#
# Created:     2020/3/25
# ------------------------------------------------------------------------------
import logging
import datetime


class Log(object):

    def __init__(self, logger=None):
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        date = str(datetime.date.today())
        handler = logging.FileHandler(filename='{}.log'.format(date), encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s [%(module)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_log(self):
        return self.logger

