# -*- coding: utf-8 -*-
# @Time    : 2019-01-03 14:28
# @Author  : Maximus
# @Site    : 
# @File    : data_helper.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import logging
from typing import Union, Set
import pandas as pd
import json
from pathlib import Path
logger = logging.getLogger(__name__)


def pd_reader(file_dir, header: Union[int, type(None)] = 0, usecols: Union[list, type(None)] = None, drop_value: Union[list, str, type(None)] = None,
              drop_axis: Union[int, type(None)] = None, drop_dup_axis: Union[int, type(None)] = None, sep='\t',
              drop_na_axis: Union[int, type(None)] = 0, qid_axis=0, label_str_axis=1, data_axis=2):
    """preprocess files drop special data, and keep useful columns"""
    if file_dir.endswith('.csv'):
        df = pd.read_csv(open(file_dir, 'rU'), engine='c', sep=sep, na_filter=True, skipinitialspace=True, header=header, usecols=usecols, lineterminator='\n')
        # df = pd.read_csv(open(file_dir, 'rU'), engine='python', sep=sep, na_filter=True, skipinitialspace=True, header=header, usecols=usecols)
    elif file_dir.endswith('.xlsx'):
        df = pd.read_excel(file_dir, header=header, usecols=usecols).dropna(axis=drop_na_axis, how='any')
    else:
        return
    logger.info('=========> source data:{} shape:{}'.format(file_dir[file_dir.rindex('/') + 1:], df.shape))
    # 自定义去掉某些行
    if drop_value is not None and drop_axis is not None:
        col_key = df.keys()[drop_axis]
        # 过滤多个值：多个语义或者多个语义id
        if isinstance(drop_value, (list, set, Set)):
            df = df[~df[col_key].isin(drop_value)]
        else:
            df = df[df[col_key] != drop_value]
        logger.info('=========> after drop value:{}'.format(df.shape))
    # 规范数据轴, 不然concat异常
    if len(usecols) == 3 and label_str_axis != qid_axis:
        df = df[[df.keys()[qid_axis], df.keys()[label_str_axis], df.keys()[data_axis]]]
    elif len(usecols) in [2, 3] and label_str_axis == qid_axis:
        assert qid_axis != data_axis
        df = df[[df.keys()[qid_axis], df.keys()[data_axis]]]
    elif len(usecols) == 1:
        df = df[[df.keys()[data_axis]]]
    else:
        raise AssertionError('WRONG USECOLS! with:{}'.format(usecols))
    if drop_dup_axis is not None:
        df = df.drop_duplicates(subset=[df.keys()[drop_dup_axis]])
        logger.info('=========> after drop dup:{}'.format(df.shape))
    return df


def load_json(save_path: str) -> Union[dict, list]:
    assert Path(save_path).is_file()
    with open(save_path, "r", encoding='utf8') as openfile:
        return json.load(openfile)


def save_json(save_path: str, json_obj):
    assert isinstance(json_obj, (dict, list))
    with open(save_path, "w", encoding='utf8') as openfile:
        json.dump(json_obj, openfile)


def save_readable_jsons(save_path: str, json_obj):
    assert isinstance(json_obj, (dict, list))
    with open(save_path, "w", encoding='utf8') as openfile:
        json.dump(json_obj, openfile, ensure_ascii=False)
