# -*- coding: utf-8 -*-
# @Time    : 2019-01-03 11:46
# @Author  : Maximus
# @Site    : 
# @File    : extract_features_test.py
# @Software: PyCharm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import logging
from extract_features import *
import json
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string("bert_json_dir", None, "")

flags.DEFINE_string("raw_data_dir", None, "")

flags.DEFINE_integer("layer_index", -2, "")
flags.DEFINE_string("dest_dir", None, "")


def get_cls_embedding(bert_json_dir, raw_data_dir, layer_index, dest_dir):
    raw_data_df = pd_reader(raw_data_dir, 0 if raw_data_dir.endswith('csv') else None, [0, 1, 2], drop_dup_axis=2)
    mat_data = None
    base_name = Path(raw_data_dir).name
    base_name = base_name[:base_name.rindex('.')]
    texts = []
    if not Path(dest_dir).is_dir():
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(dest_dir).joinpath('{}.npy'.format(base_name))
    with open(bert_json_dir, 'r') as f:
        for line in f:
            setence_tokens = json.loads(line)
            line_index = setence_tokens['linex_index']
            print('line_index:{}'.format(line_index))
            text = str(line_index)
            for token in setence_tokens['features']:
                token_name = token['token']
                if token_name == '[CLS]':
                    pass
                    # for layer_token in token['layers']:
                    #     if layer_token['index'] == layer_index:
                    #         value = np.array(layer_token['values']).astype('float16')
                    #         if mat_data is None:
                    #             mat_data = np.zeros((raw_data_df.shape[0], len(value)), dtype='float16')
                    #         mat_data[line_index] = value
                else:
                    text += token_name

            texts.append(text)

    # np.save(out_file, mat_data)
    texts.sort()
    with open(Path(dest_dir).joinpath('{}_check_texts.txt'.format(base_name)), 'w', encoding='utf8') as f:
        f.writelines(texts)


def main(_):
    get_cls_embedding(FLAGS.bert_json_dir, FLAGS.raw_data_dir, FLAGS.layer_index, FLAGS.dest_dir)


if __name__ == '__main__':
    flags.mark_flag_as_required("bert_json_dir")
    flags.mark_flag_as_required("raw_data_dir")
    flags.mark_flag_as_required("layer_index")
    flags.mark_flag_as_required("dest_dir")
    tf.app.run()
