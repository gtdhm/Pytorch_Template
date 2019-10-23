# =======================util.utils.py=========================
# This module contains simple helper functions.
#
# Version: 1.0.0
# Date: 2019.05.20
# =============================================================

import os
import math
import time
import csv
import json
# The total length of information on the screen
TOTAL_LENGTH = 76
assert TOTAL_LENGTH in range(47, 80), "[Error] TOTAL_LENGTH should in [47, 80]!"


###############################################################
# Files Processing Functions
###############################################################
def mkdirs(paths):
    """Create empty directories if they don't exist
    Inputs:
        paths: a list of directory paths(str list)
    """
    def mkdir(dirs):
        slash = "\\" if os.name == "nt" else "/"
        _dirs = dirs.split(slash)
        for i in range(1, len(_dirs)):
            if not os.path.exists(dirs.split(_dirs[i])[0] + _dirs[i]):
                os.mkdir(dirs.split(_dirs[i])[0] + _dirs[i])
    if isinstance(paths, list):
        for path in paths:
            mkdir(path)
    elif isinstance(paths, str):
        mkdir(paths)
    else:
        raise IOError("[Error] Path -> '{:s}' was not right!".format(paths))


def wrote_txt_file(_new_txt_dir, raw_data, mode='w', show=True):
    """Write the raw_data in txt format."""
    if show:
        equal_left, equal_right = cal_equal(18)
        print("\n%s Processing Files %s" % ('=' * equal_left, '=' * equal_right))
    if mode in ['w', 'w+', 'wb', 'wb+']:
        if os.path.exists(_new_txt_dir):
            os.remove(_new_txt_dir)
            if show:
                print("Remove Last Txt File -> {:s}".format(_new_txt_dir))
    with open(_new_txt_dir, mode) as opt_file:
        opt_file.write(raw_data)
        opt_file.write('\n')
    if show:
        print("Written New Txt File -> {:s}".format(_new_txt_dir))
        print("=" * TOTAL_LENGTH)


def open_json_file(_data_dir):
    """Open the json format file into the python variable."""
    if not os.path.exists(_data_dir):
        raise IOError("[Error] Json Path -> '{:s}' was not found...".format(_data_dir))

    json_file = open(os.path.abspath(_data_dir), "r")
    raw_data = json.load(json_file)
    json_file.close()

    return raw_data


def wrote_json_file(_new_json_dir, raw_data, mode='w', show=True):
    """Write the raw_data in json format."""
    if show:
        equal_left, equal_right = cal_equal(18)
        print("\n%s Processing Files %s" % ('=' * equal_left, '=' * equal_right))
    if mode in ['w', 'w+', 'wb', 'wb+']:
        if os.path.exists(_new_json_dir):
            os.remove(_new_json_dir)
            if show:
                print("Remove Last Json File -> {:s}".format(_new_json_dir))

    with open(_new_json_dir, mode) as fid:
        json.dump(raw_data, fid)
    if show:
        print("Written New Json File -> {:s}".format(_new_json_dir))
        print("=" * TOTAL_LENGTH)


def open_csv_file(_data_dir):
    """Open the csv format file into the python variable."""
    if not os.path.exists(_data_dir):
        raise IOError("[Error] Csv Path -> '{:s}' was not found...".format(_data_dir))

    birth_data = []
    csv_reader = csv.reader(open(_data_dir, 'r', encoding='utf-8-sig'))
    for row in csv_reader:
        birth_data.append(row)
    return birth_data


def wrote_csv_file(_new_csv_dir, raw_data, mode='w', show=True):
    """ Write the raw_data in csv format."""
    if show:
        equal_left, equal_right = cal_equal(18)
        print("\n%s Processing Files %s" % ('=' * equal_left, '=' * equal_right))
    if mode in ['w', 'w+', 'wb', 'wb+']:
        if os.path.exists(_new_csv_dir):
            os.remove(_new_csv_dir)
            if show:
                print("Remove Last Csv File -> {:s}".format(_new_csv_dir))
    # 打开文件，追加a
    with open(_new_csv_dir, mode, newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for one_data in raw_data:
            csv_writer.writerow(one_data)
    if show:
        print("Written New Csv File -> {:s}".format(_new_csv_dir))
        print("=" * TOTAL_LENGTH)


###############################################################
# Helper Functions
###############################################################
def cal_equal(chars_len, length=TOTAL_LENGTH):
    """Calculate the number of equal signs on the both sides of
    the string in the display information.
    Inputs:
        chars_len: the length of the string
        length: total length of the display information
    """
    assert type(chars_len) == int, "chars_len should be int type..."

    equal_len = (length - chars_len) / 2
    equal_left = math.ceil(equal_len)
    equal_right = math.floor(equal_len)
    return equal_left, equal_right


def standard_time_display(cur_time):
    """Covert time with second format into hour:minute:second format.
    Inputs:
        cur_time: current time(s), type(int)
    Returns:
        stand_time: standard time(h:m:s), type(char)
    """
    assert type(cur_time) is float
    # ms
    if cur_time < 1:
        msecond = round(cur_time * 1000)
        stand_time = '{}ms'.format(msecond)
        length = 4
    # >=1min
    elif 60 <= cur_time < 3600:
        second = round(cur_time % 60)
        minute = round(cur_time / 60)
        stand_time = '{:02d}:{:02d}'.format(minute, second)
        length = 5
    # >=1h
    elif 3600 <= cur_time < 86400:
        second = round((cur_time % 3600) % 60)
        minute = round((cur_time % 3600) / 60)
        hour = round(cur_time / 3600)
        stand_time = '{:02d}:{:02d}:{:02d}'.format(hour, minute, second)
        length = 8
    # >=1day
    elif cur_time >= 86400:
        minute = round(((cur_time % 86400) % 3600) / 60)
        hour = round((cur_time % 86400) / 3600)
        day = round(cur_time / 86400)
        stand_time = '{:02d}d{:02d}h{:02d}min'.format(day, hour, minute)
        length = 11
    # 1s>= and <1min
    else:
        second = round(cur_time)
        stand_time = '00:{:02d}'.format(second)
        length = 5

    return stand_time, length


def progress_bar(ind, long, describe='Default Progress', display=True, size=11):
    """Display the progress of the program running.
    Inputs:
        i: current variable size(int)
        long: variable size at completion(int)
        size: total length of the progress bar
    """
    ind += 1
    if ind == 1:
        global start
        start = time.time()
        if display:
            equal_left, equal_right = cal_equal(len(describe)+2)
            print("\n%s %s %s" % ('=' * equal_left, describe, '=' * equal_right))

    current = time.time() - start
    cur_time, cur_length = standard_time_display(current)
    total_time, total_length = standard_time_display((current / ind) * long)
    size = TOTAL_LENGTH - cur_length - total_length - size
    if TOTAL_LENGTH >= 60:
        size = size - len(str(ind)) - len(str(long)) - 6
    # 按比例缩小
    if long >= size:
        cache = long / size
        i_op = round(ind / cache)
    else:
        cache = size / long
        i_op = round(ind * cache)

    jd = '\r%2d%%[%s%s]>>> '
    if TOTAL_LENGTH >= 60:
        jd += '{}/{}, '.format(ind, long)
    jd += '{}/{}'.format(cur_time, total_time)
    a = '█' * i_op
    b = ' ' * (size - i_op)
    if ind >= long:
        c = 100
    else:
        c = int((i_op / size) * 100)
    print(jd % (c, a, b), end='')


def list_sort(list_dat, index=0, mode="int", separator=None):
    """Sort the members in the list.
    Inputs:
        mode: int or boxes or chars
            if int, [2, 5, 1, ...]
            if boxes, [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]
            if chars, ["[DemoNet]_epoch:5_step:60_batch:16_lr:0.01_mode:optim_time:0919170727.pth",
                       "[DemoNet]_epoch:1_step:60_batch:16_lr:0.01_mode:optim_time:0920070714.pth", ...]
        index: the index element
        separator: used to separate characters when mode is chars
    Return:
        a sorted list
    """
    judge1 = judge2 = None
    list_data = list_dat.copy()
    list_len = len(list_data) - 1
    if separator is None:
        separator = [".", "_", ":"]

    for i in range(list_len):
        sort = True
        for j in range(list_len - i):
            if mode == "int":
                judge1 = list_data[j]
                judge2 = list_data[j + 1]
            if mode == "boxes":
                judge1 = list_data[j][index]
                judge2 = list_data[j + 1][index]
            if mode == "chars":
                judge1 = ((list_data[j].split(separator[0])[-2]).split(separator[1])[index]).split(separator[2])[-1]
                judge2 = ((list_data[j + 1].split(separator[0])[-2]).split(separator[1])[index]).split(separator[2])[-1]

            if int(judge1) > int(judge2):
                temp = list_data[j]
                list_data[j] = list_data[j + 1]
                list_data[j + 1] = temp
                sort = False
        if sort:
            break
    return list_data


