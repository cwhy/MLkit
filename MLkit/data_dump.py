import uuid
from time import strftime, localtime
from typing import Optional, Any, Dict

import toml
import json
import pickle
import os
import os.path as op
from shutil import copyfile


def save_data(info_dict: Dict[str, Any],
              data_dict: [str, Any],
              out_dir: str,
              name: Optional[str],
              uid_len: Optional[int] = 4,
              use_toml: bool =False) -> str:
    _time = strftime("(%z) %Hh:%Mm:%Ss, %d/%m/%Y", localtime())
    time_tag = strftime("%H:%M_%d%m", localtime())
    g_info = {'time': _time}
    if uid_len is not None:
        uid = uuid.uuid4().hex[:uid_len]
        data_dir = op.join(out_dir, uid)
        g_info['uid'] = uid
    else:
        data_dir = op.join(out_dir, time_tag)
    g_info['data_dir'] = data_dir
    os.makedirs(data_dir, exist_ok=True)
    info_dict['Generation Information'] = g_info

    if use_toml:
        with open(op.join(data_dir, 'data_info.toml'), 'w') as file_:
            toml.dump(info_dict, file_)
    with open(op.join(data_dir, 'data_info.json'), 'w') as file_:
        json.dump(info_dict, file_, indent=4)

    with open(op.join(data_dir, 'data.pkl'), 'wb') as file_:
        pickle.dump(data_dict, file_, pickle.HIGHEST_PROTOCOL)
    print(f'Saved {data_dir} at {_time}')
    return data_dir


def load_data(data_dir: str, data_uid: str) -> Dict[str, Any]:
    with open(op.join(data_dir, data_uid, 'data.pkl'), 'rb') as file_:
        data = pickle.load(file_)
    return data


def load_info(data_dir: str, data_uid: str) -> Dict[str, Any]:
    with open(op.join(data_dir, 'data_info.json')) as file_:
        return json.load(file_)

def copy_info(data_dir: str, data_uid: str, out_dir: str) -> None:
    copyfile(op.join(data_dir, data_uid, 'data_info.json'),
             op.join(out_dir, 'data_info.json'))
