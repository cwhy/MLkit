import json
import os
import os.path as op
import uuid
import h5py
from time import strftime, localtime
from typing import Dict, NewType, Any
import numpy as np


ResultFolder = NewType('ResultFolder', str)


def set_up(data_uid: str, config: Dict[str, Any], save_dir: str) -> ResultFolder:
    env = dict()
    env['data_uid'] = data_uid
    env['exp_id'] = uuid.uuid4().hex[:3]
    env['time'] = strftime("(%z) %Hh:%Mm:%Ss, %d/%m/%Y", localtime())
    env['config'] = config
    folder_name = data_uid + '_' + env['exp_id']
    result_folder = op.join(save_dir, folder_name)
    os.makedirs(result_folder, exist_ok=True)
    with open(op.join(result_folder, 'config.json'), 'w') as f_:
        json.dump(env, f_, indent=4)
    return ResultFolder(result_folder)


def save(result_folder: ResultFolder,
         logs: Dict[str, Any],
         vars_: Dict[str, np.ndarray],
         params: Dict[str, Any],
         remark: str=""):
    results = {'remark': remark,
               'params': params,
               'logs': logs,
               'vars': {k: f'{remark}_{k}.h5' for k in vars_.keys()}}
    with open(op.join(result_folder, 'results.json'), 'w') as f_:
        json.dump(results, f_, indent=4)
    for k, v in vars_.items():
        var_path = op.join(result_folder, f'{remark}_{k}.h5')
        with h5py.File(var_path, 'w') as hf:
            hf.create_dataset(k, data=v)


