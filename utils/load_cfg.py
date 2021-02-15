import os 
import ast 
import sys 
import shutil
import tempfile
from importlib import import_module

class Config(object):
    def __init__(self, cfg_dict):
        for key in list(cfg_dict.keys()):
            setattr(self, key, cfg_dict[key])

def _syntax_check(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {file_path}: {e}')

def _load_py(file_path):
    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py')
        temp_config_name = os.path.basename(temp_config_file.name)
        shutil.copyfile(file_path, temp_config_file.name)
        temp_module_name = os.path.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        del sys.modules[temp_module_name]
        temp_config_file.close()
        return cfg_dict


def get_cfg_dict(file_path):
    """get model config dict.
    Parameters:
    ----------
    file_path: str
        The path of config file, which should be a .py file.
    """
    file_path = os.path.abspath(os.path.expanduser(file_path))
    assert os.path.isfile(file_path)
    assert file_path.endswith('.py'), "Config file should be a python file."

    _syntax_check(file_path)

    cfg_dict = _load_py(file_path)

    return Config(cfg_dict) 