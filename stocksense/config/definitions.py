import os.path

import yaml

ROOT_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))

"""
Get project root path and publish as a global variable.
"""


def get_config(config_file: str) -> dict:
    with open(
        os.path.join(ROOT_PATH, f"config/{config_file}_config.yml"), encoding="utf8"
    ) as file:
        return yaml.safe_load(file)
