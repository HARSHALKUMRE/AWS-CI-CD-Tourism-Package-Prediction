import os
import sys

import numpy as np
import pandas as pd
import dill
import yaml
from typing import Dict, Tuple
from tourism.exception import TourismException
from tourism.logger import logging


SCHEMA_CONFIG_FILE = "config/schema.yaml"

class MainUtils:
    
    @staticmethod
    def save_object(file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)

        except Exception as e:
            logging.info('Exception Occured in save_object function utils')
            raise TourismException(e, sys)
    
    @staticmethod
    def load_object(file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            logging.info('Exception Occured in load_object function utils')
            raise TourismException(e, sys)


    def read_yaml_file(self, filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise TourismException(e, sys) from e


    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(SCHEMA_CONFIG_FILE)

            return schema_config
        except Exception as e:
            raise TourismException(e, sys)

        