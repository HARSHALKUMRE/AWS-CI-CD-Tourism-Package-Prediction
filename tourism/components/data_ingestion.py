# Import all requires libraries
import os
import sys
from tourism.exception import TourismException
from tourism.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from tourism.components.data_transformation import DataTransformation, DataTransformationConfig

# Initialize data ingestion configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

# Create a class for data ingestion
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Started.")
        try:
            df = pd.read_csv("G:\\100-days-of-dl\\Krish Naik\\FSDS Ineuron Course\\projects\\AWS-CI-CD-Tourism-Package-Prediction\\notebooks\\data\\Travel.csv")
            logging.info("Dataset read as a pandas Dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)

            logging.info("Train Test Split Initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            drop_columns = 'CustomerID'

            train_set = train_set.drop(drop_columns, axis=1)
            test_set = test_set.drop(drop_columns, axis=1)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of Data is completed.")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage.')
            raise TourismException(e,sys)

# Run Data Ingestion
if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    #train_data, test_data = obj.initiate_data_ingestion()

    #data_transformation = DataTransformation()
    #train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data) 

