# Import the require libraries
import os
import sys
from dataclasses import dataclass
from tourism.exception import TourismException
from tourism.logger import logging   
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PowerTransformer
from imblearn.combine import SMOTEENN

from tourism.utils import MainUtils

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        self.utils = MainUtils()

    def get_data_transformation_object(self) -> object:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object
        """
        try:
            logging.info(
                "Got numerical, categorical, transformation columns from schema config"
            )

            schema_info = self.utils.read_schema_config_file()

            Discrete_columns = schema_info["Discrete_columns"]

            Continuous_columns = schema_info["Continuous_columns"]

            Categorical_columns = schema_info["Categorical_columns"]

            Transformation_columns = schema_info["Transformation_columns"]

            logging.info(
                "Got numerical cols,one hot cols,binary cols from schema config"
            )

            logging.info("Initialized Data Transformer pipeline.")

            discrete_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler()),
                ]
            )

            continuous_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            transform_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("transformer", PowerTransformer(standardize=True)),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("Discrete_Pipeline", discrete_pipeline, Discrete_columns),
                    ("Continuous_Pipeline", continuous_pipeline, Continuous_columns),
                    ("Categorical_Pipeline", cat_pipeline, Categorical_columns),
                    ("Power_Transformation", transform_pipe, Transformation_columns),
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )

            return preprocessor

        except Exception as e:
            logging.info("Exception occured in Data Transformation Phase")
            raise TourismException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline
        """
        try:
            ## Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformation_object()

            logging.info("Got the preprocessor object.")

            TARGET_COLUMN = 'ProdTaken'

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)

            target_feature_train_df = train_df[TARGET_COLUMN]

            logging.info("Got train features and test features of Training dataset")

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info("Got train features and test features of Testing dataset")

            logging.info("Applying preprocessing object on training and testing datasets.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info("Used the preprocessor object to fit transform the train features")
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logging.info("Used the preprocessor object to transform the test features")

            logging.info("Applying SMOTEENN on Training dataset")

            smt = SMOTEENN(sampling_strategy="minority", random_state=42)

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            logging.info("Applied SMOTEENN on training dataset")

            logging.info("Applying SMOTEENN on Testing dataset")

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )

            logging.info("Applied SMOTEENN on testing dataset")

            logging.info("Created train array and test array")

            train_arr = np.c_[
                input_feature_train_final, np.array(target_feature_train_final)
            ]

            test_arr = np.c_[
                input_feature_test_final, np.array(target_feature_test_final)
            ]

            self.utils.save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Saved the preprocessor object")

            logging.info(
                "Exited initiate_data_transformation method of Data_Transformation class"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info('Exception occured in initiate_data_transformation function')
            raise TourismException(e, sys)