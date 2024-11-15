#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd
import zipfile, shutil
from topn_baselines_neurals.Data_manager.DataReader import DataReader
from topn_baselines_neurals.Data_manager.DataReader_utils import download_from_URL
from topn_baselines_neurals.Data_manager.DatasetMapperManager import DatasetMapperManager
from topn_baselines_neurals.Data_manager.Movielens._utils_movielens_parser import _loadURM, _loadICM_genres_years
import pickle
from topn_baselines_neurals.Data_manager.split_functions.DGCF_given_train_test_splits import split_train_test_validation


class Gowalla_Yelp_Amazon_DGCF(DataReader):

    DATASET_URL = "https://github.com/NLPWM-WHU/IDS4NR/blob/main/movielens_100k/movielens100k_longtail_data.pkl"
    DATASET_SUBFOLDER = "Movielens100M_given/"
    CONFERENCE_JOURNAL = "DGCF/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_genres"]
    AVAILABLE_UCM = ["UCM_all"]
    
    IS_IMPLICIT = False
    
    FILE_NAME = "movielens100k_longtail_data.pkl"
    


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER
    
    
    def _load_data_from_give_files(self, validation = False, data_name = "yelp2018"):
        
        zipFile_path = data_name
        train_dictionary = dict()
        test_dictionary = dict()
        try:

            with open(zipFile_path/ "train.txt") as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:]]
                        train_dictionary[l[0]] = items
            
            with open(zipFile_path/"test.txt") as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        try:
                            items = [int(i) for i in l[1:]]
                            test_dictionary[l[0]] = items
                        except:
                            pass

        except FileNotFoundError:
            print(f"File not found: {zipFile_path}")

        URM_dataframe = self.convert_dictionary_to_dataframe_DGCF(train_dictionary.copy(), test_dictionary.copy())

        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_dataframe, "URM_all")
        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        if validation == True:
            URM_train, URM_test, URM_validation_train, URM_validation_test = split_train_test_validation(loaded_dataset, test_dictionary, validation=validation, validation_portion = validation_portion)
            return URM_train, URM_test, URM_validation_train, URM_validation_test
        else:
            URM_train, URM_test = split_train_test_validation(loaded_dataset, test_dictionary,   validation=validation)
            return URM_train, URM_test
        
    def convert_dictionary_to_dataframe_DGCF(self, train_dictionary, test_dictionary):

        for key, _ in test_dictionary.items():
            train_dictionary[key]+=test_dictionary[key] 
        expanded_data = [(key, value) for key, values in train_dictionary.items() for value in values]
        # Create DataFrame
        URM_dataframe = pd.DataFrame(expanded_data, columns=['UserID', 'ItemID'])
        URM_dataframe["Data"] = 1
        URM_dataframe['UserID']= URM_dataframe['UserID'].astype(str)
        URM_dataframe['ItemID']= URM_dataframe['ItemID'].astype(str)

        return URM_dataframe


       
        











