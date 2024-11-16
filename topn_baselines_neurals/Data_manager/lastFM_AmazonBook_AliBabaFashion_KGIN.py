#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""

import pandas as pd
import zipfile, shutil
from topn_baselines_neurals.Data_manager.DataReader import DataReader
from topn_baselines_neurals.Data_manager.DatasetMapperManager import DatasetMapperManager
from topn_baselines_neurals.Data_manager.Movielens._utils_movielens_parser import _loadURM, _loadICM_genres_years
from topn_baselines_neurals.Data_manager.split_functions.KGIN_given_train_test_splits import split_train_test_validation


class lastFM_AmazonBook_AliBabaFashion_KGIN(DataReader):

    DATASET_URL = "https://github.com/NLPWM-WHU/IDS4NR/blob/main/movielens_100k/movielens100k_longtail_data.pkl"
    DATASET_SUBFOLDER = "Movielens100M_given/"
    CONFERENCE_JOURNAL = "KGIN/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = ["ICM_genres"]
    AVAILABLE_UCM = ["UCM_all"]
    IS_IMPLICIT = False
    
    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER
    def _load_data_from_give_files(self, datapath, dataset = "lastFM", dataLeakage = False, validation = False , validation_portion = 0.1):
        
        zipFile_path = datapath
        try:
            train_dictionary = self.read_cf(zipFile_path / "train.txt")
            test_dictionary = self.read_cf(zipFile_path / "test.txt")
            if dataset == "lastFm" and dataLeakage == True:
                keys_with_dataLeakage = [key for key, item in train_dictionary.items() if len(test_dictionary[key].intersection(train_dictionary[key])) > 0]

                keys_itemLisDataLeakage = dict()
                for key in keys_with_dataLeakage:
                    keys_itemLisDataLeakage[key] = train_dictionary[key].union(test_dictionary[key])
                
                keyToRemove  = [key for key, items in keys_itemLisDataLeakage.items() if len(items) < 2]
                for key in keyToRemove:
                    del keys_itemLisDataLeakage[key]

                new_train, new_test = self.dataSplitingDataLeakage(keys_itemLisDataLeakage) 
                ############
                for key, _ in new_train.items():
                    train_dictionary[key] = new_train[key]
                    test_dictionary[key] = new_test[key]
                
                for key in keyToRemove:
                    del train_dictionary[key]
                    del test_dictionary[key]
                train_dictionary_temp = train_dictionary.copy()
                test_dictionary_temp = test_dictionary.copy()

                for key, _ in train_dictionary_temp.items():
                    train_dictionary[key] = list(train_dictionary_temp[key])
                    test_dictionary[key] = list(test_dictionary_temp[key])
                   
                

                
            

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
    
    def read_cf(self,file_name):
        temp_dictionary = dict()
        lines = open(file_name, "r").readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(" ")]
            u_id, pos_ids = inters[0], inters[1:]
            temp_dictionary[u_id] = set(pos_ids)
        return temp_dictionary
    def dataSplitingDataLeakage(self,keys_itemLisDataLeakage):
        new_train, new_test = dict(), dict()
        for key, items in keys_itemLisDataLeakage.items():
            temp_list = list(items)
            if len(temp_list) < 5:
                new_train[key] =  set(temp_list[:-1])
                new_test[key] =   set([temp_list[-1]])
            else:
                selectedRatio = int(len(temp_list) * 0.2)
                new_train[key] =  set(temp_list[:-selectedRatio])
                new_test[key] =   set(temp_list[-selectedRatio:])
        return new_train, new_test
        


       
        











