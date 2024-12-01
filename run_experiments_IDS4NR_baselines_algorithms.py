
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from topn_baselines_neurals.Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from topn_baselines_neurals.Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from topn_baselines_neurals.Evaluation.Evaluator import EvaluatorHoldout
from topn_baselines_neurals.Recommenders.IDS4NR.IDSNR import *
from topn_baselines_neurals.Data_manager.IDS4NR_MovleLensBeautyMusic import IDS4NR_MovleLensBeautyMusic
import os
import argparse

from pathlib import Path
def _get_instance(recommender_class, URM_train, ICM_all, UCM_all):
    if issubclass(recommender_class, BaseItemCBFRecommender):
        recommender_object = recommender_class(URM_train, ICM_all)
    elif issubclass(recommender_class, BaseUserCBFRecommender):
        recommender_object = recommender_class(URM_train, UCM_all)
    else:
        recommender_object = recommender_class(URM_train)
    return recommender_object

def tempfun(dataset = "MovieLens"):
    
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='Beauty', help="MovieLens/Music/Beauty")
    parser.add_argument('--model', type = str, default='LFM', help="LFM or NCF")
    args = parser.parse_args()

    args.dataset = dataset
    dataset_name = args.dataset
    
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset MovieLens --model NCF
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Beauty --model NCF
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Music --model NCF
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset MovieLens --model LFM
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Beauty --model LFM
    # python run_experiments_IDS4NR_baselines_algorithms.py --dataset Music --model LFM
    
    print("<<<<<<<<<<<<<<<<<<<<<< Experiments are running for  "+dataset_name+" dataset Wait for results......")
    commonFolderName = "results"
    data_path = Path("data/ID4SNR/")
    data_path = data_path.resolve()
    datasetName = args.dataset+".pkl"
    dataset_object = IDS4NR_MovleLensBeautyMusic()
    URM_train, URM_test, UCM_all = dataset_object._load_data_from_give_files(data_path = data_path / datasetName)
    ICM_all = None

    total_elements = URM_train.shape[0] * URM_train.shape[1]
    non_zero_elements = URM_train.nnz + URM_test.nnz
    density = non_zero_elements / total_elements
    print("Number of users: %s, Items: %d, Interactions: %d, Density %.5f, Number of users with no test items: %d." % 
          (URM_train.shape[0], URM_train.shape[1], non_zero_elements, density, np.sum(np.diff(URM_test.indptr) == 0)))

    resultFolder = "results"
    saved_results = "/".join([resultFolder,"ID4SNR",dataset_name] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    output_root_path = saved_results+"/"
    # Run experiments for IDS4NR
    
    
    recommender_class_list = [
        Random,
        TopPop,
        ItemKNNCFRecommender,
        UserKNNCFRecommender,
        P3alphaRecommender,
        RP3betaRecommender,
        EASE_R_Recommender

        ]
    
    
    
    evaluator = EvaluatorHoldout(URM_test, [1, 5, 10, 20, 50, 100], exclude_seen=True)
    for recommender_class in recommender_class_list:
        try:
            print("Algorithm: {}".format(recommender_class))
            recommender_object = _get_instance(recommender_class, URM_train, ICM_all, UCM_all)

            
            if(dataset_name == "Music"):
                if isinstance(recommender_object, RP3betaRecommender):
                    fit_params = {"topK": 814, "alpha": 0.13435726416026783, "beta": 0.27678107504384436, "normalize_similarity": True}
                else:
                    fit_params = {}
            elif(dataset_name == "Beauty"):
                print("********************************")
                if isinstance(recommender_object, P3alphaRecommender):
                    fit_params = {"topK": 790, "alpha": 0.0, "normalize_similarity": False}
                elif isinstance(recommender_object, RP3betaRecommender):
                    fit_params = {"topK": 1000, "alpha": 0.0, "beta": 0.0, "normalize_similarity": False}
                else:
                    fit_params = {}
            else:
                fit_params = {}

            # measure training time.....
            start = time.time()
            recommender_object.fit(**fit_params)
            training_time = time.time() - start
            
            # testing for all records.....
            start = time.time()
            results_run_1, results_run_string_1 = evaluator.evaluateRecommender(recommender_object)
            testing_time = time.time() - start
            averageTestingForOneRecord = (testing_time / len(URM_test.getnnz(axis=1) > 0)) * 1000 # get number of non-zero rows in test data
        
            results_run_1["TrainingTime(s)"] = [training_time] + [0 for i in range(results_run_1.shape[0] - 1)]
            results_run_1["TestingTimeforRecords(s)"] = [testing_time] + [0 for i in range(results_run_1.shape[0] - 1)]
            results_run_1["AverageTestingTimeForOneRecord(ms)"] = [averageTestingForOneRecord] + [0 for i in range(results_run_1.shape[0] - 1)]
            
            print("Algorithm: {}, results: \n{}".format(recommender_class, results_run_string_1))
            results_run_1["cuttOff"] = results_run_1.index
            results_run_1.insert(0, 'cuttOff', results_run_1.pop('cuttOff'))
            results_run_1.to_csv(saved_results+"/"+args.dataset+"_"+recommender_class.RECOMMENDER_NAME+".txt", sep = "\t", index = False)

        except Exception as e:
            pass
    obj1 = Run_experiments_for_IDSNR(model = args.model, dataset = data_path / datasetName, NumberOfUsersInTestingData = URM_test.shape[0])
    accuracy_measure = obj1.accuracy_values
    accuracy_measure.to_csv(output_root_path+args.dataset+"__"+args.model+".txt", index = False, sep = "\t")

if __name__ == '__main__':
    listt = ["MovieLens",  "Music", "Beauty"]
    for data in listt:
        tempfun(dataset = data)
