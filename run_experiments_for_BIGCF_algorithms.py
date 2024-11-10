from topn_baselines_neurals.Recommenders.BIGCF.BIGCF_main import *
from topn_baselines_neurals.Recommenders.Recommender_import_list import *
import os
from pathlib import Path
import argparse
import pandas as pd
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='amazonbook', help="amazonbook / gowalla / tmall")
    parser.add_argument('--epoch', type = int, default=300, help="amazonbook / gowalla / tmall")
    parser.add_argument('--ssl_reg', type=float, default=0.4, help='Reg weight for ssl loss')
    args = parser.parse_args()
    dataset_name = args.dataset

    print("<<<<<<<<<<<<<<<<<<<<<< Experiments are running for  "+dataset_name+" dataset Wait for results......")
    data_path = Path("data/DCCF/"+dataset_name)
    data_path = data_path.resolve()
    start = time.time()
    
    #best_epoch = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=True, epoch = args.epoch, ssl_reg = args.ssl_reg)
    #print("Start tuning by Best Epoch Value"+str(best_epoch))
    best_score = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=False, epoch = args.epoch, ssl_reg = args.ssl_reg)
    
    print("Recall@20: ", str(best_score["recall"][0]), " Recall@40: ", str(best_score["recall"][1]),
                        " NDCG@20: ",str(best_score["ndcg"][0]), " NDCG@40:", str(best_score["ndcg"][1]))
    end = time.time()
    df = pd.DataFrame()
    df["Recall@20"] = [best_score["recall"][0]]
    df["Recall@40"] = [best_score["recall"][1]]
    df["NDCG@20"] = [best_score["ndcg"][0]]
    df["NDCG@40"] = [best_score["ndcg"][1]]
    df["Time(seconds)"] = [end - start]
    commonFolderName = "results"
    model = "BIGCF"
    saved_results = "/".join([commonFolderName, model] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    df.to_csv(saved_results + "/"+args.dataset+"BIGCF.txt", index = False)
    
    
    