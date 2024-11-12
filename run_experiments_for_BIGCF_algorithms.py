from topn_baselines_neurals.Recommenders.Recommender_import_list import *
from topn_baselines_neurals.Recommenders.BIGCF.BIGCF_main import *
from pathlib import Path
import pandas as pd
import argparse
import time
import ast
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accept data name as input')
    parser.add_argument('--dataset', type = str, default='amazonbook', help="amazonbook / gowalla / tmall")
    parser.add_argument('--epoch', type = int, default=300, help="amazonbook / gowalla / tmall")
    parser.add_argument('--ssl_reg', type=float, default=0.4, help='Reg weight for ssl loss')
    parser.add_argument('--Ks', nargs='?', default='[1, 5, 10, 20, 40, 50, 100]', help='Metrics scale')
    args = parser.parse_args()
    dataset_name = args.dataset

    print("<<<<<<<<<<<<<<<<<<<<<< Experiments are running for  "+dataset_name+" dataset Wait for results......")
    data_path = Path("data/DCCF/"+dataset_name)
    data_path = data_path.resolve()
    start = time.time()
    
    #best_epoch = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=True, epoch = args.epoch, ssl_reg = args.ssl_reg, ks = args.Ks)
    #print("Start tuning by Best Epoch Value"+str(best_epoch))
    best_score = model_tuningAndTraining(dataset_name=dataset_name, path =data_path, validation=False, epoch = args.epoch, ssl_reg = args.ssl_reg, ks = args.Ks)
    
    normal_list = ast.literal_eval(args.Ks)
    end = time.time()
    df = pd.DataFrame()
    for key in best_score:
        for i in range(len(normal_list)):
            if (key == "recall"):
                df[key+"@"+str(normal_list[i])] = [best_score[key][i]]
                print(key+"@"+str(normal_list[i]) +":"+str(best_score[key][i]))
            else:
                df[key+"@"+str(normal_list[i])] = [best_score[key][i]]
                print(key+"@"+str(normal_list[i]) +":"+str(best_score[key][i]))
    df["Time(seconds)"] = [end - start]

    commonFolderName = "results"
    model = "BIGCF"
    saved_results = "/".join([commonFolderName, model] )
    if not os.path.exists(saved_results):
        os.makedirs(saved_results)
    df.to_csv(saved_results + "/"+args.dataset+"BIGCF.txt", index = False)
    
    
    