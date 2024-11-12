
from topn_baselines_neurals.Recommenders.DGCF_SIGIR_20.DGCF import *
import argparse
from pathlib import Path
import pandas as pd
# *************************************************** #
# Main file to run experiments......
def parse_args():
    parser = argparse.ArgumentParser(description="Run DGCF.")
    #parser.add_argument('--data_path', nargs='?', default=PATH_GETDATA,help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',help='Project path.')
    parser.add_argument('--pick', type=int, default=0,help='O for no pick, 1 for pick')
    parser.add_argument('--pick_scale', type=float, default=1e10,help='Scale')
    parser.add_argument('--dataset', nargs='?', default="gowalla", help='Choose a dataset from {yelp2018, gowalla, amazonbook}')
    parser.add_argument('--pretrain', type=int, default=0, help='0: No pretrain, 1:Use stored models.')
    parser.add_argument('--embed_name', nargs='?', default='', help='Name for pretrained model.')
    parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1, help='Number of epochs')      
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64]', help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=800, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--cor_flag', type=int, default=1, help='Correlation matrix flag')
    parser.add_argument('--corDecay', type=float, default=0.0, help='Distance Correlation Weight')
    parser.add_argument('--regs', nargs='?', default='[1e-3,1e-4,1e-4]', help='Regularizations.')    
    parser.add_argument('--n_layers', type=int, default=1, help='Layer numbers.')
    parser.add_argument('--n_factors', type=int, default=4, help='Number of factors to disentangle the original embed-size representation.')
    parser.add_argument('--n_iterations', type=int, default=2, help='Number of iterations to perform the routing mechanism.')
    parser.add_argument('--show_step', type=int, default=15, help='Test every show_step epochs.')
    parser.add_argument('--early', type=int, default=40, help='Step for stopping')           
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Metrics scale')
    parser.add_argument('--save_flag', type=int, default=0, help='0: Disable model saver, 1: Save Better Model')
    parser.add_argument('--save_name', nargs='?', default='best_model', help='Save_name.')
    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    return parser.parse_args()

args = parse_args()
dataset_name  = args.dataset
data_path = Path("data/DGCF/"+dataset_name)
data_path = data_path.resolve()
result_dict = run_experiments(data_path, args = args)
commonFolderName = "results"
model = "DGCF"
saved_results = "/".join([commonFolderName,model,args.dataset] )
df = pd.DataFrame()
for key in result_dict:
        print( key +": "   +str(result_dict[key].getScore())  )
        df[key] = [result_dict[key].getScore()]
df.to_csv(saved_results+model+"_"+args.dataset+".txt")

# python run_experiments_for_DGCF_algorithm.py --dataset yelp2018
# python run_experiments_for_DGCF_algorithm.py --dataset gowalla
# python run_experiments_for_DGCF_algorithm.py --dataset amazonbook

 
