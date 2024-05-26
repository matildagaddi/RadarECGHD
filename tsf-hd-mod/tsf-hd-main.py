import argparse
import os
import numpy as np
import torch
from exp import ExpARHD, ExpSeq2SeqHD


def main():
    parser = argparse.ArgumentParser(description="[Informer] Long Sequences Forecasting")
    parser.add_argument("--data", type=str, required=True, default="ETTh1", help="data")
    parser.add_argument("--root_path", type=str, default="/Users/matildagaddi/Documents/SEElab/DATASET/trainVal/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="radar/GDN0001_Resting_radar_6.mat", help="data file")
    parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
    parser.add_argument("--freq", type=str, default="h", help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h")
    parser.add_argument("--seq_len", type=int, default=400, help="input sequence length of Informer encoder")
    parser.add_argument("--pred_len", type=int, default=1, help="prediction sequence length")
    parser.add_argument("--hvs_len", type=int, default=24, help="dimension of the hypervectors")
    parser.add_argument("--label_len", type=int, default=48, help="start token length of Informer decoder")
    parser.add_argument("--features", type=str, default="M", help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",) #I think we are MS: 1000:1
    # parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument("--cols", type=str, nargs="+", help="certain cols from the data files as the input features")

    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    parser.add_argument("--test_bsz", type=int, default=-1)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--use_multi_gpu", type=bool, default=True, help="use use_multi_gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")
    parser.add_argument("--method", type=str, default="seq2seq-HDC", help="choose seq2seq-HDC or AR-HDC")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="choose learning rate : default 1e-3")
    parser.add_argument("--l2_lambda", type=float, default=2e-3, help="choose regularization rate l2 parameter: default 2e-3")

    args = parser.parse_args()

    training_files = []
    testing_files = []

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {"data": "radar", "T": "ecg", "M": [], "S": [], "MS": []} 
    #in and out Multivar, in and out Univar, in Multi to out Uni
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info["data"]
        args.target = data_info["T"]

        # Exp = Exp_TS2VecSupervised


    Exps = {"seq2seq-HDC": ExpSeq2SeqHD, "AR-HDC": ExpARHD, "RegHDC": ExpRegHD} # "AR-HDC": ExpARHD
    Exp = Exps[args.method] #AR-HDC

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    for tr_file in training_files:
        for ts_file in testing_files:
            exp = Exp(args, [tr_file], [ts_file])
            exp.train()
            mae_, rmse_, p, t = exp.test()
            exit()
    
    

if __name__ == "__main__":
    main()
