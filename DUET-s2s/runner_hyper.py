# -*- coding: utf-8 -*-
import numpy as np
import datetime
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
import os
import pickle

from dataset import DUETHyperData
from trainer import Hyper_Train
from duet import DUET
from utils import check_path, set_seed, EarlyStopping, get_matrix_and_num, get_sample


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default='DUET-hyper-prefreeze', type=str)
    parser.add_argument("--data_name", default='ml-1m', type=str)
    parser.add_argument("--data_dir", default='../data/ml-1m/duet_s2s/', type=str)
    parser.add_argument("--output_dir", default='output/', type=str)
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # optimizer
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--lr_dc", type=float, default=0.9, help='learning rate decay.')
    parser.add_argument("--lr_dc_step", type=int, default=10,
                        help='the number of steps after which the learning rate decay.')
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    # transformer
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--hidden_act", default="gelu", type=str, help="activation function")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout")

    # train args
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of batch_size")
    parser.add_argument("--max_seq_length", default=30, type=int, help="max sequence length")
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of model")
    parser.add_argument("--seed", default=2022, type=int, help="seed")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--patience", default=10, type=int, help="early stop patience")

    # fairness group
    parser.add_argument("--pre_train_sample", default=4, type=int, help="sample number")
    parser.add_argument("--hyper_train_sample", default=4, type=int, help="sample number")
    parser.add_argument("--base_train_sample", default=4, type=int, help="sample number")
    parser.add_argument("--dynamic_length", default=5, type=int, help="dynamic length")
    parser.add_argument("--num_generator", default=3, type=int, help="dynamic length")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    args.sample_file = args.data_dir + args.data_name + '_sample.txt'
    user_num, item_num, valid_rating_matrix, test_rating_matrix = \
        get_matrix_and_num(args.data_file)
    sample_seq = get_sample(args.sample_file)

    train_data = pickle.load(open(args.data_dir + 'base_train.pkl', 'rb'))
    valid_data = pickle.load(open(args.data_dir + 'base_valid.pkl', 'rb'))
    test_data = pickle.load(open(args.data_dir + 'base_test.pkl', 'rb'))

    args.item_size = item_num
    args.user_size = user_num

    # save model args
    args_str = f'{args.model_name}-{args.data_name}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(args)
    with open(args.log_file, 'a') as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n')
        f.write(str(args) + '\n')

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    model = DUET(args=args)

    train_dataset = DUETHyperData(args, train_data, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True, num_workers=8)

    eval_dataset = DUETHyperData(args, valid_data, test_neg_items=sample_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = DUETHyperData(args, test_data, test_neg_items=sample_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    trainer = Hyper_Train(model, args)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.eval_stage(0, test_dataloader, full_sort=False, test=True)

    else:
        pretrained_path = os.path.join(args.output_dir, f'DUET-base-nopre-{args.data_name}.pt')
        try:
            trainer.load(pretrained_path)
            for k, v in trainer.model.named_parameters():
                if 'encoder_layer' in k or 'item_encoder' in k:
                    v.requires_grad=False
            print(f'Load Checkpoint From {pretrained_path}!')

        except FileNotFoundError:
            print(f'{pretrained_path} Not Found! The Pretrain Model is not Initialization')

        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train_stage(epoch, train_dataloader)
            scores, _ = trainer.eval_stage(epoch, eval_dataloader, full_sort=False, test=False)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.eval_stage(0, test_dataloader, full_sort=False, test=True)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')


main()

