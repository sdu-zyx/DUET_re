# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset

from Fairness_eval.utils import neg_sample


class DUETPreData(Dataset):

    def __init__(self, args, data, test_neg_items=None, data_type='train'):
        self.args = args

        # data = tuple(t[:10000] for t in data)
        self.data = data
        self.test_neg_items = test_neg_items
        self.max_len = args.max_seq_length

        self.uid_list = data[0]
        self.part_sequence = data[1]
        self.part_sequence_target = data[2]
        self.part_sequence_length = data[3]
        self.length = len(data[0])

        print(data_type, self.length)

    def __getitem__(self, index):

        input_ids = self.part_sequence[index]
        target_pos = self.part_sequence_target[index]
        user_id = self.uid_list[index]

        target_neg = []
        seq_set = set(input_ids)
        for l in input_ids:
            id_neg = []
            for n in range(self.args.pre_train_sample):
                id_neg.append(neg_sample(seq_set, self.args.item_size))
            target_neg.append(id_neg)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [[0] * self.args.pre_train_sample] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]
        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return self.length


class DUETHyperData(Dataset):

    def __init__(self, args, data, test_neg_items=None, data_type='train'):
        self.args = args

        # data = tuple(t[:10000] for t in data)
        self.data = data
        self.test_neg_items = test_neg_items
        self.max_len = args.max_seq_length
        self.dynamic_len = args.dynamic_length

        self.uid_list = data[0]
        self.part_sequence = data[1]
        self.part_sequence_target = data[2]
        self.part_sequence_length = data[3]
        self.length = len(data[0])

        print(data_type, self.length)

    def __getitem__(self, index):

        input_ids = self.part_sequence[index]
        target_pos = self.part_sequence_target[index]
        user_id = self.uid_list[index]

        target_neg = []
        seq_set = set(input_ids)
        for l in input_ids:
            id_neg = []
            for n in range(self.args.pre_train_sample):
                id_neg.append(neg_sample(seq_set, self.args.item_size))
            target_neg.append(id_neg)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [[0] * self.args.pre_train_sample] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]
        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        hyper_ids = input_ids[-self.dynamic_len:]
        pad_dynamic_len = self.dynamic_len - len(hyper_ids)
        hyper_ids = [0] * pad_dynamic_len + hyper_ids

        hyper_ids = hyper_ids[-self.dynamic_len:]
        assert len(hyper_ids) == self.dynamic_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(hyper_ids, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(hyper_ids, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return self.length


class DUETBaseData(Dataset):

    def __init__(self, args, data, test_neg_items=None, data_type='train'):
        self.args = args

        # data = tuple(t[:10000] for t in data)
        self.data = data
        self.test_neg_items = test_neg_items
        self.max_len = args.max_seq_length

        self.uid_list = data[0]
        self.part_sequence = data[1]
        self.part_sequence_target = data[2]
        self.part_sequence_length = data[3]
        self.length = len(data[0])

        print(data_type, self.length)

    def __getitem__(self, index):

        input_ids = self.part_sequence[index]
        target_pos = self.part_sequence_target[index]
        user_id = self.uid_list[index]

        target_neg = []
        seq_set = set(input_ids)
        for l in input_ids:
            id_neg = []
            for n in range(self.args.pre_train_sample):
                id_neg.append(neg_sample(seq_set, self.args.item_size))
            target_neg.append(id_neg)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [[0] * self.args.pre_train_sample] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]
        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return self.length

