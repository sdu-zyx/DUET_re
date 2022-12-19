# -*- coding: utf-8 -*-
import numpy as np
import tqdm
import torch
from sklearn.metrics import roc_auc_score

from utils import recall_at_k, ndcg_k, get_metric


class Trainer:
    def __init__(self, model, args):

        self.args = args
        self.model = model
        if self.model.cuda_condition:
            self.model.to(self.model.device)

    def get_sample_scores(self, epoch, pred_list):
        auc_pred = pred_list
        auc_labels = np.zeros_like(auc_pred)
        auc_labels[:, 0] = 1
        AUC = roc_auc_score(auc_labels.flatten(), pred_list.flatten())

        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
            "AUC": '{:.4f}'.format(AUC),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR, AUC], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.model.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class Pretrain_Train(Trainer):
    def __init__(self, model, args):
        super(Pretrain_Train, self).__init__(
            model,
            args
        )

    def train_stage(self, epoch, train_dataloader):

        desc = f'max_length-{self.args.max_seq_length}-' \
               f'hidden_size-{self.args.hidden_size}'

        train_data_iter = tqdm.tqdm(enumerate(train_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(train_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        joint_loss_avg = 0.0

        for i, batch in train_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.model.device) for t in batch)

            labels = [1] + [0] * self.args.base_train_sample

            pos_ids = batch[2].view(-1, self.args.max_seq_length, 1)
            neg_ids = batch[3].view(-1, self.args.max_seq_length, self.args.base_train_sample)
            target_sample = torch.cat((pos_ids, neg_ids), -1).view(-1, (self.args.base_train_sample+1))
            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]

            seq_output = self.model.pretrain_stage(batch).view(-1, self.args.hidden_size)
            total_score = self.model.classifier(seq_output)

            sample_score = total_score[np.arange(len(total_score))[:, None], target_sample]
            labels = torch.tensor(labels, dtype=torch.float).view(1, -1).repeat(len(total_score), 1).to(self.model.device)
            loss = self.model.criterion(sample_score, labels).sum(-1)
            loss = torch.sum(loss * istarget) / torch.sum(istarget)

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

            joint_loss_avg += loss.item()
        self.model.scheduler.step()
        post_fix = {
            "epoch": epoch,
            "joint_loss_avg": '{:.4f}'.format(joint_loss_avg / len(train_data_iter)),
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

    def eval_stage(self, epoch, dataloader, full_sort=False, test=True):

        str_code = "test" if test else "eval"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        self.model.eval()

        pred_list = None

        if full_sort:
            pass

        else:
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.model.device) for t in batch)
                user_ids, inputs, target_pos, target_neg, sample_negs = batch

                test_neg_items = torch.cat((target_pos[:, -1].view(-1, 1), sample_negs), -1)

                seq_output = self.model.base_stage(batch)[:, -1, :]
                recommend_output = self.model.classifier(seq_output)

                test_logits = recommend_output[np.arange(len(recommend_output))[:, None], test_neg_items]
                test_logits = test_logits.cpu().detach().numpy().copy()
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            return self.get_sample_scores(epoch, pred_list)


class Hyper_Train(Trainer):
    def __init__(self, model, args):
        super(Hyper_Train, self).__init__(
            model,
            args
        )

    def train_stage(self, epoch, train_dataloader):

        desc = f'max_length-{self.args.max_seq_length}-' \
               f'hidden_size-{self.args.hidden_size}'

        train_data_iter = tqdm.tqdm(enumerate(train_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(train_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        joint_loss_avg = 0.0

        for i, batch in train_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.model.device) for t in batch)

            labels = [1] + [0] * self.args.base_train_sample

            pos_ids = batch[2].view(-1, self.args.max_seq_length, 1)
            neg_ids = batch[3].view(-1, self.args.max_seq_length, self.args.base_train_sample)
            target_sample = torch.cat((pos_ids, neg_ids), -1).view(-1, (self.args.base_train_sample+1))
            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]

            total_score = self.model.hyper_stage(batch).view(-1, self.args.item_size)

            sample_score = total_score[np.arange(len(total_score))[:, None], target_sample]
            labels = torch.tensor(labels, dtype=torch.float).view(1, -1).repeat(len(total_score), 1).to(self.model.device)
            loss = self.model.criterion(sample_score, labels).sum(-1)
            loss = torch.sum(loss * istarget) / torch.sum(istarget)

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

            joint_loss_avg += loss.item()
        self.model.scheduler.step()
        post_fix = {
            "epoch": epoch,
            "joint_loss_avg": '{:.4f}'.format(joint_loss_avg / len(train_data_iter)),
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

    def eval_stage(self, epoch, dataloader, full_sort=False, test=True):

        str_code = "test" if test else "eval"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        self.model.eval()

        pred_list = None

        if full_sort:
            pass

        else:
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.model.device) for t in batch)
                user_ids, inputs, target_pos, target_neg, hyper_ids, sample_negs = batch

                test_neg_items = torch.cat((target_pos[:, -1].view(-1, 1), sample_negs), -1)

                recommend_output = self.model.hyper_stage(batch)[:, -1, :]

                test_logits = recommend_output[np.arange(len(recommend_output))[:, None], test_neg_items]
                test_logits = test_logits.cpu().detach().numpy().copy()
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            return self.get_sample_scores(epoch, pred_list)


class Base_Train(Trainer):
    # same as pretrain trainer
    # all data end to end train
    def __init__(self, model, args):
        super(Base_Train, self).__init__(
            model,
            args
        )

    def train_stage(self, epoch, train_dataloader):

        desc = f'max_length-{self.args.max_seq_length}-' \
               f'hidden_size-{self.args.hidden_size}'

        train_data_iter = tqdm.tqdm(enumerate(train_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(train_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        joint_loss_avg = 0.0

        for i, batch in train_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.model.device) for t in batch)

            labels = [1] + [0] * self.args.base_train_sample

            pos_ids = batch[2].view(-1, self.args.max_seq_length, 1)
            neg_ids = batch[3].view(-1, self.args.max_seq_length, self.args.base_train_sample)
            target_sample = torch.cat((pos_ids, neg_ids), -1).view(-1, (self.args.base_train_sample+1))
            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]

            seq_output = self.model.base_stage(batch).view(-1, self.args.hidden_size)
            total_score = self.model.classifier(seq_output)

            sample_score = total_score[np.arange(len(total_score))[:, None], target_sample]
            labels = torch.tensor(labels, dtype=torch.float).view(1, -1).repeat(len(total_score), 1).to(self.model.device)
            loss = self.model.criterion(sample_score, labels).sum(-1)
            loss = torch.sum(loss * istarget) / torch.sum(istarget)

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

            joint_loss_avg += loss.item()
        self.model.scheduler.step()
        post_fix = {
            "epoch": epoch,
            "joint_loss_avg": '{:.4f}'.format(joint_loss_avg / len(train_data_iter)),
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

    def eval_stage(self, epoch, dataloader, full_sort=False, test=True):

        str_code = "test" if test else "eval"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        self.model.eval()

        pred_list = None

        if full_sort:
            pass

        else:
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.model.device) for t in batch)
                user_ids, inputs, target_pos, target_neg, sample_negs = batch

                test_neg_items = torch.cat((target_pos[:, -1].view(-1, 1), sample_negs), -1)

                seq_output = self.model.base_stage(batch)[:, -1, :]
                recommend_output = self.model.classifier(seq_output)

                test_logits = recommend_output[np.arange(len(recommend_output))[:, None], test_neg_items]
                test_logits = test_logits.cpu().detach().numpy().copy()
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            return self.get_sample_scores(epoch, pred_list)


class BaseFT_Train(Trainer):
    # same as hyper trainer
    # pretrain model used for base ft and hyper ft
    # base ft without dynamic modeling and hyper network
    def __init__(self, model, args):
        super(BaseFT_Train, self).__init__(
            model,
            args
        )

    def train_stage(self, epoch, train_dataloader):

        desc = f'max_length-{self.args.max_seq_length}-' \
               f'hidden_size-{self.args.hidden_size}'

        train_data_iter = tqdm.tqdm(enumerate(train_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(train_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        joint_loss_avg = 0.0

        for i, batch in train_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.model.device) for t in batch)

            labels = [1] + [0] * self.args.base_train_sample

            pos_ids = batch[2].view(-1, self.args.max_seq_length, 1)
            neg_ids = batch[3].view(-1, self.args.max_seq_length, self.args.base_train_sample)
            target_sample = torch.cat((pos_ids, neg_ids), -1).view(-1, (self.args.base_train_sample+1))
            istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]

            seq_output = self.model.base_pretrain_ft_stage(batch).view(-1, self.args.hidden_size)
            total_score = self.model.classifier(seq_output)

            sample_score = total_score[np.arange(len(total_score))[:, None], target_sample]
            labels = torch.tensor(labels, dtype=torch.float).view(1, -1).repeat(len(total_score), 1).to(self.model.device)
            loss = self.model.criterion(sample_score, labels).sum(-1)
            loss = torch.sum(loss * istarget) / torch.sum(istarget)

            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

            joint_loss_avg += loss.item()
        self.model.scheduler.step()
        post_fix = {
            "epoch": epoch,
            "joint_loss_avg": '{:.4f}'.format(joint_loss_avg / len(train_data_iter)),
        }
        print(desc)
        print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

    def eval_stage(self, epoch, dataloader, full_sort=False, test=True):

        str_code = "test" if test else "eval"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        self.model.eval()

        pred_list = None

        if full_sort:
            pass

        else:
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                batch = tuple(t.to(self.model.device) for t in batch)
                user_ids, inputs, target_pos, target_neg, hyper_ids, sample_negs = batch

                test_neg_items = torch.cat((target_pos[:, -1].view(-1, 1), sample_negs), -1)

                seq_output = self.model.base_stage(batch)[:, -1, :]
                recommend_output = self.model.classifier(seq_output)

                test_logits = recommend_output[np.arange(len(recommend_output))[:, None], test_neg_items]
                test_logits = test_logits.cpu().detach().numpy().copy()
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            return self.get_sample_scores(epoch, pred_list)

