import pickle
from collections import defaultdict
import numpy as np

usernum = 0
itemnum = 0
User = defaultdict(list)

total_interaction = 0
# assume user/item index starting from 1
f = open('../data/ml-1m/ml-1m.inter', 'r')
for line in f:
    if 'user_id' not in line:
        try:
            u, i, r, t = line.rstrip().split('	')
        except:
            continue
        u_id = int(u)
        i_id = int(i)
        usernum = max(usernum, u_id)
        itemnum = max(itemnum, i_id)
        User[u].append((i, t))
        total_interaction += 1

for user, item_time in User.items():
    item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
    items = []
    for t in item_time:
        items.append(t[0])
    User[user] = items
print(usernum, itemnum, total_interaction, User['6040'])

with open('../data/ml-1m/duet_s2s/ml-1m.txt', 'w') as out:
    for user, items in User.items():
        out.write(user + ' ' + ','.join(items) + '\n')

pre_va_labs = []
pre_va_seq = []
pre_va_user = []
pre_va_len = []

pre_tr_labs = []
pre_tr_seq = []
pre_tr_user = []
pre_tr_len = []

hyper_tr_labs = []
hyper_tr_seq = []
hyper_tr_user = []
hyper_tr_len = []
hyper_tr_dynamic = []

hyper_va_labs = []
hyper_va_seq = []
hyper_va_user = []
hyper_va_len = []
hyper_va_dynamic = []

hyper_te_labs = []
hyper_te_seq = []
hyper_te_user = []
hyper_te_len = []
hyper_te_dynamic = []

for user, items in User.items():
    items = [int(item) for item in items]
    item_len = len(items)
    pre_data_len = int(item_len * 0.4)
    hyper_data_len = item_len - pre_data_len

    pre_data = items[:pre_data_len]
    hyper_data = items[pre_data_len:]

    for i in range(1, len(pre_data), 30):
        if i == 1:
            pre_tr_seq += [pre_data[-(30+i):-i]]
            pre_tr_labs += [pre_data[-30:]]
        else:
            pre_tr_seq += [pre_data[-(30+i):-i]]
            pre_tr_labs += [pre_data[-(30+i)+1:-i+1]]
        pre_tr_user += [int(user)]
        pre_tr_len += [len(pre_tr_seq)]

    pre_va_labs +=  [pre_data[-29:] + [hyper_data[0]]]
    pre_va_seq += [pre_data[-30:]]
    pre_va_user += [int(user)]
    pre_va_len += [len(pre_va_seq)]

    hyper_train_seq = hyper_data[:-2]
    for i in range(1, len(hyper_train_seq)+1, 30):
        if i == 1:
            hyper_tr_seq += [pre_data + hyper_train_seq[:-i]][-30:]
            hyper_tr_labs += [pre_data + hyper_train_seq][-30:]
        else:
            hyper_tr_seq += [pre_data + hyper_train_seq[:-i]][-30:]
            hyper_tr_labs += [pre_data + hyper_train_seq[:-i+1]][-30:]
        hyper_tr_user += [int(user)]
        hyper_tr_len += [len(hyper_tr_seq)]

    hyper_va_seq += [pre_data + hyper_data[:-2]][-30:]
    hyper_va_labs += [pre_data + hyper_data[:-1]][-30:]
    hyper_va_user += [int(user)]
    hyper_va_len += [len(hyper_va_seq)]

    hyper_te_seq += [pre_data + hyper_data[:-1]][-30:]
    hyper_te_labs += [pre_data + hyper_data][-30:]
    hyper_te_user += [int(user)]
    hyper_te_len += [len(hyper_te_seq)]

# print(hyper_tr_dynamic[:1], hyper_tr_seq[:1], hyper_tr_labs[:1])
# print(hyper_te_dynamic[:1], hyper_te_seq[:1], hyper_te_labs[:1])
# print(hyper_va_dynamic[:1], hyper_va_seq[:1], hyper_va_labs[:1])
pre_tra = (pre_tr_user, pre_tr_seq, pre_tr_labs, pre_tr_len)
pre_val = (pre_va_user, pre_va_seq, pre_va_labs, pre_va_len)
hyper_tra = (hyper_tr_user, hyper_tr_seq, hyper_tr_labs, hyper_tr_len)
hyper_val = (hyper_va_user, hyper_va_seq, hyper_va_labs, hyper_va_len)
hyper_tes = (hyper_te_user, hyper_te_seq, hyper_te_labs, hyper_te_len)
pickle.dump(pre_tra, open('../data/ml-1m/duet_s2s/pre_train.pkl', 'wb'))
pickle.dump(pre_val, open('../data/ml-1m/duet_s2s/pre_valid.pkl', 'wb'))
pickle.dump(hyper_tra, open('../data/ml-1m/duet_s2s/hyper_train.pkl', 'wb'))
pickle.dump(hyper_val, open('../data/ml-1m/duet_s2s/hyper_valid.pkl', 'wb'))
pickle.dump(hyper_tes, open('../data/ml-1m/duet_s2s/hyper_test.pkl', 'wb'))

base_tr_labs = []
base_tr_seq = []
base_tr_user = []
base_tr_len = []

base_va_labs =[]
base_va_seq = []
base_va_user = []
base_va_len = []

base_te_labs = []
base_te_seq = []
base_te_user = []
base_te_len = []

for user, items in User.items():
    items = [int(item) for item in items]

    train_seq = items[:-2]
    for i in range(1, len(train_seq), 30):
        if i == 1:
            base_tr_seq += [train_seq[-(30+i):-i]]
            base_tr_labs += [train_seq[-30:]]
        else:
            base_tr_seq += [train_seq[-(30+i):-i]]
            base_tr_labs += [train_seq[-(30+i)+1:-i+1]]
        base_tr_user += [int(user)]
        base_tr_len += [len(base_tr_seq)]

    base_va_labs += [items[:-1]][-30:]
    base_va_seq += [items[:-2]][-30:]
    base_va_user += [int(user)]
    base_va_len += [len(base_va_labs)]

    base_te_labs += [items][-30:]
    base_te_seq += [items[:-1]][-30:]
    base_te_user += [int(user)]
    base_te_len += [len(base_te_labs)]

base_tra = (base_tr_user, base_tr_seq, base_tr_labs, base_tr_len)
base_val = (base_va_user, base_va_seq, base_va_labs, base_va_len)
base_tes = (base_te_user, base_te_seq, base_te_labs, base_te_len)
pickle.dump(base_tra, open('../data/ml-1m/duet_s2s/base_train.pkl', 'wb'))
pickle.dump(base_val, open('../data/ml-1m/duet_s2s/base_valid.pkl', 'wb'))
pickle.dump(base_tes, open('../data/ml-1m/duet_s2s/base_test.pkl', 'wb'))


np.random.seed(12345)

def sample_test_data(data_name, test_num=99, sample_type='random'):
    """
    sample_type:
        random:  sample `test_num` negative items randomly.
        pop: sample `test_num` negative items according to item popularity.
    """

    data_file = f'../data/ml-1m/duet_s2s/{data_name}.txt'
    test_file = f'../data/ml-1m/duet_s2s/{data_name}_sample.txt'

    item_count = defaultdict(int)
    user_items = defaultdict()

    lines = open(data_file).readlines()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(',')
        items = [int(item) for item in items]
        user_items[user] = items
        for item in items:
            item_count[item] += 1

    all_item = list(item_count.keys())
    count = list(item_count.values())
    sum_value = np.sum([x for x in count])
    probability = [value / sum_value for value in count]

    user_neg_items = defaultdict()

    for user, user_seq in user_items.items():
        test_samples = []
        while len(test_samples) < test_num:
            if sample_type == 'random':
                sample_ids = np.random.choice(all_item, test_num, replace=False)
            else: # sample_type == 'pop':
                sample_ids = np.random.choice(all_item, test_num, replace=False, p=probability)
            sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in test_samples]
            test_samples.extend(sample_ids)
        test_samples = test_samples[:test_num]
        user_neg_items[user] = test_samples

    with open(test_file, 'w') as out:
        for user, samples in user_neg_items.items():
            out.write(user+' '+','.join(samples)+'\n')

sample_test_data('ml-1m')

