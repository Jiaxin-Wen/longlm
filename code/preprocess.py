from tqdm import tqdm
import random
import os



def preprocess(in_path, output_path, split=True):
    def dump_data(split, a):
        source, target = [], []
        for i in a:
            source.append(i[0])
            target.append(i[1])
        with open(f'{output_path}/{split}.source', 'w') as f:
            for i in source:
                f.write(i)
                f.write('\n')
        with open(f'{output_path}/{split}.target', 'w') as f:
            for i in target:
                f.write(i)
                f.write('\n')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(in_path, 'r') as f:
        lines = f.readlines()
    total = len(lines)
    print(f'file = {in_path}, size = {total}, output_path = {output_path}')
    data = []
    for line in tqdm(lines):
        line = line.strip().split('\t\t')
        context = '<sep>'.join(line[:-1]) + '<sep><extra_id_1>'
        response = '<extra_id_1>' + line[-1]
        data.append((context, response))

    if split:
        # split
        random.shuffle(data)
        train_size = int(total * 0.7)
        test_size = int(total * 0.15)
        train_data = data[: train_size]
        val_data = data[train_size: -test_size]
        test_data = data[-test_size: ]
        
        dump_data('train', train_data)
        dump_data('val', val_data)
        dump_data('test', test_data)
    else:
        dump_data('test', data)



if __name__ == "__main__":
    random.seed(22)

    # files = [
    #     # ('/dataset/f1d6ea5b/gyx-eva/data/multi/train.txt', '/dataset/f1d6ea5b/wenjiaxin/lot/1.4G'),
    #     # ('/dataset/f1d6ea5b/sunhao/merge_shuffled_multi_0925_4.txt', '/dataset/f1d6ea5b/wenjiaxin/lot/merge_shuffled_multi_0925_4'),
    #     ('/dataset/f1d6ea5b/sunhao/filter/merge_shuffled_0907.txt', '/dataset/f1d6ea5b/wenjiaxin/lot/12G')
    # ]

    # for i in files:
    #     preprocess(i[0], i[1])

    preprocess('/dataset/f1d6ea5b/sunhao/EVA_test_0827/test_multi.txt', 'eva_test', split=False)