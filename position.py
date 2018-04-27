pos_labels = ['LEFT', 'ON', 'RIGHT']
rewards = [-1, 1, -1]
pos_label_dict = {l: i for i, l in enumerate(pos_labels)}
POS_NUM = len(pos_labels)


def get_index(label):
    return pos_label_dict[label]


def get_label(index):
    return pos_labels[index]


def compute_reward(pos_idx):
    return rewards[pos_idx]


if __name__ == '__main__':
    print(get_index('ON'))
    print(get_index('LEFT'))
    print(get_index('RIGHT'))

    print(get_label(0))
    print(get_label(1))
    print(get_label(2))
