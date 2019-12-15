import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

from model.svm_truth_values import jump_truth, truth_labels

class_labels = ['reserved', 'dunk', 'three-pointer']
label_str_to_id = {}
for i, label in enumerate(class_labels):
    label_str_to_id[label] = i


def get_class_label(filename):
    for label_str in class_labels:
        if label_str in filename:
            return label_str_to_id[label_str]
    return -1


def basket_position_unknown(jump):
    return np.array_equal(jump['player_position'], jump['shifted_position'])


def is_left_basket(jump):
    if basket_position_unknown(jump):
        net_movement = jump['previous_frame_ids'][-1]['player_position'][0] \
                       - jump['previous_frame_ids'][0]['player_position'][0]
        return net_movement < 0
    else:
        return jump['basket'][0] <= 0.9 * (im_width / 2)


def add_jumps(jumps, filename, dataset):
    label = get_class_label(filename)
    for jump in jumps:
        if not basket_position_unknown(jump):
            position = jump['previous_frame_ids'][-1]['shifted_position']
            left_basket = is_left_basket(jump)
            if not left_basket:
                position[0] = im_width - position[0]
            # if basket_position_unknown(jump):
            # position[0] = position[0] - basket_hspace - int(basket_width/2)
            dataset.append((position[0], position[1], get_class_label(filename), len(jumps)))


def plot_data(dataset):
    """
    color and symbol coding
    red - dunk, green - three pointer
    circle - single jump videos
    X - double jump videos
    = - triple jump videos
    triangle - four or more jump videos

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """

    # Plot dataset
    x = np.array(dataset)
    plt.figure()
    dunks = x[x[:, 2] == label_str_to_id['dunk']]
    threes = x[x[:, 2] == label_str_to_id['three-pointer']]

    plt.plot(dunks[dunks[:, 3] == 1, 0], dunks[dunks[:, 3] == 1, 1], 'rx', linewidth=2)
    plt.plot(dunks[dunks[:, 3] == 2, 0], dunks[dunks[:, 3] == 2, 1], 'rx', linewidth=2)
    plt.plot(dunks[dunks[:, 3] > 2, 0], dunks[dunks[:, 3] > 2, 1], 'rx', linewidth=2)

    plt.plot(threes[threes[:, 3] == 1, 0], threes[threes[:, 3] == 1, 1], 'gx', linewidth=2)
    plt.plot(threes[threes[:, 3] == 2, 0], threes[threes[:, 3] == 2, 1], 'gx', linewidth=2)
    plt.plot(threes[threes[:, 3] > 2, 0], threes[threes[:, 3] > 2, 1], 'gx', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-200, 50)
    plt.xlim(-100, 100)
    plt.show()


def process():
    import glob
    dataset = list()
    for filename in glob.glob(jumps_dir + "/*_jumps.pkl"):
        jumps, player_travels, player_positions, shifted_positions = pickle.load(open(filename, "rb"))
        add_jumps(jumps, filename, dataset)

    plot_data(dataset)
    plot_data(truth_labels)
    plot_data(jump_truth)


def plot_keras_history(history):
    plt.figure()

    plt.plot(history.epoch, history.history['loss'], 'r', linewidth=1)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jumps_dir', type=str, required=True, help='jumps pkl directory')

    args = parser.parse_args()
    jumps_dir = args.jumps_dir
    three_pointer_jumps_file = jumps_dir + "/jumps_three-pointer.pkl"
    dunks_jumps_file = jumps_dir + "/jumps_dunks.pkl"
    im_width = 1920
    basket_width = 70
    basket_hspace = 140

    plot_keras_history(pickle.load(open("../model/mm5.h5-training-history.pkl", "rb")))
    process()
