import pickle
import numpy as np


def load_dataset():
    with open('dunk-3-ptr.pkl', 'rb') as handle:
        b = pickle.load(handle)

    all_dunks = []
    for key in b['dunk']:
        for fv in b['dunk'][key][0]:
            all_dunks.append(fv)

    all_three = []
    for key in b['three']:
        for fv in b['three'][key][0]:
            all_three.append(fv)

    number_of_dunk_examples = len(all_dunks)
    number_of_three_pointer_examples = len(all_three)

    label_dunks = []
    for i in range(number_of_dunk_examples):
        label_dunks.append([1, 0])

    label_three = []
    for i in range(number_of_three_pointer_examples):
        label_three.append([0, 1])

    all_fv = []
    test_fv = []
    all_labels = []
    test_labels = []

    for i in range(len(all_dunks)):
        if i < len(all_dunks) * 0.8:
            all_fv.append(all_dunks[i])
        else:
            test_fv.append(all_dunks[i])

    for j in range(len(label_dunks)):
        if j < len(label_dunks) * 0.8:
            all_labels.append(label_dunks[j])
        else:
            test_labels.append(label_dunks[j])

    for k in range(len(all_three)):
        if k < len(all_three) * 0.8:
            all_fv.append(all_three[k])
        else:
            test_fv.append(all_three[k])

    for l in range(len(label_three)):
        if l < len(label_three) * 0.8:
            all_labels.append(label_three[l])
        else:
            test_labels.append(label_three[l])

    train_fv = np.stack(all_fv)
    train_labels = np.stack(all_labels)

    test_fv = np.stack(test_fv)
    test_labels = np.stack(test_labels)

    return train_fv, train_labels, test_fv, test_labels


def train():
    from keras.layers import Input, Dense, merge, Flatten, Dropout, Embedding, Lambda, Concatenate, Reshape, concatenate
    from keras.models import Model
    from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, Flatten, GlobalMaxPooling1D, Concatenate, Flatten

    n_classes = 2
    no_of_filters = 4
    frame_vector_size = 2
    inputs = Input(shape=(None, frame_vector_size))
    convs = []

    filter_sizes = [1, 2]
    for filterSize in filter_sizes:
        conv_layer = Conv1D(padding="same", filters=no_of_filters, activation="tanh", kernel_size=1,
                            name='conv1d_' + str(filterSize))(inputs)
        max_pool_layer = GlobalMaxPooling1D(name='globalMaxPool_' + str(filterSize))(conv_layer)
        convs.append(max_pool_layer)

    l_merge = Concatenate()(convs)
    l_drop = Dropout(0.37)(l_merge)
    dense_layer = Dense(n_classes, activation="sigmoid")(l_drop)
    model = Model(inputs=inputs, outputs=dense_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()

    hist = model.fit(train_fv, train_labels, validation_data=(test_fv, test_labels), shuffle=True, epochs=1000)
    model.save(model_file)
    with open(history_file, 'wb') as handle:
        pickle.dump(hist, handle)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dunk-3-ptr.pkl', help='dataset file')
    parser.add_argument('--model_file', type=str, default='mm5.h5', help='file to save model')
    parser.add_argument('--history_file', type=str, default='mm5.h5-training-history.pkl',
                        help='pkl file to save history')

    args = parser.parse_args()
    model_file = args.model_file
    history_file = args.history_file
    dataset = args.dataset

    train_fv, train_labels, test_fv, test_labels = load_dataset()
    train()

