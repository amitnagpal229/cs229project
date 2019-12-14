import os
import argparse
import pickle
import numpy as np

from pose.action import bp_id_to_str
from pose.action import bp_str_to_id
from pose.action import limb_id_to_str
from pose.action import limb_str_to_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, required=True, help='directory containing features pkl files')
    parser.add_argument('--generate_features', type=str, help='directory containing features pkl files')

    args = parser.parse_args()
    file_dir = args.file_dir

    persons = 0
    frames = 0
    parts = np.zeros((len(bp_id_to_str),))
    limbs = np.zeros((len(limb_id_to_str),))
    files = os.listdir(file_dir)
    count = 0
    for file in files:
        count += 1
        print(f"{count}) {file}")
        features = pickle.load(open(file_dir + file, "rb"))
        for frame in features:
            frames += 1
            body_parts = features[frame]["body_parts"]
            people = features[frame]["people"]
            for part in body_parts:
                parts[bp_str_to_id[part]] += len(body_parts[part])
            for person in people:
                persons += 1
                for limb in person:
                    limbs[limb_str_to_id[limb]] += 1

    def get_counts(plist):
        count = 0
        for p in plist:
            count += parts[bp_str_to_id[p]]
        return count / frames, count / persons

    def get_limb_counts(plist):
        count = 0
        for p in plist:
            count += limbs[limb_str_to_id[p]]
        return count / frames, count / persons

    print(f"persons={persons}, frames={frames}, persons/frame={persons/frames}")
    print(f'knee: {get_counts(("lknee", "rknee"))}')
    print(f'wrist: {get_counts(("lwrist", "rwrist"))}')
    print(f'ankle: {get_counts(("lankle", "rankle"))}')
    print(f'elbow: {get_counts(("lelbow", "relbow"))}')
    print(f'neck: {get_counts(["neck"])}')
    print(f'shoulder: {get_limb_counts(("lshoulder", "rshoulder"))}')
    print(f'upper-leg: {get_limb_counts(("lupperleg", "rupperleg"))}')
    print(f'upper-arm: {get_limb_counts(("lupperarm", "rupperarm"))}')
    print(f'lower-leg: {get_limb_counts(("llowerleg", "rlowerleg"))}')
    print(f'lower-arm: {get_limb_counts(("llowerarm", "rlowerarm"))}')
