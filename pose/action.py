import os
import sys
import argparse
import cv2
import time
import pickle
import numpy as np
import math
import json
from config_reader import config_reader
import pose_opensource.util as util

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)

bp_id_to_str = ["nose", "neck", "rshoulder", "relbow", "rwrist", "lshoulder", "lelbow", "lwrist", "rhip", "rknee",
                "rankle", "lhip", "lknee", "lankle", "reye", "leye", "rear", "lear"]
bp_str_to_id = {}
for i, part in enumerate(bp_id_to_str):
    bp_str_to_id[part] = i

limb_id_to_str = ["rshoulder", "lshoulder", "rupperarm", "rlowerarm", "lupperarm", "llowerarm", "rhipneck",
                  "rupperleg", "rlowerleg", "lhipneck", "lupperleg", "llowerleg", "noseneck", "reyenose", "reyeear",
                  "leyenose", "leyeear"]
limb_str_to_id = {}
for i, limb in enumerate(limb_id_to_str):
    limb_str_to_id[limb] = i


def generate_model_blobs(in_video_file, starting_frame, ending_frame):
    from pose_opensource.model.cmu_model import get_testing_model
    from pose.processing_action import get_model_blob
    # load model
    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # Video reader
    cam = cv2.VideoCapture(in_video_file)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if ending_frame is None:
        ending_frame = video_length

    scale_search = [1, .5, 1.5, 2]  # [.5, 1, 1.5, 2]
    scale_search = scale_search[0:process_speed]
    params['scale_search'] = scale_search

    if ending_frame is None:
        ending_frame = video_length

    i = 0
    if starting_frame > 0:
        while (cam.isOpened()) and ret_val is True and i < starting_frame:
            ret_val, orig_image = cam.read()
            i += 1

    blobs = {}
    while (cam.isOpened()) and ret_val is True and i < ending_frame:
        if i % frame_rate_ratio == 0:
            input_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

            tic = time.time()
            blobs[i] = get_model_blob(input_image, params, model, model_params)
            toc = time.time()
            print(f"Generate model blob, frame: {i}, took {toc - tic} seconds")

        ret_val, orig_image = cam.read()
        i += 1
    return blobs


def generate_pose(blobs):
    from pose.processing_action import extract_parts
    pose = {}
    for frame in blobs:
        tic = time.time()
        pose[frame] = extract_parts(blobs[frame], params, model_params)
        toc = time.time()
        print(f"Generate pose, frame: {frame}, took {toc - tic} seconds")
    return pose


def save_original_video(in_video_file, out_video_file, ending_frame):
    # Video reader
    cam = cv2.VideoCapture(in_video_file)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if ending_frame is None:
        ending_frame = video_length

    # Video writer
    output_fps = input_fps / frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_file, fourcc, output_fps, (orig_image.shape[1], orig_image.shape[0]))

    i = 0
    if starting_frame > 0:
        while (cam.isOpened()) and ret_val is True and i < starting_frame:
            ret_val, orig_image = cam.read()
            i += 1

    while (cam.isOpened()) and ret_val is True and i < ending_frame:
        if i % frame_rate_ratio == 0:
            out.write(orig_image)
        ret_val, orig_image = cam.read()
        i += 1
    out.release()


def save_output_video(in_video_file, out_video_file, pose, ending_frame):
    from pose.processing_action import draw

    # Video reader
    cam = cv2.VideoCapture(in_video_file)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if ending_frame is None:
        ending_frame = video_length

    # Video writer
    output_fps = input_fps / frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_file, fourcc, output_fps, (orig_image.shape[1], orig_image.shape[0]))

    i = 0
    while (cam.isOpened()) and ret_val is True and i < ending_frame:
        if i in pose:
            tic = time.time()
            canvas = draw(orig_image, pose[i][0], pose[i][1], pose[i][2])
            out.write(canvas)
            toc = time.time()
            print(f"Save output video, frame: {i}, took {toc - tic} seconds")

        ret_val, orig_image = cam.read()
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='input video file name')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')
    parser.add_argument('--frame_ratio', type=int, default=1, help='analyze every [n] frames')
    parser.add_argument('--process_speed', type=int, default=4,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--end', type=int, default=None, help='Last video frame to analyze')
    parser.add_argument('--start', type=int, default=None, help='First video frame to analyze')
    parser.add_argument('--output_file_prefix', type=str, default=None, help='extra label in filename')
    parser.add_argument('--generate_model_blobs', action='store_true', help='generate model blobs on gpu')
    parser.add_argument('--generate_pose_from_blob', action='store_true', help='generate pose from model blobs')
    parser.add_argument('--save_output_video', action='store_true', help='draw pose on input video file')
    parser.add_argument('--save_original_video', action='store_true', help='save original video segment')
    parser.add_argument('--generate_pose', action='store_true', help='generate pose from video file')
    parser.add_argument('--generate_features', type=str, help='generate custom features based on pose pkl')

    args = parser.parse_args()
    keras_weights_file = args.model
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    ending_frame = args.end
    starting_frame = args.start
    output_file_prefix = ""

    if args.output_file_prefix is not None:
        output_file_prefix = "_" + args.output_file_prefix

    # load config
    params, model_params = config_reader()
    blobs_file = args.video.rsplit(".", 1)[0] + output_file_prefix + "_blobs.pkl"
    pose_file = args.video.rsplit(".", 1)[0] + output_file_prefix + "_pose.pkl"
    output_file = args.video.rsplit(".", 1)[0] + output_file_prefix + "_output.mp4"
    original_file = args.video.rsplit(".", 1)[0] + output_file_prefix + "_original.mp4"

    if starting_frame is None:
        starting_frame = 0

    print('start processing...')
    if args.generate_model_blobs:
        model_blobs = generate_model_blobs(args.video, starting_frame, ending_frame)
        pickle.dump(model_blobs, open(blobs_file, "wb"))

    if args.generate_pose_from_blob:
        model_blobs = pickle.load(open(blobs_file, "rb"))
        pose = generate_pose(model_blobs)
        pickle.dump(pose, open(pose_file, "wb"))

    if args.save_output_video:
        pose = pickle.load(open(pose_file, "rb"))
        save_output_video(args.video, output_file, pose, ending_frame)

    if args.generate_pose:
        model_blobs = generate_model_blobs(args.video, starting_frame, ending_frame)
        pose = generate_pose(model_blobs)
        pickle.dump(pose, open(pose_file, "wb"))

    if args.save_original_video:
        save_original_video(args.video, original_file, ending_frame)

    if args.generate_features is not None:
        pose = pickle.load(open(args.generate_features, "rb"))

        video_positions = {}
        for frame in pose:
            all_peaks, subset, candidate = pose[frame]

            body_parts = {}  # body part id -> list of all instances (across all people)
            for i in range(len(bp_id_to_str)):
                key = bp_id_to_str[i]
                body_parts[key] = list()
                for peak in all_peaks[i]:
                    body_parts[key].append({"x": int(peak[0]), "y": int(peak[1]), "s": peak[2]})

            people = list()  # list of persons. person is a list of LimbLocations
            for person_id in range(len(subset)):
                person = {}
                for limb in range(len(limb_id_to_str)):
                    key = limb_id_to_str[limb]
                    index = subset[person_id][np.array(util.limbSeq[limb])-1]
                    if -1 in index:
                        continue
                    X = candidate[index.astype(int), 0]
                    Y = candidate[index.astype(int), 1]
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    person[key] = {"from": (int(X[0]), int(Y[0])), "to": (int(X[1]), int(Y[1])), "len": float(length), "ang": float(angle), "s": 1}
                people.append(person)

            video_positions[frame] = {"body_parts": body_parts, "people": people}

        feature_file = args.generate_features.rsplit("_", 1)[0] + "_features"
        pickle.dump(video_positions, open(feature_file + ".pkl", "wb"))
        json.dump(video_positions, open(feature_file + ".json", "w"), sort_keys=True, indent=4, separators=(',', ': '))



