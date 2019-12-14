####pip install boto3

import boto3
import cv2
import json
import pickle
import os
import argparse


client = boto3.client('rekognition')
dict_keywords = {'sf': ['stale', 'state farm', 'state far' 'state', 'farm', 'statefarm', 'statefarn'],
                'rap': ['aptors.com', 'raptors.com', '.com', 'com', 'co'],
                'cel': ["oiic", "olic", "elric", "elic", "@celtics", "@celtic", "@celti", "aceltic", "aceltics",
                        "eceltic", "eceltics", "celric", "Oceltics", "Ocelrics", "celticscom", "celtics-com",
                        "celticsco", "celtics-con", "celtic", "celic"], 'fh': ["fieldhouse", "steak n shake"],
                 'gei': ["geico", "gaico", "gacd"], 'av': ["aviva"], 'am': ["america fii", "america"],
                 'lak': ["aker", "lake", "laker", "lakers", "lakers.com", "akers.com"], 'ar': ["arby"]}


def get_desired_text_bbox(img, client, desired_text_list):
    converted = cv2.imencode('.jpg', img)
    ll1 = converted[1].tostring()
    dct = client.detect_text(Image={'Bytes': ll1})
    lines = {}
    for dd in dct['TextDetections']:
        if 'WORD' in dd['Type']:
            lines[dd['DetectedText']] = dd['Geometry']['BoundingBox']
    # print(lines.keys())
    fresh_d = None
    for key in lines:
        print("----" + key)
        if key.lower() in desired_text_list or any([kk in key.lower() for kk in desired_text_list]):
            print(key)
            fresh_d = lines[key]
            break
    return fresh_d


def get_hoop(ll):
    if ll['Left'] >= 0.5:
        ## why 108? .. measured in a particular video and came to be so
        top_right = (int(ll['Left'] * 1920 - 108), int(ll['Top'] * 1080))
        bottom_left = (top_right[0] - 70, top_right[1] + 50)
    else:
        ## 108 on left side is too much, may be the camera is further away... use
        top_right = (ll['Left'] * 1920 + ll['Width'] * 1920 + 108 - 50 + 70, ll['Top'] * 1080 + ll['Height'] * 1080)
        bottom_left = (top_right[0] - 70, top_right[1] + 50)
    return bottom_left, top_right


def draw(img, bottom_left, top_right, img_name):
    color = (0, 255, 255)
    thickness = 10
    img_n = cv2.rectangle(img.copy(), (int(bottom_left[0]), int(bottom_left[1])),
                          (int(top_right[0]), int(top_right[1])), color, thickness)
    cv2.imwrite(img_name, img_n)


def process():
    buckets_d = {}

    for keyv in video_paths_vs_keyword:
        video_path = base_path + prefix_path + keyv + ".mp4"
        cap = cv2.VideoCapture(video_path)
        json_paths = [f for f in os.listdir(base_json_path) if
                     os.path.isfile(os.path.join(base_json_path, f)) and 'json' in f and not 'pkl' in f and prefix_path + keyv + "_" in f]
        for json_p in json_paths:
            fp = open(base_json_path + json_p)
            print("~~~~~~~" + json_p)
            jj = json.loads(fp.read())
            frame_numbers = []
            for key in jj:
                frame_numbers.append(int(key))
            i = 0
            important_frames = {}
            ret = True
            cap = cv2.VideoCapture(video_path)
            while ret and len(frame_numbers) > len(important_frames.keys()):
                ret, frame = cap.read()
                if i in frame_numbers:
                    important_frames[i] = frame
                i = i + 1
            cap.release()
            print("parsed " + video_path)
            ## frame data finally captures any clip's frame wise basket positions, topRight, and the bottomLeft
            frame_data = {}
            for frame_no in important_frames:
                print("@@@@@~~~~~" + str(video_paths_vs_keyword[keyv]))
                ll = get_desired_text_bbox(important_frames[frame_no], client, dict_keywords[video_paths_vs_keyword[keyv]])
                if ll is not None:
                    bottom_left, top_right = get_hoop(ll)
                    frame_data[frame_no] = (bottom_left, top_right)
            buckets_d[json_p] = frame_data
            print(frame_data)
            with open(base_json_path + json_p + '.pkl', 'wb') as handle:
                pickle.dump(frame_data, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/tmp/', help='base path')
    parser.add_argument('--prefix', type=str, default='output.', help='file prefix')

    args = parser.parse_args()

    base_path = args.base_path
    prefix_path = args.prefix

    base_json_path = base_path + "features/"
    video_paths_vs_keyword = {'9': 'sf'}

    process()

