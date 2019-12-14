import argparse
import cv2
import time
import pickle
import numpy as np
import math
import json

from action import limb_str_to_id, limb_id_to_str
from util import colors

pose_array_length = len(limb_id_to_str)*7  # number of pose limbs by 7 numbers per limb (x0, y1, x1, y1, len, angle, vel)
gray = [128, 128, 128]
COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

def init_video():
    # Video reader
    cam_in = cv2.VideoCapture(args.video)
    im_width = int(cam_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cam_in.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    cam_out = cv2.VideoWriter(out_prefix+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), args.out_fps, (im_width, im_height))
    return cam_in, cam_out, im_width, im_height


def get_orig_image(previous_frame_id, frame_id):
    for i in range(previous_frame_id + 1, frame_id):
        cam_in.read()

    return cam_in.read()


def get_track_weight_matrix(bbox):
    matrix = np.zeros((im_height, im_width))
    x0, y0, x1, y1 = bbox[:4]
    matrix[y0:y1, x0:x1] = 1
    return matrix


def get_limb_polygon(limb):
    x0, y0 = limb['from']
    x1, y1 = limb['to']
    m_x = np.mean([x0, x1])
    m_y = np.mean([y0, y1])
    length = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
    angle = math.degrees(math.atan2(y0 - y1, x0 - x1))
    return cv2.ellipse2Poly((int(m_x), int(m_y)), (int(length / 2), 4), int(angle), 0, 360, 1)


def get_person_pose_weight_matrix(person_pose):
    matrix = np.zeros((im_height, im_width))
    for limb in person_pose:
        polygon = get_limb_polygon(person_pose[limb])
        for point in polygon:
            if point[0] < im_width and point[1] < im_height:
                matrix[point[1]][point[0]] = 1
    return matrix


def get_pose_limits_xyxy(person_pose):
    x0, y0, x1, y1 = (im_width-1, im_height-1, 0, 0)
    for limb in person_pose.values():
        x0 = min(x0, limb['from'][0], limb['to'][0])
        x1 = max(x1, limb['from'][0], limb['to'][0])
        y0 = min(y0, limb['from'][1], limb['to'][1])
        y1 = max(y1, limb['from'][1], limb['to'][1])
    return x0, y0, x1, y1


def match_tracks_poses(bboxes, people_poses):
    if len(bboxes) == 0 or len(people_poses) == 0:
        return {}

    match_matrix = np.zeros((len(bboxes), len(people_poses)))
    tracks = list(bboxes.keys())

    for track, track_id in enumerate(tracks):
        bbox = bboxes[track_id]
        x0, y0, x1, y1 = bbox[:4]
        for person_id, person_pose in enumerate(people_poses):
            px0, py0, px1, py1 = get_pose_limits_xyxy(person_pose)
            if not (x1 < px0 or x0 > px1 or y1 < py0 or y0 > py1):
                person_pose_weight_matrix = get_person_pose_weight_matrix(person_pose)
                track_weight_matrix = get_track_weight_matrix(bbox)
                match_matrix[track][person_id] = np.sum(np.multiply(person_pose_weight_matrix, track_weight_matrix))

    tracks_pose = {}
    index = np.unravel_index(np.argmax(match_matrix), match_matrix.shape)
    while match_matrix[index] > 0:
        tracks_pose[tracks[int(index[0])]] = int(index[1])
        match_matrix[index[0], :] = 0
        match_matrix[:, index[1]] = 0
        index = np.unravel_index(np.argmax(match_matrix), match_matrix.shape)

    return tracks_pose


def get_pose_array(pose):
    pose_array = np.zeros(pose_array_length)
    for limb in pose:
        index = limb_str_to_id[limb] * 7
        limb_position = pose[limb]
        pose_array[index] = limb_position['from'][0]
        pose_array[index + 1] = limb_position['from'][1]
        pose_array[index + 2] = limb_position['to'][0]
        pose_array[index + 3] = limb_position['to'][1]
        pose_array[index + 4] = limb_position['len']
        pose_array[index + 5] = limb_position['ang']
        pose_array[index + 6] = 0  # future for velocity
    return pose_array


def draw_pose(input_image, pose, color=None):
    canvas = input_image.copy()

    for limb in pose:
        polygon = get_limb_polygon(pose[limb])
        if color is None:
            color = colors[limb_str_to_id[limb]]
        cur_canvas = canvas.copy()
        cv2.fillConvexPoly(cur_canvas, polygon, color)
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas


def draw_bbox(input_image, track_id, bbox, color=None):
    x1, y1, x2, y2 = bbox[:4]
    if color is None:
        color = COLORS_10[track_id % len(COLORS_10)]
    label = '{}{:d}'.format("", track_id)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(input_image, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(input_image, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
    cv2.putText(input_image, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return input_image


def draw_frame_num(input_image, frame_id, original_frame_id):
    label = '{}{:d}/{:d}'.format("", frame_id, original_frame_id)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(input_image, (1750, 90), (1750 + t_size[0] + 3, 90 + t_size[1] + 4), [0, 0, 0], -1)
    cv2.putText(input_image, label, (1750, 90 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return input_image


def filter_big_bboxes(bboxes):
    for track_id in list(bboxes.keys()):
        x1, y1, x2, y2 = bboxes[track_id][:4]
        if abs((x1-x2)*(y1-y2)) > 0.1 * im_area:
            del bboxes[track_id]
    return bboxes


def process():
    frame_count = len(all_poses)
    tracks_pose = {}  # track_id -> frame_count x pose matrix
    unmatched_poses = {}  # frame -> poses
    previous_frame_id = -1  # Previous video frame processed
    out_frame_id = -1
    pose_frame_id_offset = min(all_poses.keys())

    for pose_frame_id in sorted(all_poses.keys()):
        start_frame = time.time()
        frame_id = pose_frame_id - pose_frame_id_offset
        ret_val, orig_image = get_orig_image(previous_frame_id, frame_id)
        previous_frame_id = frame_id
        out_frame_id += 1

        people_poses = all_poses[pose_frame_id]["people"]
        bboxes = filter_big_bboxes(all_tracks[frame_id])

        frame_tracks_pose = match_tracks_poses(bboxes, people_poses)

        drawn_pose_indices = list()
        canvas = orig_image.copy()
        unmatched_poses[pose_frame_id] = list()
        for track in bboxes:
            if track in frame_tracks_pose:
                if track not in tracks_pose:
                    # tracks_pose[track] = np.zeros((frame_count, pose_array_length))
                    tracks_pose[track] = {}
                pose_index = frame_tracks_pose[track]
                tracks_pose[track][pose_frame_id] = people_poses[pose_index]
                #pose_array = get_pose_array(people_poses[pose_index])
                #tracks_pose[track][out_frame_id] = pose_array
                drawn_pose_indices.append(pose_index)

                #  draw bbox and pose
                color = COLORS_10[track % len(COLORS_10)]
                canvas = draw_pose(canvas, people_poses[pose_index], color)
                canvas = draw_bbox(canvas, track, bboxes[track], color)
            else:  # draw bbox in grayscale
                canvas = draw_bbox(canvas, track, bboxes[track], gray)

        for index in range(len(people_poses)):  # draw pose in grayscale
            if index not in drawn_pose_indices:
                pose = people_poses[index]
                unmatched_poses[pose_frame_id].append(pose)
                canvas = draw_pose(canvas, pose, gray)

        draw_frame_num(canvas, out_frame_id, pose_frame_id)
        cam_out.write(canvas)
        print_time(f'frame {frame_id}', start_frame, time.time())

    return tracks_pose, unmatched_poses


def print_time(label, start, end):
    print("{}: {:.2} secs".format(label, end-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, required=True, help='pose features pkl file name')
    parser.add_argument('--track', type=str, required=True, help='track pkl file name')
    parser.add_argument('--video', type=str, required=True, help='video clip corresponding to pose & track pkl files')
    parser.add_argument('--out_fps', type=int, required=True, help='fps of output video')
    parser.add_argument('--out_dir', type=str, default='.', help='output base directory')

    args = parser.parse_args()
    out_prefix = args.video.rsplit(".", 1)[0].rsplit("_", 1)[0] + "_trackpose"

    # Pickle files for pose and tracking
    all_poses = pickle.load(open(args.features, "rb"))
    all_tracks = pickle.load(open(args.track, "rb"))

    # init input and output videos
    cam_in, cam_out, im_width, im_height = init_video()
    im_area = im_height * im_width

    tracks_pose = process()

    pickle.dump(tracks_pose, open(out_prefix+'_person.pkl', "wb"))
    # json.dump(tracks_pose, open(out_prefix+'_person.json', "w"), sort_keys=True, indent=4, separators=(',', ': '))
