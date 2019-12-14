import pickle
import cv2
import json
import sys
import math

number_of_vertical_slices = 6
number_of_horizontal_slices = 6


def is_audience(jp):
    missing_parts = 0
    lower_corner = False
    if 'rlowerleg' not in jp:
        missing_parts = missing_parts + 1
    if 'llowerleg' not in jp:
        missing_parts = missing_parts + 1
    if 'llowerarm' not in jp:
        missing_parts = missing_parts + 1
    if 'rlowerarm' not in jp:
        missing_parts = missing_parts + 1
    if ('lupperarm' in jp and (0.2 > jp['lupperarm']['from'][1] / 1080 >= 0.85)) or (
            'rupperarm' in jp and (0.2 > jp['rupperarm']['from'][1] / 1080 >= 0.85)):
        lower_corner = True
    if missing_parts >= 2 and lower_corner:
        return True
    else:
        return False


def get_sitting_persons(jps):
    person_heights = {}
    sitting_person_indices = []
    i = 0
    for jp in jps:
        if not is_audience(jp) and ('llowerleg' in jp and 'lhipneck' in jp):
            size = jp['llowerleg']['to'][1] - jp['lhipneck']['from'][1]
            person_heights[i] = size
        else:
            sitting_person_indices.append(i)
        i = i + 1
    max_height = max([person_heights[j] for j in person_heights])
    for j in person_heights:
        if person_heights[j] / max_height <= 0.6:
            sitting_person_indices.append(j)
    return sitting_person_indices, max_height


def get_corrected_parts(jps, bps):
    removed_parts = []
    removed_people, max_height = get_sitting_persons(jps)
    for p in removed_people:
        jp = jps[p]
        for part_name in jp:
            from_p = jp[part_name]['from']
            to_p = jp[part_name]['to']
            removed_parts.append(from_p)
            removed_parts.append(to_p)
    newbps = {}
    for part_name in bps:
        points_arr = bps[part_name]
        new_points_arr = []
        for pp in points_arr:
            point_x = pp['x']
            point_y = pp['y']
            point_found = False
            for rem in removed_parts:
                if point_x == rem[0] and point_y == rem[1]:
                    point_found = True
                    # print("point found"+str(pp))
                    break
            if not point_found:
                new_points_arr.append(pp)
        newbps[part_name] = new_points_arr
    return newbps, removed_people, removed_parts, max_height


# only adds filtered people parts
def get_corrected_parts_v2(jps, bps):
    non_audience_people = []
    removed_people, max_height = get_sitting_persons(jps)
    for index in range(len(jps)):
        if index not in removed_people:
            non_audience_people.append(jps[index])
    keep_parts = []
    for jp in non_audience_people:
        for part_name in jp:
            from_p = jp[part_name]['from']
            to_p = jp[part_name]['to']
            keep_parts.append(from_p)
            keep_parts.append(to_p)
    newbps = {}
    for part_name in bps:
        points_arr = bps[part_name]
        new_points_arr = []
        for pp in points_arr:
            point_x = pp['x']
            point_y = pp['y']
            for point in keep_parts:
                if point_x == point[0] and point_y == point[1]:
                    new_points_arr.append(pp)
                    break
        newbps[part_name] = new_points_arr
    return newbps, removed_people, [], max_height


def get_corrected_parts_v3(jps, bps):
    non_audience_people = []
    removed_people, max_height = get_sitting_persons(jps)
    for index in range(len(jps)):
        if index not in removed_people:
            non_audience_people.append(jps[index])
    return non_audience_people, max_height, removed_people


def get_key_features(json_path, f, important_frames):
    fjson = open(json_path)
    poses = json.loads(fjson.read())
    # calculation for every frame
    fbuckets = {}
    frame_max_height = {}
    for frame_number in f:
        joints = {}
        all_body_parts = poses[str(frame_number)]['body_parts']
        people = poses[str(frame_number)]['people']
        non_audience_body_parts, removed_people, removed_parts, max_height = get_corrected_parts_v2(people, all_body_parts)
        frame_max_height[frame_number] = max_height
        for key in non_audience_body_parts:
            if key not in joints:
                joints[key] = []
            for dd in poses[str(frame_number)]['body_parts'][key]:
                joints[key].append((dd['x'], dd['y']))
        (bottom_left, top_right) = f[frame_number]
        bucket_mean_x = (top_right[0] + bottom_left[0]) / 2.0
        bucket_mean_y = (top_right[1] + bottom_left[1]) / 2.0
        buckets = {}
        if bucket_mean_x / 1920 >= 0.5:
            # right side
            slice_size = int(bucket_mean_x / number_of_vertical_slices)
            # print("slice size "+str(slice_size))
            # count number of parts in each bucket and put in count
            for key in joints:
                for joint in joints[key]:
                    bkt_x = int(bucket_mean_x - joint[0]) % number_of_vertical_slices
                    if joint[1] < bucket_mean_y:
                        bkt_y = -1
                    else:
                        bkt_y = (int(joint[1] - bucket_mean_y)) % number_of_horizontal_slices
                    if bkt_x not in buckets:
                        buckets[bkt_x] = {}
                    if bkt_y not in buckets[bkt_x]:
                        buckets[bkt_x][bkt_y] = {}
                    if key not in buckets[bkt_x][bkt_y]:
                        buckets[bkt_x][bkt_y][key] = []
                    buckets[bkt_x][bkt_y][key].append(joint)
        fbuckets[frame_number] = buckets
    return fbuckets, poses, frame_max_height


def add_joint(bkt_x, bkt_y, key, joint_ee, buckets):
    if bkt_x not in buckets:
        buckets[bkt_x] = {}
    if bkt_y not in buckets[bkt_x]:
        buckets[bkt_x][bkt_y] = {}
    if key not in buckets[bkt_x][bkt_y]:
        buckets[bkt_x][bkt_y][key] = []
    buckets[bkt_x][bkt_y][key].append(joint_ee)


def get_key_features_v2(json_path, f, important_frames, del_xf1, del_xf2, del_xb):
    fjson = open(json_path)
    poses = json.loads(fjson.read())
    # calculation for every frame
    fbuckets = {}
    frame_max_height = {}
    del_xy = {}
    for frame_number in f:
        all_body_parts = poses[str(frame_number)]['body_parts']
        people = poses[str(frame_number)]['people']
        non_audience_body_parts, removed_people, removed_parts, max_height = get_corrected_parts_v2(people, all_body_parts)
        frame_max_height[frame_number] = max_height
        joints = non_audience_body_parts
        (bottom_left, topRight) = f[frame_number]
        bucket_mean_x = (topRight[0] + bottom_left[0]) / 2.0
        bucket_mean_y = (topRight[1] + bottom_left[1]) / 2.0
        # print(str(frame_number)+" "+str(bucket_mean_x)+" "+str(bucket_mean_y))
        buckets = {}
        del_x_fff1 = del_xf1 * max_height
        del_x_fff2 = del_xf2 * max_height
        del_x_bff = del_xb * max_height
        del_xy[frame_number] = (del_x_fff2, del_x_fff1, del_x_bff)
        if bucket_mean_x / 1920 >= 0.5:
            # right side
            slice_size = int(bucket_mean_x / number_of_vertical_slices)
            # print("slice size "+str(slice_size))
            # count number of parts in each bucket and put in count
            for key in joints:
                for joint_ee in joints[key]:
                    joint_x = joint_ee['x']
                    joint_y = joint_ee['y']
                    if (bucket_mean_x >= joint_x >= bucket_mean_x - del_x_fff1 and joint_y >= bucket_mean_y) or (
                            bucket_mean_x <= joint_x <= bucket_mean_x + del_x_bff and joint_y >= bucket_mean_y):
                        bkt_y = 0
                        bkt_x = 0
                        add_joint(bkt_x, bkt_y, key, joint_ee, buckets)
                        break
                    if (bucket_mean_x - del_x_fff1 > joint_x >= bucket_mean_x - del_x_fff2 and joint_y >= bucket_mean_y)\
                            or (bucket_mean_x <= joint_x <= bucket_mean_x + del_x_bff and joint_y >= bucket_mean_y):
                        bkt_y = 0
                        bkt_x = 1
                        add_joint(bkt_x, bkt_y, key, joint_ee, buckets)
                        break
        fbuckets[frame_number] = buckets
    return fbuckets, poses, frame_max_height, del_xy


def inside_box(joint_x, joint_y, bucket_mean_x, bucket_mean_y, del_xf_ff1, del_xb_ff):
    return (bucket_mean_x >= joint_x >= bucket_mean_x - del_xf_ff1 and joint_y >= bucket_mean_y) or (
            bucket_mean_x <= joint_x <= bucket_mean_x + del_xb_ff and joint_y >= bucket_mean_y)


def inside_exterior_of_d(joint_x, joint_y, bucket_mean_x, bucket_mean_y, del_xf_ff2, del_xf_ff1, del_xb_ff):
    return (bucket_mean_x - del_xf_ff1 > joint_x >= bucket_mean_x - del_xf_ff2 and joint_y >= bucket_mean_y) or (
            bucket_mean_x <= joint_x <= bucket_mean_x + del_xb_ff and joint_y >= bucket_mean_y)


def add_person(bkt_x, bkt_y, pp, buckets):
    if bkt_x not in buckets:
        buckets[bkt_x] = {}
    if bkt_y not in buckets[bkt_x]:
        buckets[bkt_x][bkt_y] = []
    buckets[bkt_x][bkt_y].append(pp)


def cluster_people(removed_people_indices, people, parts):
    all_people = people
    parts_of_removed_people = {}
    parts_of_selected_people = {}
    for ri in removed_people_indices:
        person = all_people[ri]
        for key in person:
            parts_of_removed_people[(person[key]['from'][0], person[key]['from'][1])] = 1
            parts_of_removed_people[(person[key]['to'][0], person[key]['to'][1])] = 1
    for ri in range(len(all_people)):
        if ri not in removed_people_indices:
            person = all_people[ri]
            for key in person:
                parts_of_selected_people[(person[key]['from'][0], person[key]['from'][1])] = 1
                parts_of_selected_people[(person[key]['to'][0], person[key]['to'][1])] = 1
    all_parts = parts
    hanging_joints = {}
    for joint in all_parts:
        joints = all_parts[joint]
        if joint not in hanging_joints:
            hanging_joints[joint] = []
        for jjk in joints:
            if not ((jjk['x'], jjk['y']) in parts_of_removed_people) or ((jjk['x'], jjk['y']) in parts_of_selected_people):
                hanging_joints[joint].append(jjk)
    return parts_of_removed_people, parts_of_selected_people, hanging_joints


def distance(joint, person_t):
    return math.sqrt((pow(joint['x'] - person_t[0], 2) + pow(joint['y'] - person_t[1], 2)))


def is_joint_closed_to_selected_people(max_height, joint, parts_of_removed_people, parts_of_selected_people, ratio):
    min_removed_distance = sys.float_info.max
    for removedP in parts_of_removed_people:
        min_removed_distance = min(min_removed_distance, distance(joint, removedP))
    min_selected_distance = sys.float_info.max
    for selected_p in parts_of_selected_people:
        min_selected_distance = min(min_selected_distance, distance(joint, selected_p))
    if min_selected_distance == 0:
        return False
    else:
        return min_removed_distance == 0 or (min_selected_distance / max_height <= ratio)


def filter_closer_to_selected_people_hanging_joints(max_height, removed_people_indices, people, parts, ratio):
    # cluster parts based on selected and not selected people
    parts_of_removed_people, parts_of_selected_people, hanging_joints = cluster_people(removed_people_indices, people, parts)
    filtered_hanging_joints = {}
    for jointName in hanging_joints:
        if jointName not in filtered_hanging_joints:
            filtered_hanging_joints[jointName] = []
        for joint in hanging_joints[jointName]:
            if is_joint_closed_to_selected_people(max_height, joint, parts_of_removed_people, parts_of_selected_people, ratio):
                filtered_hanging_joints[jointName].append(joint)
    return filtered_hanging_joints


def reverse_index_buckets(buckets):
    reverse_person_buckets = {}
    for bkt_x in buckets:
        for bkt_y in buckets[bkt_x]:
            for pp in buckets[bkt_x][bkt_y]:
                for joint_name in pp:
                    jj_from = pp[joint_name]['from']
                    jj_to = pp[joint_name]['to']
                    reverse_person_buckets[(jj_from[0], jj_from[1])] = (bkt_x, bkt_y)
                    reverse_person_buckets[(jj_to[0], jj_to[1])] = (bkt_x, bkt_y)
    return reverse_person_buckets


def find_nearest_person(joint, reverse_person_buckets, max_height, ratio):
    joint_x = joint['x']
    joint_y = joint['y']
    b_x = -1
    b_y = -1
    distance_min = sys.float_info.max
    for kk in reverse_person_buckets:
        # print(str(joint)+" "+str(kk))
        if distance_min > min(distance_min, distance(joint, kk)):
            distance_min = min(distance_min, distance(joint, kk))
            b_x, b_y = reverse_person_buckets[kk]
    if not distance_min / max_height <= ratio:
        b_x = -1
        b_y = -1
    return b_x, b_y


def get_key_features_v3(json_path, f, important_frames, del_xf1, del_xf2, del_xb,
                        ratio_of_height_to_hanging_joint_distance):
    fjson = open(json_path)
    poses = json.loads(fjson.read())
    # calculation for every frame
    fbuckets = {}
    f_hoops = {}
    removed_people_in_frames = {}
    frame_max_height = {}
    del_xy = {}
    all_non_aud = {}
    filtered_hanging_joints_in_frames = {}
    bucketed_hanging_joints_in_frame = {}
    for frame_number in f:
        all_body_parts = poses[str(frame_number)]['body_parts']
        people = poses[str(frame_number)]['people']
        # find likely non audience full person poses
        non_audience_people, max_height, removed_people_indices = get_corrected_parts_v3(people, all_body_parts)
        filtered_hanging_joints = \
            filter_closer_to_selected_people_hanging_joints(max_height, removed_people_indices, people, all_body_parts,
                                                            ratio_of_height_to_hanging_joint_distance)
        filtered_hanging_joints_in_frames[frame_number] = filtered_hanging_joints
        removed_people_in_frames[frame_number] = removed_people_indices
        all_non_aud[frame_number] = non_audience_people
        frame_max_height[frame_number] = max_height
        (bottom_left, top_right) = f[frame_number]
        bucket_mean_x = (top_right[0] + bottom_left[0]) / 2.0
        bucket_mean_y = (top_right[1] + bottom_left[1]) / 2.0
        f_hoops[frame_number] = (bucket_mean_x, bucket_mean_y)
        buckets = {}
        del_x_fff1 = del_xf1 * max_height
        del_x_fff2 = del_xf2 * max_height
        del_x_bff = del_xb * max_height
        del_xy[frame_number] = (del_x_fff2, del_x_fff1, del_x_bff)
        limbs = ['llowerleg', 'rlowerleg', 'lhipneck', 'rhipneck']
        if bucket_mean_x / 1920 >= 0.5:
            for pp in non_audience_people:
                if 'llowerleg' in pp:
                    joint_ee = pp['llowerleg']
                elif 'rlowerleg' in pp:
                    joint_ee = pp['rlowerleg']
                elif 'lhipneck' in pp:
                    joint_ee = pp['lhipneck']
                elif 'rhipneck' in pp:
                    joint_ee = pp['rhipneck']
                else:
                    # print("person without leg:"+str(pp))
                    break

                joint_from = joint_ee['from']
                # print(joint_from)
                joint_from_x = joint_from[0]
                joint_from_y = joint_from[1]
                joint_to = joint_ee['to']
                joint_to_x = joint_to[0]
                joint_to_y = joint_to[1]
                # insideBox(joint_from_x, joint_from_y, bucket_mean_x, bucket_mean_y, del_x_fff1,del_x_bff) or
                if inside_box(joint_from_x, joint_from_y, bucket_mean_x, bucket_mean_y, del_x_fff1, del_x_bff) or \
                        inside_box(joint_to_x, joint_to_y, bucket_mean_x, bucket_mean_y, del_x_fff1, del_x_bff):
                    bkt_y = 0
                    bkt_x = 0
                    add_person(bkt_x, bkt_y, pp, buckets)
                    continue
                # insideExteriorOfD(joint_from_x, joint_from_y, bucket_mean_x, bucket_mean_y, del_x_fff2, del_x_fff1,del_x_bff) or
                if inside_exterior_of_d(
                        joint_from_x, joint_from_y, bucket_mean_x, bucket_mean_y, del_x_fff2, del_x_fff1, del_x_bff) or\
                        inside_exterior_of_d(joint_to_x, joint_to_y, bucket_mean_x, bucket_mean_y, del_x_fff2,
                                             del_x_fff1, del_x_bff):
                    bkt_y = 0
                    bkt_x = 1
                    add_person(bkt_x, bkt_y, pp, buckets)
                    continue
        # hanging joints bucketing
        reverse_person_buckets = reverse_index_buckets(buckets)
        bucketed_hanging_joints = {}
        for jointName in filtered_hanging_joints:
            joints = filtered_hanging_joints[jointName]
            for joint in joints:
                b_x, b_y = find_nearest_person(joint, reverse_person_buckets, max_height,
                                               ratio_of_height_to_hanging_joint_distance)
                if b_x == -1 or b_y == -1:
                    continue
                if b_x not in bucketed_hanging_joints:
                    bucketed_hanging_joints[b_x] = {}
                if b_y not in bucketed_hanging_joints[b_x]:
                    bucketed_hanging_joints[b_x][b_y] = {}
                if jointName not in bucketed_hanging_joints[b_x][b_y]:
                    bucketed_hanging_joints[b_x][b_y][jointName] = []
                bucketed_hanging_joints[b_x][b_y][jointName].append(joint)
        bucketed_hanging_joints_in_frame[frame_number] = bucketed_hanging_joints
        fbuckets[frame_number] = buckets
    return fbuckets, poses, frame_max_height, del_xy, all_non_aud, f_hoops, removed_people_in_frames, \
           filtered_hanging_joints_in_frames, bucketed_hanging_joints_in_frame


# to draw and debug the near/ far people
def draw_sectors(k, img, features, fname):
    removed_people_indices = features[6][k]
    all_people = features[1][str(k)]['people']
    parts_of_removed_people = {}
    for ri in removed_people_indices:
        person = all_people[ri]
        for key in person:
            parts_of_removed_people[(person[key]['from'][0], person[key]['from'][1])] = 1
            parts_of_removed_people[(person[key]['to'][0], person[key]['to'][1])] = 1
    allparts = features[1][str(k)]['body_parts']
    for joint in allparts:
        joints = allparts[joint]
        for jjk in joints:
            if not (jjk['x'], jjk['y']) in parts_of_removed_people:
                cv2.circle(img, (jjk['x'], jjk['y']), 6, (255, 255, 0), -1)
    good_hanging_joints = features[7][k]
    # print("good hanging joints")
    for joint_name in good_hanging_joints:
        joints = good_hanging_joints[joint_name]
        for jj in joints:
            cv2.circle(img, (jj['x'], jj['y']), 6, (255, 255, 255), -1)
    all_pp = features[4][k]
    for pp in all_pp:
        for limb in pp:
            joints = pp[limb]
            jj = joints['from']
            cv2.circle(img, (jj[0], jj[1]), 6, (0, 255, 0), -1)
            jj = joints['to']
            cv2.circle(img, (jj[0], jj[1]), 6, (0, 255, 0), -1)
    bucket = features[5][k]
    cv2.circle(img, (int(bucket[0]), int(bucket[1])), 10, (0, 0, 255), -1)
    for xk in features[0][k]:
        for yk in features[0][k][xk]:
            if xk == 0 and yk == 0:
                color = (255, 0, 0)
            elif xk == 1 and yk == 0:
                color = (0, 255, 255)
            for pp in features[0][k][xk][yk]:
                for limb in pp:
                    joints = pp[limb]
                    jj = joints['from']
                    cv2.circle(img, (jj[0], jj[1]), 6, color, -1)
                    jj = joints['to']
                    cv2.circle(img, (jj[0], jj[1]), 6, color, -1)
    for xk in features[8][k]:
        for yk in features[8][k][xk]:
            if xk == 0 and yk == 0:
                color = (255, 0, 0)
            elif xk == 1 and yk == 0:
                color = (0, 255, 255)
            for joint_name in features[8][k][xk][yk]:
                joints = features[8][k][xk][yk][joint_name]
                for jj in joints:
                    cv2.circle(img, (jj['x'], jj['y']), 6, color, -1)
    cv2.imwrite(fname + "-" + str(k) + "-all-sectors.jpg", img)
    # cv2.imwrite(fname+"-"+str(k)+".jpg",t7[1][k])
    return img
