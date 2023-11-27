def get_index_by_name(name):
    keypoint_mapping = {
        "Nose": 0,
        "Left Eye Inner": 1,
        "Left Eye": 2,
        "Left Eye Outer": 3,
        "Right Eye Inner": 4,
        "Right Eye": 5,
        "Right Eye Outer": 6,
        "Left Ear": 7,
        "Right Ear": 8,
        "Mouth Left": 9,
        "Mouth Right": 10,
        "Left Shoulder": 11,
        "Right Shoulder": 12,
        "Left Elbow": 13,
        "Right Elbow": 14,
        "Left Wrist": 15,
        "Right Wrist": 16,
        "Left Pinky": 17,
        "Right Pinky": 18,
        "Left Index": 19,
        "Right Index": 20,
        "Left Thumb": 21,
        "Right Thumb": 22,
        "Left Hip": 23,
        "Right Hip": 24,
        "Left Knee": 25,
        "Right Knee": 26,
        "Left Ankle": 27,
        "Right Ankle": 28,
        "Left Heel": 29,
        "Right Heel": 30,
        "Left Foot Index": 31,
        "Right Foot Index": 32
    }

    return keypoint_mapping.get(name, -1)
 