import argparse
import os
import shutil
from glob import glob

import cv2
import numpy as np

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from midiutil import MIDIFile


def parse_args() -> dict:
    """
    Parse command line arguments.

    :return: arguments
    :rtype: dict {'arg_dest': 'value'}
    """

    parser = argparse.ArgumentParser(description="It's a command parser for arguments")
    parser.add_argument(
        "-file",
        default="",
        action="store",
        dest="FILE",
        type=str,
        required=True,
        help="Add your file.",
    )

    # args
    args = parser.parse_args()  # it's used in the loop eval

    # result dictionary
    argument_dest = ["FILE"]
    arguments = {}
    for dest in argument_dest:
        work = "args" + "." + dest
        arguments.update({dest: eval(work)})

    return arguments


def replace_png2jpg(path: str) -> None:
    """
    Replace all png files to jpg in directory.

    :param path:
    :type path: str
    :rtype: None
    """

    png_files_names = glob(f"{path}/*.png")

    # png 2 jpg
    for file in png_files_names:
        img = cv2.imread(file)
        cv2.imwrite(f"{os.path.splitext(file)[0]}.jpg", img)
        os.remove(file)


def replace_jpeg2jpg(path: str) -> None:
    """
    Replace all jpeg files to jpg in directory.

    :param path:
    :type path: str
    :rtype: None
    """

    jpeg_files_names = glob(f"{path}/*.jpeg")

    # jpeg 2 jpg
    for file in jpeg_files_names:
        img = cv2.imread(file)
        cv2.imwrite(f"{os.path.splitext(file)[0]}.jpg", img)
        os.remove(file)


def IsItMusicSheet(model, image_path: str) -> bool:
    """
    Checks if the image contains music sheet.

    :param model: CNN model from filter_model directory.
    :type model: keras.src.engine.sequential.Sequential
    :param image_path:
    :type image_path: str
    :return: is it music sheet?
    :rtype: bool
    """

    # Preprocessing
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)

    return bool(pred[0][0] > 0.5)


def filter_images_in_dir(model, dir_path: str) -> None:
    """
    Filters images in a directory to: music sheets & others directories.

    :param model: CNN model from filter_model directory
    :type model: keras.src.engine.sequential.Sequential
    :param dir_path: directory path
    :rtype: None
    """

    # new directories paths
    other_images_path = dir_path + "other/"
    music_sheets_path = dir_path + "music_sheets/"

    # drop and create them
    if not os.path.exists(other_images_path):
        os.makedirs(other_images_path)
    else:
        shutil.rmtree(other_images_path)
        os.makedirs(other_images_path)

    if not os.path.exists(music_sheets_path):
        os.makedirs(music_sheets_path)
    else:
        shutil.rmtree(music_sheets_path)
        os.makedirs(music_sheets_path)

    # images sort
    images_paths = glob(dir_path + "*.jpg")

    for img_path in images_paths:
        if IsItMusicSheet(model, img_path):
            shutil.copy2(img_path, music_sheets_path)
        else:
            shutil.copy2(img_path, other_images_path)
        os.remove(img_path)


def create_fill_work_dir(work_data_path: str, start_data_path: str) -> None:
    """
    Create and fill work directory with start_data_path images in jpg format.

    :param work_data_path:
    :type work_data_path: str
    :param start_data_path:
    :type start_data_path: str
    :rtype: None
    """

    if not os.path.exists(work_data_path):
        os.makedirs(work_data_path)
    else:
        shutil.rmtree(work_data_path)
        os.makedirs(work_data_path)

    shutil.copytree(start_data_path, work_data_path, dirs_exist_ok=True)

    replace_png2jpg(work_data_path)
    replace_jpeg2jpg(work_data_path)


def get_convert_to_midi_seq(staff: np.ndarray, model):
    def get_black_poses(arr):
        poses = []
        for i in range(len(arr)):
            if arr[i] in list(range(0, 21)):
                poses.append(i)
        return poses

    detected_notes = model(staff)

    # Boxes
    boxes = detected_notes[0].boxes.cpu().numpy()
    boxes_coords = {}
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        xywh = box.xywh

        x = xywh[0, 0]
        y = xywh[0, 1]
        w = xywh[0, 2]
        h = xywh[0, 3]

        boxes_coords.update({(i, cls): [x, y, w, h]})

    sorted_boxes_coords = dict(sorted(boxes_coords.items(), key=lambda item: item[1][0]))

    # Get lines
    lines = []
    for key, value in sorted_boxes_coords.items():
        if key[1] == 12:
            lines.append(value)

    left_help_lines_coords = []
    right_help_lines_coords = []

    for line in lines:
        x = line[0]
        y = line[1]
        w = line[2]
        h = line[3]
        staff_g = cv2.cvtColor(staff, cv2.COLOR_BGR2GRAY)

        left_column = staff_g[int(y - h / 2): int(y + h / 2), int(x - w / 4)]
        right_column = staff_g[int(y - h / 2): int(y + h / 2), int(x + w / 4)]

        left_line = get_black_poses(left_column)
        right_line = get_black_poses(right_column)

        #print('left_line', type(left_line), len(left_line), left_line)
        #print('right_line', type(right_line), len(right_line), right_line)

        # main lines
        if len(left_line) <= 5:
            #print('left')
            left_main_lines_coords = [(int(x + w / 4), int(y - h / 2 + pose)) for pose in left_line]
            left_help_lines_coords.append(left_main_lines_coords)
            #print('left_help', type(left_help_lines_coords), len(left_help_lines_coords), left_help_lines_coords)


        if len(right_line) <= 5:
            #print('right')
            right_main_lines_coords = [(int(x + w / 4), int(y - h / 2 + pose)) for pose in right_line]
            right_help_lines_coords.append(right_main_lines_coords)

            #print('right_help', type(right_help_lines_coords), len(right_help_lines_coords), right_help_lines_coords)

    #y_lines_coords = [item[1] for item in right_help_lines_coords[0]]
    y_lines_coords = []

    #print('Coords', type(right_help_lines_coords), len(right_help_lines_coords), right_help_lines_coords)
    #print('Coords1', type(right_help_lines_coords[0]), right_help_lines_coords[0])
    if len(right_help_lines_coords) > 0:
        for item in right_help_lines_coords[0]:
            y_lines_coords.append(item[1])
    elif len(left_help_lines_coords) > 0:
        for item in left_help_lines_coords[0]:
            y_lines_coords.append(item[1])
    else:
        left_help_len = len(left_help_lines_coords)
        right_help_len = len(right_help_lines_coords)
        print(f'ERROR! Got less then 5 main lines: left: {left_help_len}, right: {right_help_len} ')
        return []

    print('Y:', y_lines_coords)

    note_line1_y = y_lines_coords[0]
    note_line2_y = y_lines_coords[1]

    h = note_line2_y - note_line1_y
    print('h:', h)

    x, y = 250, note_line1_y
    r = 5
    cv2.circle(staff, (x, note_line1_y), r, (0, 0, 255), -1)
    cv2.circle(staff, (x, note_line2_y), r, (0, 0, 255), -1)
    cv2.circle(staff, (x, note_line2_y+h), r, (0, 0, 255), -1)
    cv2.circle(staff, (x, note_line2_y+2*h), r, (0, 0, 255), -1)
    cv2.circle(staff, (x, note_line2_y+3*h), r, (0, 0, 255), -1)

    def show_img(image, name):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    show_img(staff, 'rrr')



    # alpha
    # midi_pitch: line_number
    alphabet = {
        21: 26.5, 23: 26, 24: 25.5, 26: 25, 28: 24.5, 29: 24,
        31: 23.5, 33: 23, 35: 22.5, 36: 22, 38: 21.5, 40: 21,
        41: 20.5, 43: 20, 45: 19.5, 47: 19, 48: 18.5, 50: 18,
        52: 17.5, 53: 17, 55: 16.5, 57: 16, 59: 15.5, 60: 15,
        62: 14.5, 64: 14, 65: 13.5, 67: 13, 69: 12.5, 71: 12,
        72: 11.5, 74: 11, 76: 10.5, 77: 10, 79: 9.5, 81: 9,
        83: 8.5, 84: 8, 86: 7.5, 88: 7, 89: 6.5, 91: 6,
        93: 5.5, 95: 5, 96: 4.5, 98: 4, 100: 3.5, 101: 3,
        103: 2.5, 105: 2, 107: 1.5, 108: 1
    }

    # find all clefs
    clef_classes = [0, 1, 27]
    clef_dict = {}
    for key, value in sorted_boxes_coords.items():
        if key[1] in clef_classes:
            clef_dict[key] = value

    # Notes
    notes_classes = [10, 11, 14, 15, 17]
    notes_elements = {}

    for key, value in sorted_boxes_coords.items():
        if key[1] in notes_classes:
            notes_elements.update({key: value})

            # Create all 26 lines
    all_lines_y = {}
    first_key = list(clef_dict.keys())[0][1]

    h = y_lines_coords[1] - y_lines_coords[0]
    all_keys = np.arange(1.5, 25.6, 0.5)

    for key in notes_elements.keys():
        if first_key == 1:
            # Bass-clef
            all_lines_y[1] = y_lines_coords[0] - 15 * h
            all_lines_y[26.5] = y_lines_coords[1] + 9 * h
        else:
            # NOT a Bass-clef
            all_lines_y[1] = y_lines_coords[0] - 9 * h
            all_lines_y[26.5] = y_lines_coords[1] + 15 * h

        for i, key in enumerate(all_keys):
            all_lines_y[key] = all_lines_y[1] + h * (i + 1) / 2

    all_lines_y = dict(sorted(all_lines_y.items()))
    numbered_all_lines_dict = {i: (key, value) for i, (key, value) in enumerate(all_lines_y.items(), start=0)}

    # Get notes y-es
    lines_y_values = np.array(list(all_lines_y.values()))

    notes_classes = [10, 11, 14, 15, 17]
    note_line_dict = {}

    for key, value in notes_elements.items():
        if key[1] in notes_classes:
            line_ind = np.argmin(np.abs(lines_y_values - value[1]))
            note_line_dict.update({key[0]: numbered_all_lines_dict[line_ind][0]})

    # Get midi seq
    midi_seq = []
    for id, line_num in note_line_dict.items():
        for pitch, line_num2 in alphabet.items():
            if line_num == line_num2:
                midi_seq.append(pitch)

    return midi_seq

def create_midi(pitch_array, duration, output_file):
    midi = MIDIFile(1)

    track = 0
    time = 0

    midi.addTrackName(track, time, "Sample Track")
    midi.addTempo(track, time, 120)

    channel = 0
    volume = 100

    for p in pitch_array:
        midi.addNote(track, channel, p, time, duration, volume)
        time += 1

    with open(output_file, "wb") as f:
        midi.writeFile(f)
