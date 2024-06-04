from classes.Image import Image
from functions.functions import *
from ultralytics import YOLO
import tensorflow as tf

WORK_DATA_PATH = "work_data/"
START_DATA_PATH = "start_data/"
IMAGES_TYPES_PATH = "images_types/"
PERFECT_SHEETS_PATH = "perfect_sheets/"


# example: python3 main.py -file 'your_path'
def main():
    # arguments = parse_args()
    # START_DATA_PATH = arguments["FILE"]
    # print('Start_data', START_DATA_PATH)

    #filter_model = tf.keras.models.load_model("/home/toni/aristek/internship/music_sheet_dubbing/models/filter/checkpoints/best_model.h5")

    # print('Create work_dir...')
    # create_fill_work_dir(WORK_DATA_PATH, START_DATA_PATH)
    #
    # print('Filtering images...')
    # filter_images_in_dir(filter_model, WORK_DATA_PATH)

    jpg_file_names = glob(f"{WORK_DATA_PATH + str('music_sheets/')}*.jpg")

    #jpg_file_names = glob(IMAGES_TYPES_PATH + "*.jpg")
    jpg_file_names = glob(PERFECT_SHEETS_PATH + "*.jpg")

    stuff_detect_model = YOLO('models/staff_detection/V2/runs/detect/train2/weights/best.pt')
    notes_detection_model = YOLO('models/notes_detection/V2/train/weights/best.pt')

    # good [3:4],
    for ind, image_file in enumerate(jpg_file_names[3:4]):
        img = Image(image_file)
        img.image = img.get_color_image()
        img.split_into_staffs(stuff_detect_model)

        img.show_image_plt(img.image, img.file_path)

        midi_seq = []
        staffs = img.staffs
        for i, staff in enumerate(staffs):
            staff.show_plt(f'Stuff: {i}')

            print('\n--> Staff:', i)
            midi = get_convert_to_midi_seq(staff.staff, notes_detection_model)
            print('midi', midi)
            midi_seq += midi

        print('midi_seq:', midi_seq)
        create_midi(midi_seq, 960, f'output{ind}.mid')


if __name__ == "__main__":
    main()
