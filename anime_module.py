import os
import sys
import cv2
import shutil
import matplotlib.pyplot as plt

def get_list(folder_path):

    photo_list = os.listdir(folder_path)

    original_photo_list = []
    flipLR_photo_list = []
    flipTB_photo_list = []
    spin90_photo_list = []
    spin270_photo_list = []

    for photo in photo_list:

        if photo.find("_") == -1:
            original_photo_list.append(photo)

        elif photo.startswith("flipLR") == 1:
            flipLR_photo_list.append(photo)

        elif photo.startswith("flipTB") == 1:
            flipTB_photo_list.append(photo)

        elif photo.startswith("spin90") == 1:
            spin90_photo_list.append(photo)

        elif photo.startswith("spin270") ==1:
            spin270_photo_list.append(photo)

    photo_list = []

    for i in range(len(original_photo_list)):
        photo_list.append(original_photo_list[i])
        photo_list.append(flipLR_photo_list[i])
        photo_list.append(flipTB_photo_list[i])
        photo_list.append(spin90_photo_list[i])
        photo_list.append(spin270_photo_list[i])

    return photo_list

def make_folder():
    f = open("character_list.txt", "r", encoding = "utf-8")
    char_list = f.read()
    char_list = char_list.split("\n")[:-1]
    # キャラクターリストに名前がなければフォルダを作成
    for c in char_list:
        if not os.path.exists(c):
            os.mkdir(c)
    print("フォルダを作成しました")
