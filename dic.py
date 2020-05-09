import copy
import os
import random
path="static/img"
label_list = os.listdir(path)
img_names = copy.deepcopy(label_list)

for index,name in enumerate(label_list):
    label_list[index] = name.replace('.jpg', '')
    label_list[index] = label_list[index].split(',')

dic = {img_names[i]: label_list[i] for i in range(len(img_names))}

def find_key(input_dict, key_word):
    return [key for key, value in input_dict.items() if key_word in value]

key_word = input()
sticker_name = find_key(dic,key_word)
print(random.choice(sticker_name))
