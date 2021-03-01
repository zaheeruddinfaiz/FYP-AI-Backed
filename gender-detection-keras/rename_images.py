import os
i = 0
for file_name in os.listdir('./gender_dataset_face/woman'):
    dst = "woman_"+str(i)+".jpg"
    src = "./gender_dataset_face/woman/" + file_name
    dst = "./gender_dataset_face/woman/" + dst
    os.rename(src, dst)
    i += 1
