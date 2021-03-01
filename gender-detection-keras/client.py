import read_video_frame
import gc
import time
from yoloface import get_age_gender
import requests
import os
import webbrowser
tries = 0
jwt = os.environ['jwt']
while True:
    tries += 1
    if(tries >= 10):

        # Read the frames from the given video file
        frame = read_video_frame.get_frame(seconds=10)

        # Extract the age and gender of the subject based on the facial features
        age_gender = get_age_gender(frame)

        # If age and gender data is valid, send the data to the server
        if(age_gender is not None):
            print('Requesting Server')
            tries = 0
            webbrowser.open(
                url=f'http://localhost:3000/{age_gender[0]}/{age_gender[1].split(":")[0]}/{jwt}', new=0)

            time.sleep(30)
