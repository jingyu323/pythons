import subprocess

import cv2
import pydub


def audio_test():
    mp3 = pydub.AudioSegment.from_mp3('../mp3/wobuyuanniyigeren.mp3')
    m = mp3[:21*1000]
    m.export("ss.mp3", format="mp3")
# 音视频合并
def audio_video_merge():
    cmd = "ffmpeg -i  ../video/video.mp4 -i ss.mp3   output.mp4"
    c = subprocess.call(cmd, shell=True)
    print(c)





if __name__ == '__main__':
    audio_video_merge()