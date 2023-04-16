import cv2
import os


def make_dir(dir_path: str) -> None:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    else:
        pass

vid1_path = "dataset/original/1.mp4"
vid2_path = "dataset/original/2.mp4"
vid3_path = "dataset/original/3.mp4"
vid4_path = "dataset/original/4.mp4"

# define a video capture object
# vid1 = cv2.VideoCapture('rtsp://192.168.5.101/main')
# vid2 = cv2.VideoCapture('rtsp://192.168.5.102/main')
# vid3 = cv2.VideoCapture('rtsp://192.168.5.103/main')
# vid4 = cv2.VideoCapture('rtsp://192.168.5.104/main')
vid1 = cv2.VideoCapture(vid1_path)
vid2 = cv2.VideoCapture(vid2_path)
vid3 = cv2.VideoCapture(vid3_path)
vid4 = cv2.VideoCapture(vid4_path)

list_input_vid = [vid1, vid2, vid3, vid4]

make_dir("video")
list_video_frames_dir = []
for i in range(len(list_input_vid)):
    video_frames_dir = f"video/{i+1}"
    list_video_frames_dir.append(video_frames_dir)
    make_dir(video_frames_dir)

while(True):
    count = 0

    _, frame1 = vid1.read()
    # _, frame2 = vid2.read()
    # _, frame3 = vid3.read()
    # _, frame4 = vid4.read()

    cv2.imwrite(f"{list_video_frames_dir[0]}/frame{count}.jpg", frame1)
    # cv2.imwrite(f"{list_video_frames_dir[1]}/frame{count}.jpg", frame2)
    # cv2.imwrite(f"{list_video_frames_dir[2]}/frame{count}.jpg", frame3)
    # cv2.imwrite(f"{list_video_frames_dir[3]}/frame{count}.jpg", frame4)

    count += 1
    
    if 0xFF == ord('q'):
        break
# After the loop release the cap object
# vid1.release()
# Destroy all the windows
# cv2.destroyAllWindows()