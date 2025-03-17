# import glob
# from datetime import datetime

# import cv2
# import numpy as np
# import yaml

# from webcam import WebcamSource


# def record_video(width: int, height: int, fps: int) -> None:
#     """
#     Create a mp4 video file with `width`x`height` and `fps` frames per second.
#     Shows a preview of the recording every 5 frames.

#     :param width: width of the video
#     :param height: height of the video
#     :param fps: frames per second
#     :return: None
#     """

#     source = WebcamSource(width=width, height=height, fps=fps, buffer_size=10)
#     video_writer = cv2.VideoWriter(f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
#     for idx, frame in enumerate(source):
#         video_writer.write(frame)
#         source.show(frame, only_print=idx % 5 != 0)


# def calibration(image_path, every_nth: int = 1, debug: bool = False, chessboard_grid_size=(7, 7)):
#     """
#     Perform camera calibration on the previously collected images.
#     Creates `calibration_matrix.yaml` with the camera intrinsic matrix and the distortion coefficients.

#     :param image_path: path to all png images
#     :param every_nth: only use every n_th image
#     :param debug: preview the matched chess patterns
#     :param chessboard_grid_size: size of chess pattern
#     :return:
#     """

#     x, y = chessboard_grid_size

#     # termination criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((y * x, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

#     # Arrays to store object points and image points from all the images.
#     objpoints = []  # 3d point in real world space
#     imgpoints = []  # 2d points in image plane.

#     images = glob.glob(f'{image_path}/*.png')[::every_nth]

#     found = 0
#     for fname in images:
#         img = cv2.imread(fname)  # Capture frame-by-frame
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Find the chess board corners
#         ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

#         # If found, add object points, image points (after refining them)
#         if ret == True:
#             objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
#             corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#             imgpoints.append(corners2)

#             found += 1

#             if debug:
#                 # Draw and display the corners
#                 img = cv2.drawChessboardCorners(img, chessboard_grid_size, corners2, ret)
#                 cv2.imshow('img', img)
#                 cv2.waitKey(100)

#     print("Number of images used for calibration: ", found)

#     # When everything done, release the capture
#     cv2.destroyAllWindows()

#     # calibration
#     rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#     print('rms', rms)

#     # transform the matrix and distortion coefficients to writable lists
#     data = {
#         'rms': np.asarray(rms).tolist(),
#         'camera_matrix': np.asarray(mtx).tolist(),
#         'dist_coeff': np.asarray(dist).tolist()
#     }

#     # and save it to a file
#     with open("calibration_matrix.yaml", "w") as f:
#         yaml.dump(data, f)

#     print(data)


# if __name__ == '__main__':
#     # 1. record video
#     record_video(width=1280, height=720, fps=30)
#     # 2. split video into frames e.g. `ffmpeg -i 2021-10-15_10:30:00.mp4 -f image2 frames/video_01-%07d.png` and delete blurry images
#     # 3. run calibration on images
#     calibration('./frames', 30, debug=True)

import glob
import cv2
import numpy as np
import yaml
import subprocess
from datetime import datetime
from webcam import WebcamSource

def record_video(width: int, height: int, fps: int) -> str:
    """웹캠 녹화 후 MP4 파일 저장"""
    video_filename = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4'
    source = WebcamSource(width=width, height=height, fps=fps, buffer_size=10)
    video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
    
    print("🎥 녹화 시작! (q를 눌러 종료)")
    for idx, frame in enumerate(source):
        video_writer.write(frame)
        source.show(frame, only_print=idx % 5 != 0)
    
    print(f"✅ 녹화 완료: {video_filename}")
    return video_filename  # MP4 파일 경로 반환

def extract_frames(video_path: str, output_folder: str) -> None:
    """FFmpeg를 사용해 MP4를 프레임 이미지로 변환"""
    print(f"📸 프레임 추출 중... {video_path} → {output_folder}")
    subprocess.run(["mkdir", "-p", output_folder])
    subprocess.run(["ffmpeg", "-i", video_path, "-f", "image2", f"{output_folder}/frame_%07d.png"])
    print("✅ 프레임 추출 완료!")

def calibration(image_path, every_nth: int = 1, debug: bool = False, chessboard_grid_size=(7, 7)):
    """체커보드 패턴을 인식하여 카메라 캘리브레이션 진행"""
    x, y = chessboard_grid_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((y * x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    images = glob.glob(f'{image_path}/*.png')[::every_nth]
    found = 0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            found += 1

            if debug:
                img = cv2.drawChessboardCorners(img, chessboard_grid_size, corners2, ret)
                cv2.imshow('Calibration', img)
                cv2.waitKey(100)

    cv2.destroyAllWindows()
    print(f"📊 사용된 캘리브레이션 이미지 개수: {found}")

    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(f'📏 RMS Error: {rms}')

    data = {'rms': rms, 'camera_matrix': mtx.tolist(), 'dist_coeff': dist.tolist()}
    with open("calibration_matrix2.yaml", "w") as f:
        yaml.dump(data, f)
    
    print("✅ 캘리브레이션 완료! → calibration_matrix.yaml 저장됨.")

if __name__ == '__main__':
    video_path = record_video(width=1280, height=720, fps=30)  # 1️⃣ 녹화
    extract_frames(video_path, "./frames")                     # 2️⃣ MP4 → 프레임 변환
    calibration('./frames', every_nth=30, debug=True)          # 3️⃣ 캘리브레이션 진행
