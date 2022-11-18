import os
import time

import cv2
import numpy as np


class VideoProcessor:
    """
    Extracting frames from video
    TODO:
        - Processing live video
    """
    def __init__(self, video_file_dir, output_dir, downscale=True, live=False):
        self.downscale = downscale
        self._video_file_dir = video_file_dir
        self._output_dir = output_dir
        self._output_frames_dir = self._output_dir + "tmp/frames/"
        # Init
        self._read_video(self._video_file_dir)
        self._init_output_dirs()
        if not live:
            self._print_video_summary()

    def extract_all_frames_from_video(self, capture_interval, save_frames=False):
        capture_interval = max(1 / self._fps, capture_interval)
        count, sec = 0, 0
        time_start = time.time()
        print("Number of frames in the video: {}".format(self._n_frames))
        print("Converting video... \n")
        success = self._has_frame(sec)
        all_frames = []
        while success:
            success = self._has_frame(sec)
            if success:
                frame = self._get_frame(sec)
                if save_frames:
                    self._save_frame(frame, str(sec))
                else:
                    all_frames.append(frame)
            sec += capture_interval
            sec = round(sec, 2)
            count += 1
        print("Conversion completed in {} seconds".format(
            time.time() - time_start)
        )
        if not save_frames:
            return all_frames

    # def clip_video_and_extract_frames(
    #     self, start_time, end_time,
    #     capture_interval, save_frames=False
    # ):
    #     capture_interval = max(1 / self._fps, capture_interval)
    #     # Clip the video
    #     clipped_video_dir = self._output_video_dir + "clipped_video.mp4"
    #     ffmpeg_extract_subclip(
    #         self._video_file_dir, start_time, end_time,
    #         targetname=clipped_video_dir
    #     )
    #     # Extract frames
    #     self._read_video(clipped_video_dir)
    #     frames = self.extract_all_frames_from_video(capture_interval)
    #     # Delete the clipped video
    #     os.remove(clipped_video_dir)
    #     return [f for f in frames if f is not None]

    def apply_processing_to_frames(self, frames, proc_fn):
        proc_frames = []
        for frame in frames:
            proc_frames.append(proc_fn(frame))
        return proc_frames

    def _has_frame(self, sec):
        self._cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        has_frame, _ = self._cap.read()
        return has_frame

    def _get_frame(self, sec):
        if self._has_frame(sec):
            _, frame = self._cap.read()
            if self.downscale:
                return self._downscale_frame(frame)
            else:
                return frame

    def _rotate_frame(self, frame):
        return np.rot90(frame)

    def _downscale_frame(self, frame, dim=(54, 54)):
        """Downgrades to 320x240 by default.
        """
        return cv2.resize(frame, dim)

    def _save_frame(self, frame, frame_name):
        frame_out_dir = self._output_frames_dir + \
            "{}_frame.jpg".format(frame_name)
        cv2.imwrite(frame_out_dir, frame)

    def _read_video(self, video_file_dir):
        # Read video and initialize properties
        self._cap = cv2.VideoCapture(video_file_dir)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        self._cap_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._cap_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def _print_video_summary(self):
        print("Video properties")
        print("Frame rate: {}".format(self._fps))
        print("Total frames in the video: {}".format(self._n_frames))
        print("Height: {} | Width: {}".format(self._cap_height, self._cap_width))

    def _init_output_dirs(self):
        try:
            # os.makedirs(self._output_video_dir)
            os.makedirs(self._output_frames_dir)
        except OSError:
            pass