import os
import cv2
import json
import datetime
import numpy as np
import time
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from rich import print

class EpisodeWriter():
    def __init__(
        self,
        task_dir,
        frequency=30,
        image_shape=(480, 640, 3),
        data_keys=['rgb'],
        save_video: bool = False,
        video_fps: float = 0.0,
        video_codec: str = "mp4v",
    ):
        """
        image_shape: [width, height, channel]
        state_shape: [29]
        action_shape: [29]
        """
        print("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.frequency = frequency
        self.image_shape = image_shape
        self.data_keys = data_keys
        self.save_video = bool(save_video)
        self.video_fps = float(video_fps) if float(video_fps) > 1e-6 else float(frequency)
        self.video_codec = str(video_codec or "mp4v")
        
        self.data = {}
        self.episode_data = []
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            print(f"==> task_dir already exists, episode_id: {self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            print("==> episode directory does not exist, creating one.\n")

        # Optional: video output root under task_dir
        self.videos_dir = os.path.join(self.task_dir, "videos")
        if self.save_video:
            os.makedirs(self.videos_dir, exist_ok=True)
        self.data_info()
        self.text_desc()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(maxsize=100)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        print("==> EpisodeWriter initialized successfully.\n")

    def data_info(self, version='1.0.0', date=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "YanjieZe",
                "image": {"width":self.image_shape[0], "height":self.image_shape[1], "fps":self.frequency},
            }
        
    def text_desc(self, goal="pick up the red cup on the table.", 
                  desc="Pick up the cup from the table and place it in another position. The operation should be smooth and the water in the cup should not spill out",
                  steps="step1: searching for cups. step2: go to the target location. step3: pick up the cup"):
        self.text = {
            "goal": goal,
            "desc": desc,
            "steps":steps,
        }
 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            print("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.episode_data = []
        self.episode_id = self.episode_id + 1
        

        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        os.makedirs(self.episode_dir, exist_ok=True)
        
        if "rgb" in self.data_keys:
            self.rgb_dir = os.path.join(self.episode_dir, 'rgb')
            os.makedirs(self.rgb_dir, exist_ok=True)
            print(f"==> rgb_dir: {self.rgb_dir}")
       

        self.json_path = os.path.join(self.episode_dir, 'data.json')

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        print(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, data_dict):
        # Increment the item ID
        self.item_id += 1
        # Enqueue the item data
        self.item_data_queue.put(data_dict)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    print(f"[EpisodeWriter] [WARN] Failed to process item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']

        # vision
        rgb = item_data.get('rgb', None)

        # body and hand state
        state_body = item_data.get('state_body', None)
        state_hand_left = item_data.get('state_hand_left', None)
        state_hand_right = item_data.get('state_hand_right', None)

        # body and hand action
        action_body = item_data.get('action_body', None)

        # human data
        human_data = item_data.get('human_data', None)

        # retarget data
        retarget_data = item_data.get('retarget_data', None)

        # low level action
        action_low_level = item_data.get('action_low_level', None)

        # Save images
        if rgb is not None:
            color_name = f'{str(idx).zfill(6)}.jpg'
            save_path = os.path.join(self.rgb_dir, color_name)
            if not cv2.imwrite(save_path, rgb):
                print(f"[EpisodeWriter] [WARN] Failed to save rgb image: {save_path}")
            item_data['rgb'] = str(Path(save_path).relative_to(Path(self.json_path).parent))
            
                
        # state and action are directly saved to the episode_data
        if state_body is not None:
            item_data['state_body'] = state_body
        if state_hand_left is not None:
            item_data['state_hand_left'] = state_hand_left
        if state_hand_right is not None:
            item_data['state_hand_right'] = state_hand_right

        if action_body is not None:
            item_data['action_body'] = action_body

        if human_data is not None:
            item_data['human_data'] = human_data
        if retarget_data is not None:
            item_data['retarget_data'] = retarget_data
        if action_low_level is not None:
            item_data['action_low_level'] = action_low_level

        # Save item_data to episode_data
        # Update episode data
        self.episode_data.append(item_data)

        current_record_time = time.time()
        print(f"==> episode_id:{self.episode_id}  item_id:{self.item_id}  current_time:{current_record_time}")

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.need_save = True  # Set the save flag
        print("==> Episode save started...")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        self.data['info'] = self.info
        self.data['text'] = self.text
        self.data['data'] = self.episode_data
        with open(self.json_path, 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(self.data, indent=4, ensure_ascii=False))

        # Optional: save an mp4 video for this episode under task_dir/videos/
        if self.save_video and ("rgb" in self.data_keys):
            try:
                self._save_episode_video()
            except Exception as e:
                print(f"[EpisodeWriter] [WARN] Failed to save episode video: {e}")
        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        print(f"==> Episode (length:{len(self.episode_data)}) saved successfully to {self.json_path}.")

    def _save_episode_video(self):
        if not hasattr(self, "episode_dir") or not hasattr(self, "rgb_dir"):
            return
        os.makedirs(self.videos_dir, exist_ok=True)
        out_path = os.path.join(self.videos_dir, f"episode_{str(self.episode_id).zfill(4)}.mp4")

        # Collect frame paths in order (prefer episode_data entries for alignment with idx)
        frame_paths = []
        for item in self.episode_data:
            p = item.get("rgb", None)
            if isinstance(p, str) and p:
                frame_paths.append(os.path.join(self.episode_dir, p))
        if not frame_paths:
            jpgs = sorted([p for p in os.listdir(self.rgb_dir) if p.lower().endswith(".jpg")])
            frame_paths = [os.path.join(self.rgb_dir, n) for n in jpgs]
        if not frame_paths:
            return

        first = None
        for p in frame_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None and img.ndim == 3:
                first = img
                break
        if first is None:
            return

        h, w = int(first.shape[0]), int(first.shape[1])
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        writer = cv2.VideoWriter(out_path, fourcc, float(self.video_fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter failed to open: {out_path}")

        n_written = 0
        for p in frame_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            writer.write(img)  # BGR
            n_written += 1
        writer.release()
        print(f"[EpisodeWriter] [OK] saved video: {out_path} ({n_written} frames @ {self.video_fps:.1f} fps)")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()