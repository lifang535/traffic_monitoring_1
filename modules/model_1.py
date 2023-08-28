import os
import time
import threading
import multiprocessing

import cv2
import torch
import numpy as np
from PIL import Image

from transformers import YolosImageProcessor, YolosForObjectDetection
# from transformers import DetrImageProcessor, DetrForObjectDetection
from logger import logger_model_1, logger_latency, logger_model_1_rate
from request import Request

vehicle_classes = ["bicycle", "car", "motorbike", "bus", "truck"]
person_classes = ["person"]

model_1_lock = multiprocessing.Lock()

class Model_1(multiprocessing.Process):
    def __init__(self, id, input_video_paths_list, car_frames_list, person_frames_list, end_signal, to_monitor_rate):
        super().__init__()
        self.id = id
        self.input_video_paths_list = input_video_paths_list
        self.car_frames_list = car_frames_list
        self.person_frames_list = person_frames_list
        self.end_signal = end_signal
        
        self.device = None
        self.model = None
        self.processor = None

        self.timer_logger_model_1 = time.time()
        self.to_monitor_rate = to_monitor_rate

    def run(self):
        if self.id // 2 == 0:
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cuda:0")
        self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').to(self.device)
        self.processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

        # self.model = torch.load("models/yolos-tiny/yolos-tiny_model.pt").to(self.device)
        # self.processor = YolosImageProcessor.from_pretrained("models/yolos-tiny/yolos-tiny_image_processor")

        # self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)
        # self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        self.end_signal.value += 1

        # if self.id == 1:
        #     thread_monitor_rate = threading.Thread(target=self.monitor_rate)
        #     thread_monitor_rate.start()

        print(f"[Model_1_{self.id}] start")
        logger_model_1.info(f"[Model_1_{self.id}] start")
        while True:
            time.sleep(0.01)
            request = None
            with model_1_lock:
                # TODO: 缓存 to_monitor_rate，防堵塞开进程
                request = self.input_video_paths_list.get()
                if request.signal == -1:
                    self.input_video_paths_list.put(request) # put the end signal back
                    if self.car_frames_list.qsize() == 0 and self.person_frames_list.qsize() == 0:
                        print(f"[Model_1_{self.id}] end")
                        logger_model_1.info(f"[Model_1_{self.id}] end")
                        self.end_signal.value -= 1
                        # print(f"[Model_1_{self.id}] self.end_signal.value: {self.end_signal.value}")
                        logger_model_1.info(f"[Model_1_{self.id}] self.end_signal.value: {self.end_signal.value}")
                        if self.end_signal.value == 0:
                            self.car_frames_list.put(request) # TODO: 特殊信号 -> Request
                            self.person_frames_list.put(request)
                        break
                    else:
                        continue
                # input_video_path = self.input_video_paths_list.pop(0)
            if request is not None:
                # 等到上一个 output_frames_videos 被删除
                # while True:
                #     time.sleep(0.01)
                #     if not os.path.exists("output_frames_video_" + str(int(input_video_path.split('/')[-1].split('.')[0].split('_')[-1]) - 1)):
                #         print(f"[Model_1_{self.id}] {input_video_path.split('/')[-1].split('.')[0]} is being processed")
                #         break
                # logger_lantency.info(f"{request.path.split('/')[-1].split('.')[0]} start at {time.time()}")
                logger_latency.info(f"video_{request.ids[0]} start at {time.time()}")
                self.process_video(request)

    def monitor_rate(self): # 需要优化
        rates = []
        sliding_window_size = 1
        last_file_video_path = ""
        last_input_video_paths_list_len = 0
        while True:
            time.sleep(1e-6)
            with model_1_lock:
                if self.end_signal.value == 0:
                    break
                if (len(self.input_video_paths_list) > 0 and self.input_video_paths_list[-1] != last_file_video_path) or len(self.input_video_paths_list) > last_input_video_paths_list_len:
                    self.to_monitor_rate.append(time.time())
                    last_file_video_path = self.input_video_paths_list[-1]
                last_input_video_paths_list_len = len(self.input_video_paths_list)

                if len(self.to_monitor_rate) > 1:
                    rate = round((len(self.to_monitor_rate) - 1) / (self.to_monitor_rate[-1] - self.to_monitor_rate[0]), 3)
                    rates.append(rate)
                    if len(rates) > sliding_window_size:
                        rates.pop(0)
                    total_weight = sum(range(1, len(rates) + 1))
                    weighted_sum = sum((i + 1) * rate for i, rate in enumerate(rates))
                    moving_average = round(weighted_sum / total_weight, 3)
                    # print(f"[Model_1_{self.id}] rate: {moving_average}")
                    logger_model_1.info(f"[Model_1_{self.id}] rate: {moving_average}")
                    logger_model_1_rate.info(f"{moving_average}")
                    self.to_monitor_rate[:] = self.to_monitor_rate[-1:]
    
    def process_video(self, request):
        # Input video file path
        input_video_path = f"../input_videos/video_{request.ids[0]}.mp4"
        print(f"[Model_1_{self.id}] input_video_path: ", input_video_path)
        logger_model_1.info(f"[Model_1_{self.id}] input_video_path: {input_video_path}")

        # Output directory for frames
        output_frames_dir = "frames_" + input_video_path.split('/')[-1].split('.')[0]
        output_video_dir = "../output_videos"
        # if exists, delete it
        if os.path.exists(output_frames_dir):
            os.system("rm -rf " + output_frames_dir)
        # if os.path.exists(output_video_dir): # ？
            # os.system("rm -rf " + output_video_dir)
        os.makedirs(output_frames_dir, exist_ok=True)
        os.makedirs(output_video_dir, exist_ok=True)

        # Output video file path
        output_video_path = "../output_videos/processed_" + input_video_path.split('/')[-1]
        print(f"[Model_1_{self.id}] output_video_path: ", output_video_path)
        logger_model_1.info(f"[Model_1_{self.id}] output_video_path: {output_video_path}")

        # Open the video file
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        # fps = int(cap.get(5))

        # Define the codec and create a VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        request.sub_requests.append(frame_count)

        threads = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            time.sleep(0.01)

            # Save the processed frame as a JPG file
            frame_filename = os.path.join(output_frames_dir, f"frame_{int(cap.get(1))}.jpg")
            cv2.imwrite(frame_filename, frame)

            # Write the processed frame to the output video
            # out.write(frame)

            # frame = frame.tobytes()
            image_array = np.array(Image.open(frame_filename))

            # Process the frame (example: apply a filter)
            request_copy = request.copy()
            request_copy.ids.append(int(cap.get(1)))
            request_copy.data = image_array

            thread = threading.Thread(target=self.process_frame, args=(request_copy,))
            threads.append(thread)
            thread.start()
            if len(threads) > 16:
                threads[0].join()
                threads.pop(0)
            # self.process_frame(frame, output_frames_dir + "/frame_" + str(int(cap.get(1))) + ".jpg")

        # Release the video capture and writer objects
        cap.release()
        # out.release()

        for thread in threads:
            thread.join()
        threads = []

        # Clean up: Delete the processed frames
        os.system("rm -rf " + output_frames_dir)

    def process_frame(self, request):
        frame = request.data

        if time.time() - self.timer_logger_model_1 > 5:
            logger_model_1.info(f"[Model_1_{self.id}] request_id: {request.ids}")
            self.timer_logger_model_1 = time.time()

        image = Image.fromarray(frame)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}  # move input data to GPU
        with torch.no_grad():  # execute model inference, make sure we do not compute gradientss
            outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        car = False
        person = False
        index = 0
        num_drawn_boxes = 0

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            if self.model.config.id2label[label.item()] in vehicle_classes or self.model.config.id2label[label.item()] in person_classes:
                num_drawn_boxes += 1

        request.sub_requests.append(num_drawn_boxes)

        if num_drawn_boxes == 0:
            request_copy = request.copy()
            request_copy.ids.append(index)
            request_copy.box = None
            self.person_frames_list.put(request_copy)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            if self.model.config.id2label[label.item()] in vehicle_classes:
                # car = True
                index += 1
                request_copy = request.copy()
                request_copy.ids.append(index)
                request_copy.box = box
                self.car_frames_list.put(request_copy)
            if self.model.config.id2label[label.item()] in person_classes:
                # person = True
                index += 1
                request_copy = request.copy()
                request_copy.ids.append(index)
                request_copy.box = box
                self.person_frames_list.put(request_copy)

        # if car:
        #     self.car_frames_list.append([frame, frame_filename])
        # if person:
        #     self.person_frames_list.append([frame, frame_filename])

        del image, inputs, outputs, results

        return
    