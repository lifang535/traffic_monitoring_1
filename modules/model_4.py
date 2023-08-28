import os
import time
import threading
import multiprocessing
from PIL import Image, ImageDraw, ImageFont

import cv2
from logger import logger_model_4, logger_model_4_rate, logger_latency

model_4_lock = multiprocessing.Lock()

# Load a font with a larger size
font_size = 16
font = ImageFont.truetype("../fonts/times new roman.ttf", font_size)

class Model_4(multiprocessing.Process):
    def __init__(self, id, draw_message_list, end_signal, to_monitor_rate, table):
        super().__init__()
        self.id = id
        self.draw_message_list = draw_message_list
        self.end_signal = end_signal

        # self.lock = threading.Lock()
        # self.frame_files_to_be_processed = frame_files_to_be_processed

        self.timer_logger_model_4 = time.time()
        self.to_monitor_rate = to_monitor_rate

        self.draw_messages_check = {} # dict() {} table
        # self.draw_messages_check = {
        #     "reuqest_id": [request, request...], # until len(value) == request.sub_requests[1]
        #     ...
        # }
        self.frames_check = {}
        # self.frames_check = {
        #     video_id: num_frames, # until num_frames == request.sub_requests[0]
        #     ...
        # }

        self.threads = []
        self.threading_lock = threading.Lock()

    def run(self):
        print(f"[Model_4_{self.id}] start")
        logger_model_4.info(f"[Model_4_{self.id}] start")

        self.end_signal.value += 1

        # if self.id == 1:
        #     # thread_monitor_rate = threading.Thread(target=self.monitor_rate)
        #     # thread_monitor_rate.start()

        # threads = []

        while True:
            time.sleep(0.01)
            with model_4_lock:
                request = self.draw_message_list.get()
                if request.signal == -1:
                    for thread in self.threads: # !!!
                        thread.join()
                    self.threads = []
                    if self.draw_messages_check != {} or self.frames_check != {}:
                        self.draw_message_list.put(request) # put the end signal back
                        print(f"[Model_4_{self.id}] self.draw_message_list: {self.draw_message_list.qsize()}")
                        print(f"[Model_4_{self.id}] self.draw_messages_check: {self.draw_messages_check}")
                        print(f"[Model_4_{self.id}] self.frames_check: {self.frames_check}")
                        continue

                    # if not self.draw_message_list.qsize() == 0:
                    self.draw_message_list.put(request) # put the end signal back
                    self.end_signal.value -= 1 # 0.5 (-1 will come in twice)

                    logger_model_4.info(f"[Model_4_{self.id}] self.draw_message_list: {self.draw_message_list.qsize()}")
                    print(f"[Model_4_{self.id}] end")
                    logger_model_4.info(f"[Model_4_{self.id}] end")
                    # print(f"[Model_4_{self.id}] self.end_signal.value: {self.end_signal.value}")
                    logger_model_4.info(f"[Model_4_{self.id}] self.end_signal.value: {self.end_signal.value}")
                    if self.end_signal.value == 0:
                        # self.frame_files_to_be_processed.append(request.signal)
                        print(f"[Model_4_{self.id}] self.draw_messages_check = {self.draw_messages_check}, and self.frames_check = {self.frames_check}") # !!!
                    break
                
                # [1, 2, 3] -> [1, 2]
                if time.time() - self.timer_logger_model_4 > 5:
                    # print(f"[Model_4_{self.id}] {draw_message[0]} is being processed, and draw_message_list: {len(self.draw_message_list)}")
                    logger_model_4.info(f"[Model_4_{self.id}] {request.ids} is being processed, and draw_message_list: {self.draw_message_list.qsize()}")
                    self.timer_logger_model_4 = time.time()
                
                # self.process_draw_message(draw_message)
                if f"{request.ids[0]}-{request.ids[1]}]" in self.draw_messages_check:
                    self.draw_messages_check[f"{request.ids[0]}-{request.ids[1]}]"].append(request)
                else:
                    self.draw_messages_check[f"{request.ids[0]}-{request.ids[1]}]"] = [request]

                if len(self.draw_messages_check[f"{request.ids[0]}-{request.ids[1]}]"]) >= request.sub_requests[1]:
                    thread = threading.Thread(target=self.process_requests, args=(self.draw_messages_check[f"{request.ids[0]}-{request.ids[1]}]"],))
                    self.threads.append(thread)
                    thread.start()
                    if len(self.threads) > 16:
                        self.threads[0].join()
                        self.threads.pop(0)
                    self.draw_messages_check.pop(f"{request.ids[0]}-{request.ids[1]}]")

    
    def process_requests(self, requests):
        for request in requests:
            self.process_draw_message(request)

        with self.threading_lock:
            if request.ids[0] in self.frames_check:
                self.frames_check[request.ids[0]] += 1
            else:
                self.frames_check[request.ids[0]] = 1

            if self.frames_check[request.ids[0]] >= request.sub_requests[0]:
                self.process_video(f"output_frames_video_{request.ids[0]}")
                self.frames_check.pop(request.ids[0])

    def monitor_rate(self):
        rates = []
        sliding_window_size = 10
        last_draw_message = ""
        last_draw_messages_list_len = 0
        while True:
            time.sleep(1e-6)
            with model_4_lock:
                if self.end_signal.value == 0:
                    break
                try:
                    if (len(self.draw_message_list) > 0 and self.draw_message_list[-1][0] != last_draw_message) or len(self.draw_message_list) > last_draw_messages_list_len:
                        self.to_monitor_rate.append(time.time())
                        last_draw_message = self.draw_message_list[-1][0]
                    last_draw_messages_list_len = len(self.draw_message_list)
                except Exception as e:
                    # logger_model_4.warning(f"[Model_4_{self.id}] {e}, and draw_message_list[-1]: {self.draw_message_list[-1]}, and last_draw_message: {last_draw_message}")
                    ...

                if len(self.to_monitor_rate) > 1:
                    rate = round((len(self.to_monitor_rate) - 1) / (self.to_monitor_rate[-1] - self.to_monitor_rate[0]), 3)
                    rates.append(rate)
                    if len(rates) > sliding_window_size:
                        rates.pop(0)
                    total_weight = sum(range(1, len(rates) + 1))
                    weighted_sum = sum((i + 1) * rate for i, rate in enumerate(rates))
                    moving_average = round(weighted_sum / total_weight, 3)
                    # print(f"[Model_4_{self.id}] rate: {moving_average}")
                    logger_model_4.info(f"[Model_4_{self.id}] rate: {moving_average}")
                    logger_model_4_rate.info(f"{moving_average}")
                    self.to_monitor_rate[:] = self.to_monitor_rate[-1:]

    def process_draw_message(self, request):
        label, box, image_array = request.label, request.box, request.data # output_frames_video_1/frame_6.jpg
        output_frame_filename = f"output_frames_video_{request.ids[0]}/frame_{request.ids[1]}.jpg"

        # Draw bounding boxes on the image
        try:
            if os.path.exists(output_frame_filename): # TODO: frame_filename 的命名规则
                image = Image.open(output_frame_filename)
            else:
                image = Image.fromarray(image_array)
                os.makedirs(os.path.dirname(output_frame_filename), exist_ok=True)
                image.save(output_frame_filename)
                
            if box is None: # TODO
                image.save(output_frame_filename)
                # if file_count >= request.sub_requests[0] and request.ids[2] == request.sub_requests[1]:
                #     self.frame_files_to_be_processed.append(request.ids[0])
                #     self.frame_files_to_be_processed.append(request.ids[0])
                return

            draw = ImageDraw.Draw(image)

            # label_text = f"{label} {round(score * 100, 1)}%"
            if not "Mr." in label:
            # if label == "car":
                draw.rectangle(box, outline="green", width=3)
                draw.text((box[0], box[1]), label, fill="red", font=font)
            elif label == "person":
                draw.rectangle(box, outline="blue", width=3)
                draw.text((box[0], box[1]), label, fill="yellow", font=font)
            else:
                draw.rectangle(box, outline="blue", width=3)
                draw.text((box[0], box[1]), label, fill="yellow", font=font)

            # Save the annotated image
            image.save(output_frame_filename)
            
        except Exception as e:
            logger_model_4.error(f"[Model_4_{self.id}] {e}")

        # output_frame_dir = f"output_frames_video_{request.ids[0]}"

    def process_video(self, frame_filename):
        # frame_filename = f"output_{frame_filename}"
        print(f"[Model_4_{self.id}] frame_filename: ", frame_filename)
        logger_model_4.info(f"[Model_4_{self.id}] frame_filename: {frame_filename}")
        video_id = frame_filename.split('_')[-1]
        # Input video file path
        input_video_path = f"../input_videos/video_{video_id}.mp4"
        print(f"[Model_4_{self.id}] input_video_path: ", input_video_path)
        logger_model_4.info(f"[Model_4_{self.id}] input_video_path: {input_video_path}")

        # Output directory for frames
        output_frames_dir = frame_filename
        output_video_dir = "../output_videos"
        os.makedirs(output_video_dir, exist_ok=True)

        # Output video file path
        output_video_path = f"../output_videos/processed_video_{video_id}.mp4"
        print(f"[Model_4_{self.id}] output_video_path: ", output_video_path)
        logger_model_4.info(f"[Model_4_{self.id}] output_video_path: {output_video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Combine the processed frames back into a video
        output_frames = [os.path.join(output_frames_dir, filename) for filename in os.listdir(output_frames_dir)]
        output_frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        for frame_path in output_frames:
            frame = cv2.imread(frame_path)
            output_video.write(frame)
        
        # Release the output video writer
        output_video.release()

        logger_latency.info(f"video_{video_id} end at {time.time()}")
        
        # Clean up: Delete the processed frames
        for frame_path in output_frames:
            os.remove(frame_path)

        # Clean up: Delete the frames directory
        os.rmdir(output_frames_dir)
        
        print(f"[Model_4_{self.id}] {input_video_path} processed successfully")
        logger_model_4.info(f"[Model_4_{self.id}] {input_video_path} processed successfully")    

# TODO: combine 4, 5 and remove special signal except -1
# TODO: change models in 2, 3
# TODO: monitor rate
