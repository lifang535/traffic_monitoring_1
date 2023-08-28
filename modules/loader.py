import os
import time
import random
import multiprocessing

from logger import logger_loader
from request import Request

class Loader(multiprocessing.Process):
    def __init__(self, input_video_paths_list):
        super().__init__()
        self.input_video_paths_list = input_video_paths_list

    def run(self):
        print("[Loader] start")
        logger_loader.info("[Loader] start")
        input_video_dir = "../input_videos"
        input_video_paths = [os.path.join(input_video_dir, filename) for filename in os.listdir(input_video_dir)]
        input_video_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        print(f"[Loader] input_video_paths: {input_video_paths}")

        for _ in range(10):
            input_video_paths.pop(-1)

        # input_video_paths = [input_video_paths[4]]

        for input_video_path in input_video_paths:
            request_2 = Request(ids=[int(input_video_path.split('_')[-1].split('.')[0])], \
                    sub_requests=[],
                    data=None,
                    box=None,
                    label=None,
                    signal=None,
                    start_time=time.time())

            # print(f"[Loader] request_id: {request.request_id}")
            self.input_video_paths_list.put(request_2)
            logger_loader.info(f"[Loader] input_video_path: {input_video_path}")

            # logger_lantency.info(f"{input_video_path.split('/')[-1].split('.')[0]} start at {time.time()}")
            # time.sleep(random.randint(2, 4))
            time.sleep(1)
        
        while True:
            time.sleep(0.01)
            if self.input_video_paths_list.qsize() == 0:
                print("[Loader] end")
                logger_loader.info("[Loader] end")
                request = Request(ids=None, \
                        sub_requests=None,
                        data=None,
                        box=None,
                        label=None,
                        signal=-1,
                        start_time=time.time())
                self.input_video_paths_list.put(request)
                break
