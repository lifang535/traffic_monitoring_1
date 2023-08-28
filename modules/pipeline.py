import multiprocessing

from loader import Loader
from model_1 import Model_1
from model_2 import Model_2
from model_3 import Model_3
from model_4 import Model_4
from model_5 import Model_5

model_1_lock = multiprocessing.Lock()
model_2_lock = multiprocessing.Lock()
model_3_lock = multiprocessing.Lock()
model_4_lock = multiprocessing.Lock()
model_5_lock = multiprocessing.Lock()

vehicle_classes = ["car", "truck", "bus", "train", "motorcycle", "bicycle"]

person_classes = ["person"]

"""
def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)        """

def _pipeline():
    # setup_gpu()
    # multiprocessing.set_start_method('spawn', True)

    manager_1 = multiprocessing.Manager()
    manager_2 = multiprocessing.Manager()
    manager_3 = multiprocessing.Manager()
    manager_4 = multiprocessing.Manager()
    manager_5 = multiprocessing.Manager()

    manager_to_monitor_rate_1 = multiprocessing.Manager()
    manager_to_monitor_rate_2 = multiprocessing.Manager()
    manager_to_monitor_rate_3 = multiprocessing.Manager()
    manager_to_monitor_rate_4 = multiprocessing.Manager()
    manager_to_monitor_rate_5 = multiprocessing.Manager()

    input_video_paths_list = manager_1.Queue()
    car_frames_list = manager_2.Queue()
    person_frames_list = manager_3.Queue()
    draw_message_list = manager_4.Queue()

    to_monitor_rate_1 = manager_to_monitor_rate_1.list()
    to_monitor_rate_2 = manager_to_monitor_rate_2.list()
    to_monitor_rate_3 = manager_to_monitor_rate_3.list()
    to_monitor_rate_4 = manager_to_monitor_rate_4.list()
    to_monitor_rate_5 = manager_to_monitor_rate_5.list()

    end_signal_1 = manager_1.Value('i', 0)
    end_signal_2 = manager_2.Value('i', 0)
    end_signal_3 = manager_3.Value('i', 0)
    end_signal_4 = manager_4.Value('i', 0)
    end_signal_5 = manager_5.Value('i', 0)

    table = manager_1.dict()

    loader = Loader(input_video_paths_list)
    model_1s = []
    model_2s = []
    model_3s = []
    model_4s = []
    model_5s = []

    for i in range(5):
        model_1 = Model_1(i + 1, input_video_paths_list, car_frames_list, person_frames_list, end_signal_1, to_monitor_rate_1)
        model_2 = Model_2(i + 1, car_frames_list, draw_message_list, end_signal_2, to_monitor_rate_2)
        model_3 = Model_3(i + 1, person_frames_list, draw_message_list, end_signal_3, to_monitor_rate_3)
        model_4 = Model_4(i + 1, draw_message_list, end_signal_4, to_monitor_rate_4, table)
        # model_5 = Model_5(i + 1, frame_files_to_be_processed, end_signal_5, to_monitor_rate_5)
        model_1s.append(model_1)
        model_2s.append(model_2)
        model_3s.append(model_3)
        model_4s.append(model_4)
        # model_5s.append(model_5)

    loader.start()
    for i in range(1):
        model_1s[i].start()    
    for i in range(2):
        model_2s[i].start()
    for i in range(3):
        model_3s[i].start()
    for i in range(1):
        model_4s[i].start()
    # for i in range(1):
    #     model_5s[i].start()

    loader.join()
    for i in range(1):
        model_1s[i].join()
    for i in range(2):
        model_2s[i].join()
    for i in range(3):
        model_3s[i].join()
    for i in range(1):
        model_4s[i].join()
    # for i in range(1):
    #     model_5s[i].join()
        

if __name__ == "__main__":
    """
    # opt_run = torch.compile(run)
    run()
    """
    _pipeline()

# TODO: Queue、Request、While True、image_array、monitor_rate、print(.qsize())

