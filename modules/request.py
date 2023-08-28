import numpy as np

class Request(object):
    def __init__(self,                  # take request_1-2(2)-1(2) as an example  [r_1-1-1, r_1-1-2, r_1-2-1, r_1-2-2]
                 ids: list,             # [1, 2] [1, 2] [1, 2] [1, 2]
                 sub_requests: list,    # [2, 2]
                 data: np.ndarray,      # image
                 # path: str,             # path
                 box: list,
                 label: str,
                 signal: int,
                 start_time: float) -> None:
        self.ids = ids
        self.sub_requests = sub_requests
        self.data = data
        # self.path = path
        self.box = box
        self.label = label
        self.signal = signal
        self.start_time = start_time

    def copy(self):
        return Request(ids=self.ids.copy(), sub_requests=self.sub_requests.copy(), data=self.data, box=self.box, label=self.label, signal=self.signal, start_time=self.start_time)

# request = Request(ids=[1], sub_requests=[], data=None, path=None, start_time=123)

# print(request.ids)
# request_copy = request.copy()
# print(request_copy.ids)
# request_copy.ids.append(2)
# print(request_copy.ids)
# print(request.ids)

# send -> m1

# m1:
# for i in range(2):
#     sub_request = request.copy()
#     sub_request.ids.append(i) # [1, 2, 3, 3] / [1, 3, 1, 1, 1] / [1, 4, 2, 2, 2, 2]
#     sub_request.sub_requests.append(2) # [2, 3, 4]


# import numpy as np

# class Request(object):
#     def __init__(self, request_id: str, sub_requests: list, data: np.ndarray, box: list, label: str, start_time: float) -> None:
#         self.request_id = request_id
#         self.sub_requests = sub_requests
#         self.data = data
#         self.box = box
#         self.label = label
#         self.start_time = start_time

# request_1 = Request(request_id="r_1", sub_requests=["r_1-1", "r_1-2"], data=None, path=None, box=None, label=None, start_time=0.0)
# request_1_1 = Request(request_id="r_1-1", sub_requests=["r_1-1-1", "r_1-1-2"], data=None, path=None, box=None, label=None, start_time=0.0)

# print(request_1.request_id)     # 输出: r_1
# print(request_1_1.request_id)   # 输出: r_1-1
