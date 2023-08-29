# traffic_monitoring_1
This is a test of multi-model_app.

## code logics

### Five modules: 

`Clients`: output: video_path

`object_detection`: input: video_path; output: frame, box;
model: hustvl/yolos-tiny from https://huggingface.co/hustvl/yolos-tiny.

`license_plate_recognition`: input: frame, box; output: frame, box, license_plate_number, score;
model: EasyOCR from https://github.com/JaidedAI/EasyOCR.

`face_recognition`: input: frame, box; output: frame, box, person_name, score;
model: cv2.CascadeClassifier('haarcascade_frontalface_default.xml').

`traffic_summary`: input: draw_message; output: processed_video.

![Image](https://github.com/lifang535/traffic_monitoring_1/blob/main/app.png)

Test videos are in: (consider changing videos).

### Request in data transmission:

1 * video : n * frame : n * m * draw_message

```
         path                         num_frame                         num_car                         num_person
'input_videos/video_1.mp4'              150                               327                              459
'input_videos/video_2.mp4'              240                               39                               166
'input_videos/video_3.mp4'              90                                1003                             1
'input_videos/video_4.mp4'              248                               24                               4
'input_videos/video_5.mp4'              301                               2777                             0
'input_videos/video_6.mp4'              636                               1326                             1047
'input_videos/video_7.mp4'              162                               831                              88
'input_videos/video_8.mp4'              184                               2                                0
'input_videos/video_9.mp4'              115                               0                                95
'input_videos/video_10.mp4'             365                               0                                0
'input_videos/video_11.mp4'             60                                0                                11
'input_videos/video_12.mp4'             249                               0                                0
'input_videos/video_13.mp4'             264                               0                                22
'input_videos/video_14.mp4'             143                               0                                109
'input_videos/video_15.mp4'             176                               0                                0
```

### Throughout of model:

When the modules are 1 : 1 : 1 : 1 : 1 for inference, the process time of requests are:
```
object_detection:          [..., 0.103, 0.123, 0.117, 0.130, 0.106, 0.110, 0.093, 0.111, 0.109, 0.094, 0.072, 0.093, 0.093, 0.082, 0.099, 0.085, 0.104, 0.095, 0.084, 0.065, ...] # each frame

license_plate_recognition: [..., 0.038, 0.040, 0.036, 0.028, 0.037, 0.029, 0.035, 0.036, 0.037, 0.030, 0.040, 0.033, 0.034, 0.031, 0.030, 0.031, 0.032, 0.037, 0.036, 0.030, ...] # each frame with box

face_recognition:          [..., 0.058, 0.047, 0.046, 0.054, 0.044, 0.050, 0.051, 0.048, 0.047, 0.046, 0.042, 0.049, 0.047, 0.047, 0.047, 0.049, 0.043, 0.048, 0.056, 0.046, ...] # each frame with box
```

Throughout:
```
throughout of object_detection ≈ 10.16 req/s (cuda:1  MEM: 1688MiB; UTL: 0 ~ 20％, avg-8％)

throughout of license_plate_recognition ≈ 29.41 req/s (cuda:0  MEM: 4054MiB; UTL: 0 ~ 10％, avg-3％)

throughout of face_recognition ≈ 20.73 req/s (cpu  UTL: 10 ~ 40％， avg-15％)
```

### Latency:

When the modules are 1 : 1 : 1 : 1 : 1 for inference, there is request heap in `license_plate_recognition` and `face_recognition`.

![Image](https://github.com/lifang535/traffic_monitoring_1/blob/main/latency.png)
