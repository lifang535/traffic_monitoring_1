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

### Request in data transmission:

1 * video : n * frame : n * m * draw_message

### Throughout of model:

When the modules are 1 : 1 : 1 : 1 : 1 for inference, the process time of requests are:
```
object_detection:          [..., 0.103, 0.123, 0.117, 0.130, 0.106, 0.110, 0.093, 0.111, 0.109, 0.094, 0.072, 0.093, 0.093, 0.082, 0.099, 0.085, 0.104, 0.095, 0.084, 0.065, ...] # each frame, ≈ 10 req/s

license_plate_recognition: [..., 0.038, 0.040, 0.036, 0.028, 0.037, 0.029, 0.035, 0.036, 0.037, 0.030, 0.040, 0.033, 0.034, 0.031, 0.030, 0.031, 0.032, 0.037, 0.036, 0.030, ...] # each frame with box, ≈ 3 req/s

face_recognition:          [..., 0.013, 0.017, 0.010, 0.008, 0.013, 0.012, 0.009, 0.012, 0.012, 0.011, 0.012, 0.015, 0.013, 0.010, 0.014, 0.012, 0.008, 0.015, 0.011, 0.010, ...] # each frame with box, ≈ 90 req/s

# face_recognition's size is small, trained by my own data.
```

### Latency:

When the modules are 1 : 1 : 1 : 1 : 1 for inference, there is request heap in `license_plate_recognition` and `face_recognition`.

![Image](https://github.com/lifang535/traffic_monitoring_1/blob/main/latency.png)
