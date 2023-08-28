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

When the modules are 1 : 3 : 3 : 1 : 1 for inference, the process time of audios are:
```
object_detection:          [13.755, 14.695, 5.131, 9.726, 20.354, 25.487, 6.726, 7.12, 4.979, 14.983, 2.593, 8.596, 16.668, 5.65, 5.951] # each video

license_plate_recognition: [0.798, 0.042, 0.034, 0.039, 0.044, 0.036, 0.038, 0.043, 0.048, 0.031, 0.032, 0.044, 0.033, 0.044, 0.028, 0.953, 0.036, 0.037, 0.038, 0.038, 0.034, 0.044, 0.042, 0.044, 0.034, 0.045, 0.036, 0.044, 0.032, ...] # each frame with box

face_recognition:          [0.029, 0.031, 0.021, 0.016, 0.017, 0.017, 0.018, 0.018, 0.015, 0.013, 0.011, 0.018, 0.012, 0.015, 0.020, 0.015, 0.015, 0.020, 0.015, 0.014, 0.014, 0.020, 0.013, 0.012, 0.014, 0.011, 0.024, 0.017, 0.011, ...] # each frame with box
```

![Image](https://github.com/lifang535/traffic_monitoring_1/blob/main/latency.png)
