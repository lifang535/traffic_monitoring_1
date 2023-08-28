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
model: cv2.CascadeClassifier('haarcascade_frontalface_default.xml') from ?.

`traffic_summary`: input: draw_message; output: processed_video.

![Image](https://github.com/lifang535/traffic_monitoring_1/blob/main/app.png)

### Request in data transmission:

1 * video : n * frame : n * m * draw_message

### Throughout of model:

![Image](https://github.com/lifang535/traffic_monitoring_1/blob/main/latency.png)
