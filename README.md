# AMB82 Mini YOLO Person Height Estimation

This project combines an AMB82 Mini camera, a person-only YOLOv7-tiny model,
and an HC-SR04 ultrasonic sensor to estimate a person's height. It provides a
browser camera stream, on-screen bounding boxes, distance and height data, and
Serial Monitor diagnostics.

## Features

- Person-only YOLOv7-tiny inference on AMB82 Mini
- One displayed primary-person bounding box
- Lightweight multi-person association and bounding-box EMA smoothing
- Five-sample ultrasonic median filter with a configurable distance offset
- Height estimation calibrated from 1280x720 person measurements
- Confidence, distance, bounding-box height, and estimated height on the web UI
- H.264 RTSP output and an MJPEG browser stream
- Invalid-result handling for no echo and multiple-person scenes

## HC-SR04 wiring

| HC-SR04 | AMB82 Mini |
| --- | --- |
| VCC | V_USB 5 V |
| GND | GND |
| TRIG | GPIO 4 PF11 |
| ECHO | GPIO 5 PF12 through a 1 kOhm and 2 kOhm voltage divider |

The HC-SR04 ECHO output is 5 V. Do not connect it directly to GPIO 5.

## Setup

1. Install the AmebaPro2 Arduino board package.
2. Copy `config.example.h` to `config.h`.
3. Enter the Wi-Fi SSID and password in the local `config.h` file.
4. Keep `yolov7_tiny.nb` in the sketch folder.
5. Open `mid.ino` and select the AMB82 Mini board.
6. Select the flash model source option and upload the sketch.
7. Open Serial Monitor at 115200 baud to obtain the board IP address.

`config.h` is intentionally ignored by Git because it contains local Wi-Fi
credentials. Never commit that file.

## Calibration

The online height calculation is:

```text
height_cm = bbox_height_at_720p *
            (0.001374586854 * distance_cm + 0.022063368530)
```

The calibrated distance range is 180 to 300 cm. The ultrasonic reading is
filtered with a five-sample median and then increased by 5 cm. Measurements
outside the calibrated range are extrapolations.

The calibration spreadsheet and the measurement-accuracy analysis are included
in this repository. The current calibration used manually checked full-person
boxes and includes shoes, so live YOLO results may have additional error.

## Web endpoints

- Dashboard: `http://BOARD_IP/`
- JSON data: `http://BOARD_IP/data`
- Browser camera stream: `http://BOARD_IP:81/stream`

