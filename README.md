# AMB82 Mini YOLO Person Height Estimation

This project combines an AMB82 Mini camera, a person-only YOLOv7-tiny model,
and a TOF400H infrared ToF sensor to estimate a person's height. It provides a
browser camera stream, on-screen bounding boxes, filtered distance and height,
and Serial Monitor diagnostics.

## Features

- Person-only YOLOv7-tiny inference on AMB82 Mini
- One displayed primary-person bounding box
- Lightweight multi-person association and bounding-box EMA smoothing
- TOF400H UART/Modbus distance input with a five-sample median filter
- Height estimation calibrated from 1280x720 person measurements
- Height calculation limited to once per second with a five-sample median filter
- Last valid distance and height values retained for three seconds during short interruptions
- Confidence, distance, bounding-box height, and estimated height on the web UI
- H.264 RTSP output and an MJPEG browser stream
- Optional HW-493 laser aiming indicator on GPIO 7
- Invalid-result handling for missing distance data and multiple-person scenes

## TOF400H wiring

The sketch uses the module's default 115200-bps UART/Modbus RTU mode. UART
signals must be crossed.

| TOF400H | AMB82 Mini |
| --- | --- |
| VIN | V_USB 5 V |
| GND | GND |
| TX | GPIO 18 / Serial2 RX |
| RX | GPIO 19 / Serial2 TX |
| SDA | Not connected in UART mode |
| SCL | Not connected in UART mode |

The firmware verifies Modbus CRC, reads distance register `0x0010`, and enables
the 4 m long-range mode only when register `0x0004` is not already set to 1.
All modules must share GND. Keep the module's UART signal levels compatible
with the AMB82 Mini; use a divider or level shifter if a module variant outputs
5 V logic on TX.

## HW-493 laser wiring

The laser is an aiming light only; TOF400H still performs the distance
measurement. The sketch uses GPIO 7 and turns the laser on only after the
system is ready. Continuous mode re-applies the GPIO output once per second in
case another SDK component changes the pin mode. GPIO7 is used because GPIO6
also has an alternate LOG_TX function in the AMB82 board package. Use a transistor driver
instead of powering the laser from the GPIO pin.

| HW-493 / driver | AMB82 Mini |
| --- | --- |
| S and middle `+` pin | V_USB 5 V |
| `-` pin | NPN collector |
| NPN emitter | GND |
| NPN base | GPIO 7 through 1 kOhm |

A 10 kOhm resistor from the NPN base to GND is recommended. All modules must
share the same ground. Set `LASER_MODULE_ENABLE` or
`LASER_ON_WHEN_SYSTEM_READY` to `0` in `config.h` to disable it.
`LASER_CONTINUOUS_MODE` controls the periodic GPIO keepalive.

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

The calibrated distance range is 180 to 300 cm. TOF400H distance uses a
five-sample median filter. Height is calculated once per second and passed
through a separate five-sample median filter. Low-confidence boxes are not
added to the height window. When the sensor read or person track is briefly
interrupted, the filter state resets but the last valid displayed value remains
for three seconds. Measurements outside the calibrated range are extrapolations.

The calibration spreadsheet and the measurement-accuracy analysis are included
in this repository. The current calibration used manually checked full-person
boxes and includes shoes, so live YOLO results may have additional error.

## Web endpoints

- Dashboard: `http://BOARD_IP/`
- JSON data: `http://BOARD_IP/data`
- Browser camera stream: `http://BOARD_IP:81/stream`

The JSON response includes `distanceCm`, bounding-box data, confidence,
the raw height sample, the median-filtered height, and height-filter status.

