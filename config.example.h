#ifndef AMB82_PERSON_MONITOR_CONFIG_H
#define AMB82_PERSON_MONITOR_CONFIG_H

// Copy this file to config.h, then enter the local Wi-Fi credentials.
#define WIFI_AP_SSID       "YOUR_WIFI_SSID"
#define WIFI_AP_PASSWORD   "YOUR_WIFI_PASSWORD"
#define WIFI_AP_CHANNEL    "1"
#define WIFI_AP_HIDDEN     0

#define WIFI_SSID          WIFI_AP_SSID
#define WIFI_PASSWORD      WIFI_AP_PASSWORD

// Runtime object-detection model.
#define OBJECT_DETECTION_MODEL DEFAULT_YOLOV7TINY

// YOLO post-processing thresholds.
#define YOLO_CONFIDENCE_THRESHOLD         0.40f
#define YOLO_NMS_THRESHOLD                0.32f

// Camera / ISP preprocessing applied after Camera.videoInit() and before YOLO.
// Set the master switch to 0 to keep the camera ISP defaults.
#define CAMERA_PREPROCESS_ENABLE           1
#define CAMERA_PREPROCESS_BRIGHTNESS       0    // -64..64, default 0
#define CAMERA_PREPROCESS_CONTRAST         55   // 0..100, default 50
#define CAMERA_PREPROCESS_SATURATION       50   // 0..100, default 50
#define CAMERA_PREPROCESS_SHARPNESS        58   // 0..100, default 50
#define CAMERA_PREPROCESS_POWER_LINE_FREQ  2    // 0 off, 1 50Hz, 2 60Hz, 3 auto
#define CAMERA_PREPROCESS_EXPOSURE_MODE    1    // 0 manual, 1 auto
#define CAMERA_PREPROCESS_AWB_MODE         1    // 0 manual, 1 auto
#define CAMERA_PREPROCESS_WDR_MODE         0    // 0 off, 1 manual, 2 auto
#define CAMERA_PREPROCESS_WDR_LEVEL        50   // 50..100, manual WDR only
#define CAMERA_PREPROCESS_LDC_ENABLE       0    // 0 off, 1 on
#define CAMERA_PREPROCESS_MIN_FPS          10   // 1..30

// HW-493 laser aiming indicator. GPIO7 avoids GPIO6's alternate LOG_TX role.
// Use a transistor; do not use an AMB82 GPIO pin as the laser power source.
#define LASER_MODULE_ENABLE                  1
#define LASER_CONTROL_PIN                    7
#define LASER_ACTIVE_HIGH                    1
#define LASER_ON_WHEN_SYSTEM_READY           1
#define LASER_CONTINUOUS_MODE                 1
#define LASER_KEEPALIVE_INTERVAL_MS           1000UL

// Bounding-box smoothing.
#define BBOX_POSITION_ALPHA               0.65f
#define BBOX_SIZE_ALPHA                   0.20f
#define BBOX_FAST_ALPHA                   0.65f
#define BBOX_SIZE_DEADBAND                0.012f
#define BBOX_FAST_CHANGE_THRESHOLD        0.15f
#define BBOX_RELIABLE_SCORE               55
#define BBOX_LOW_CONFIDENCE_POSITION_ALPHA 0.35f
#define BBOX_FAST_CONFIRM_FRAMES          3
#define BBOX_FAST_MODE_FRAMES             3
#define BBOX_MAX_PERSON_TRACKS            4
#define BBOX_MISSING_RESET_FRAMES         6
#define BBOX_MATCH_IOU_THRESHOLD          0.15f
#define BBOX_MATCH_CENTER_THRESHOLD       0.20f
#define BBOX_MIN_RAW_COVERAGE              0.90f
#define BBOX_HORIZONTAL_PADDING_RATIO      0.0f
#define BBOX_VERTICAL_PADDING_RATIO        0.0f
#define BBOX_EDGE_GUARD_TOLERANCE          0.006f
#define BBOX_OVERSIZE_SHRINK_ALPHA         0.35f
#define BBOX_PRIMARY_SWITCH_RATIO          1.25f

// TOF400H polling interval. Valid readings are used directly without filtering.
#define DISTANCE_MEASUREMENT_INTERVAL_MS    250UL

// TOF400H UART/Modbus: TOF TX -> GPIO18, TOF RX -> GPIO19, external 5V, common GND.
#define TOF400H_BAUD_RATE                    115200
#define TOF400H_MODBUS_ADDRESS               0x01
#define TOF400H_RESPONSE_TIMEOUT_MS          60UL
#define TOF400H_MIN_DISTANCE_MM              40
#define TOF400H_MAX_DISTANCE_MM              4000
#define TOF400H_DIAGNOSTIC_ENABLE            1
#define TOF400H_ERROR_LOG_INTERVAL_MS        1000UL
#define TOF400H_LONG_RANGE_ENABLE            1

// Person-height calibration for a 1280x720 reference image.
#define HEIGHT_CALIBRATION_IMAGE_HEIGHT_PX 720.0f
#define HEIGHT_CALIBRATION_SLOPE           0.001374586854f
#define HEIGHT_CALIBRATION_INTERCEPT       0.022063368530f
#define HEIGHT_CALIBRATION_MIN_DISTANCE_CM 180.0f
#define HEIGHT_CALIBRATION_MAX_DISTANCE_CM 300.0f
#define HEIGHT_UPDATE_INTERVAL_MS          1000UL

#endif

