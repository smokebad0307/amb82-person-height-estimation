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

// Ultrasonic distance filtering and calibration.
#define ULTRASONIC_MEDIAN_WINDOW           5
#define ULTRASONIC_DISTANCE_OFFSET_MM      50

// Person-height calibration for a 1280x720 reference image.
#define HEIGHT_CALIBRATION_IMAGE_HEIGHT_PX 720.0f
#define HEIGHT_CALIBRATION_SLOPE           0.001374586854f
#define HEIGHT_CALIBRATION_INTERCEPT       0.022063368530f
#define HEIGHT_CALIBRATION_MIN_DISTANCE_CM 180.0f
#define HEIGHT_CALIBRATION_MAX_DISTANCE_CM 300.0f

#endif

