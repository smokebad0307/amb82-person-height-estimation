// AMB82-MINI project:
//   1. HC-SR04 distance measurement (TRIG GPIO 4, ECHO GPIO 5)
//   2. Person-only YOLOv7-tiny detection
//   3. Only the largest detected person is drawn on RTSP and browser video
//   4. Camera, distance, and bounding-box height are shown on a web page
//
// HC-SR04 wiring:
//   VCC  -> AMB82-MINI V_USB (5 V)
//   GND  -> AMB82-MINI GND
//   TRIG -> GPIO 4 (PF11)
//   ECHO -> 1 kOhm -> GPIO 5 (PF12), with 2 kOhm from GPIO 5 to GND
//
// IMPORTANT: HC-SR04 ECHO is 5 V. Do not connect it directly to GPIO 5.

#include <WiFi.h>
#include "StreamIO.h"
#include "VideoStream.h"
#include "RTSP.h"
#include "NNObjectDetection.h"
#include "VideoStreamOverlay.h"
#include "config.h"
#include <math.h>

#define VIDEO_CHANNEL 0
#define JPEG_CHANNEL  1
#define NN_CHANNEL    3

#define NN_WIDTH  576
#define NN_HEIGHT 320

const int TRIG_PIN = 4;
const int ECHO_PIN = 5;
const int PERSON_CLASS_ID = 0;  // COCO class 0 is "person"

const unsigned long ECHO_TIMEOUT_US = 30000UL;
const unsigned long DISTANCE_INTERVAL_MS = 250UL;

// Wi-Fi APIs require writable character buffers.
char wifiSsid[] = WIFI_SSID;
char wifiPassword[] = WIFI_PASSWORD;
int wifiStatus = WL_IDLE_STATUS;

VideoSetting videoConfig(VIDEO_FHD, 30, VIDEO_H264, 0);
VideoSetting jpegConfig(1024, 576, 10, VIDEO_JPEG, 1);
VideoSetting nnConfig(NN_WIDTH, NN_HEIGHT, 10, VIDEO_RGB, 0);

NNObjectDetection objectDetector;
RTSP rtsp;
StreamIO videoToRtsp(1, 1);
StreamIO videoToNN(1, 1);
WiFiServer webServer(80);
WiFiServer streamServer(81);

IPAddress boardIp;
int rtspPort = 0;

// Shared sensor/detection values used by the web server.
// Integers are used so the callback and loop can exchange values simply.
volatile int distanceMm = -1;
volatile bool personDetected = false;
volatile int personCandidateCount = 0;
volatile int personScore = 0;
volatile int bboxHeightPx = 0;
volatile int bboxCalibrationHeightX10 = 0;
volatile int bboxHeightPercentX10 = 0;
volatile int bboxXMin = 0;
volatile int bboxYMin = 0;
volatile int bboxXMax = 0;
volatile int bboxYMax = 0;

// Fixed storage keeps the ultrasonic filter deterministic and heap-free.
int ultrasonicSamplesMm[ULTRASONIC_MEDIAN_WINDOW] = {};
int ultrasonicSampleCount = 0;
int ultrasonicSampleIndex = 0;

struct NormalizedBox {
    float cx;
    float cy;
    float width;
    float height;
};

struct PersonBoxTrack {
    bool active;
    bool matchedThisFrame;
    int missingFrames;
    int largeChangeDirection;
    int largeChangeCount;
    int fastFramesRemaining;
    int score;
    NormalizedBox raw;
    NormalizedBox filtered;
};

// Fixed-size storage avoids heap allocation in the per-frame callback.
PersonBoxTrack personTracks[BBOX_MAX_PERSON_TRACKS] = {};
int primaryTrackIndex = -1;

void objectDetectionCallback(std::vector<ObjectDetectionResult> results);
NormalizedBox makeNormalizedBox(float xmin, float ymin, float xmax, float ymax);
float boxIntersectionOverUnion(const NormalizedBox &a, const NormalizedBox &b);
void initializePersonTrack(PersonBoxTrack &track, const NormalizedBox &raw, int score);
void updatePersonTrack(PersonBoxTrack &track, const NormalizedBox &raw, int score);
void normalizedBoxToPixels(
    const NormalizedBox &box,
    int frameWidth,
    int frameHeight,
    int &xmin,
    int &ymin,
    int &xmax,
    int &ymax
);
int calculateEstimatedHeightX10(int distanceMmValue, int calibrationHeightX10);
bool heightDistanceInCalibrationRange(int distanceMmValue);
void resetUltrasonicMedianFilter();
int addUltrasonicMedianSample(int rawDistanceMm);
void updateUltrasonic();
void handleWebClient();
void sendDashboard(WiFiClient &client);
void sendJsonData(WiFiClient &client);
void sendNotFound(WiFiClient &client);
void streamCameraTask(void *parameter);

// The SDK model packer cannot expand OBJECT_DETECTION_MODEL from config.h.
// Fail at compile time if config.h no longer matches the literal model below.
static_assert(
    OBJECT_DETECTION_MODEL == DEFAULT_YOLOV7TINY,
    "Update the literal modelSelect model in mid.ino to match config.h"
);
static_assert(
    ULTRASONIC_MEDIAN_WINDOW > 0 && (ULTRASONIC_MEDIAN_WINDOW % 2) == 1,
    "ULTRASONIC_MEDIAN_WINDOW must be a positive odd number"
);

void setup()
{
    Serial.begin(115200);

    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    digitalWrite(TRIG_PIN, LOW);

    // Connect AMB82-MINI to the same router as the browser/VLC device.
    Serial.print("Connecting to Wi-Fi: ");
    Serial.println(wifiSsid);
    while (wifiStatus != WL_CONNECTED) {
        wifiStatus = WiFi.begin(wifiSsid, wifiPassword);
        if (wifiStatus != WL_CONNECTED) {
            Serial.println("Wi-Fi connection failed; retrying...");
            delay(3000);
        }
    }

    boardIp = WiFi.localIP();
    webServer.begin();
    streamServer.begin();

    // Configure the camera output and the lower-resolution NN input.
    videoConfig.setBitrate(2 * 1024 * 1024);
    Camera.configVideoChannel(VIDEO_CHANNEL, videoConfig);
    Camera.configVideoChannel(JPEG_CHANNEL, jpegConfig);
    Camera.configVideoChannel(NN_CHANNEL, nnConfig);
    Camera.videoInit();

    // RTSP video output.
    rtsp.configVideo(videoConfig);
    rtsp.begin();
    rtspPort = rtsp.getPort();

    // Custom person-only YOLOv7-tiny model selected in config.h.
    objectDetector.configVideo(nnConfig);
    objectDetector.configThreshold(YOLO_CONFIDENCE_THRESHOLD, YOLO_NMS_THRESHOLD);
    objectDetector.setResultCallback(objectDetectionCallback);
    // Keep this call on one line. The AmebaPro2 4.0.9 packaging tool parses
    // this source line to decide which .nb model is included in flash.
    objectDetector.modelSelect(OBJECT_DETECTION, DEFAULT_YOLOV7TINY, NA_MODEL, NA_MODEL);
    objectDetector.begin();

    videoToRtsp.registerInput(Camera.getStream(VIDEO_CHANNEL));
    videoToRtsp.registerOutput(rtsp);
    if (videoToRtsp.begin() != 0) {
        Serial.println("ERROR: RTSP StreamIO start failed");
    }
    Camera.channelBegin(VIDEO_CHANNEL);
    Camera.channelBegin(JPEG_CHANNEL);

    videoToNN.registerInput(Camera.getStream(NN_CHANNEL));
    videoToNN.setStackSize();
    videoToNN.setTaskPriority();
    videoToNN.registerOutput(objectDetector);
    if (videoToNN.begin() != 0) {
        Serial.println("ERROR: NN StreamIO start failed");
    }
    Camera.channelBegin(NN_CHANNEL);

    OSD.configVideo(VIDEO_CHANNEL, videoConfig);
    OSD.configVideo(JPEG_CHANNEL, jpegConfig);
    OSD.begin();

    // Keep the long-lived MJPEG connection off the Arduino loop so distance
    // measurements and /data responses continue while video is playing.
    if (xTaskCreate(streamCameraTask, "mjpeg", 2048, NULL, tskIDLE_PRIORITY + 1, NULL) != pdPASS) {
        Serial.println("ERROR: MJPEG task creation failed");
    }

    Serial.println();
    Serial.println("System ready");
    Serial.print("Connected Wi-Fi: ");
    Serial.println(wifiSsid);
    Serial.print("Dashboard: http://");
    Serial.println(boardIp);
    Serial.print("Browser video: http://");
    Serial.print(boardIp);
    Serial.println(":81/stream");
    Serial.print("RTSP stream: rtsp://");
    Serial.print(boardIp);
    Serial.print(":");
    Serial.println(rtspPort);
}

void loop()
{
    updateUltrasonic();
    handleWebClient();
    delay(2);
}

float clampFloat(float value, float minimum, float maximum)
{
    if (value < minimum) return minimum;
    if (value > maximum) return maximum;
    return value;
}

NormalizedBox makeNormalizedBox(float xmin, float ymin, float xmax, float ymax)
{
    xmin = clampFloat(xmin, 0.0f, 1.0f);
    ymin = clampFloat(ymin, 0.0f, 1.0f);
    xmax = clampFloat(xmax, 0.0f, 1.0f);
    ymax = clampFloat(ymax, 0.0f, 1.0f);

    NormalizedBox box;
    box.width = xmax - xmin;
    box.height = ymax - ymin;
    box.cx = (xmin + xmax) * 0.5f;
    box.cy = (ymin + ymax) * 0.5f;
    return box;
}

float boxIntersectionOverUnion(const NormalizedBox &a, const NormalizedBox &b)
{
    float aLeft = a.cx - a.width * 0.5f;
    float aTop = a.cy - a.height * 0.5f;
    float aRight = a.cx + a.width * 0.5f;
    float aBottom = a.cy + a.height * 0.5f;
    float bLeft = b.cx - b.width * 0.5f;
    float bTop = b.cy - b.height * 0.5f;
    float bRight = b.cx + b.width * 0.5f;
    float bBottom = b.cy + b.height * 0.5f;

    float intersectionWidth = ((aRight < bRight) ? aRight : bRight) -
                              ((aLeft > bLeft) ? aLeft : bLeft);
    float intersectionHeight = ((aBottom < bBottom) ? aBottom : bBottom) -
                               ((aTop > bTop) ? aTop : bTop);
    if (intersectionWidth <= 0.0f || intersectionHeight <= 0.0f) {
        return 0.0f;
    }

    float intersection = intersectionWidth * intersectionHeight;
    float unionArea = a.width * a.height + b.width * b.height - intersection;
    return (unionArea > 0.000001f) ? (intersection / unionArea) : 0.0f;
}

void clampNormalizedBox(NormalizedBox &box)
{
    box.width = clampFloat(box.width, 0.001f, 1.0f);
    box.height = clampFloat(box.height, 0.001f, 1.0f);
    box.cx = clampFloat(box.cx, box.width * 0.5f, 1.0f - box.width * 0.5f);
    box.cy = clampFloat(box.cy, box.height * 0.5f, 1.0f - box.height * 0.5f);
}

void initializePersonTrack(PersonBoxTrack &track, const NormalizedBox &raw, int score)
{
    track.active = true;
    track.matchedThisFrame = true;
    track.missingFrames = 0;
    track.largeChangeDirection = 0;
    track.largeChangeCount = 0;
    track.fastFramesRemaining = 0;
    track.score = score;
    track.raw = raw;
    track.filtered = raw;
    track.filtered.width *= 1.0f + 2.0f * BBOX_HORIZONTAL_PADDING_RATIO;
    track.filtered.height *= 1.0f + 2.0f * BBOX_VERTICAL_PADDING_RATIO;
    clampNormalizedBox(track.filtered);
}

void updatePersonTrack(PersonBoxTrack &track, const NormalizedBox &raw, int score)
{
    // Keep an untouched copy of this frame's YOLO output for diagnostics.
    track.raw = raw;
    NormalizedBox &filtered = track.filtered;
    bool reliableDetection = score >= BBOX_RELIABLE_SCORE;

    // Weak detections may maintain continuity, but influence position less.
    float positionAlpha = reliableDetection
                          ? BBOX_POSITION_ALPHA
                          : BBOX_LOW_CONFIDENCE_POSITION_ALPHA;
    filtered.cx += positionAlpha * (raw.cx - filtered.cx);
    filtered.cy += positionAlpha * (raw.cy - filtered.cy);

    float widthBase = (filtered.width > 0.001f) ? filtered.width : 0.001f;
    float heightBase = (filtered.height > 0.001f) ? filtered.height : 0.001f;
    float relativeWidthChange = fabsf(raw.width - filtered.width) / widthBase;
    float relativeHeightChange = fabsf(raw.height - filtered.height) / heightBase;
    float largestRelativeChange = (relativeWidthChange > relativeHeightChange)
                                    ? relativeWidthChange
                                    : relativeHeightChange;
    bool largeChange = largestRelativeChange >= BBOX_FAST_CHANGE_THRESHOLD;
    bool sizeWasUpdated = false;

    if (!reliableDetection) {
        // Do not let a weak box change width/height or confirm a fast resize.
        track.largeChangeCount = 0;
        track.largeChangeDirection = 0;
        if (track.fastFramesRemaining > 0) {
            track.fastFramesRemaining--;
        }
    } else if (largeChange && track.fastFramesRemaining == 0) {
        float rawArea = raw.width * raw.height;
        float filteredArea = filtered.width * filtered.height;
        int direction = (rawArea >= filteredArea) ? 1 : -1;

        if (direction == track.largeChangeDirection) {
            track.largeChangeCount++;
        } else {
            track.largeChangeDirection = direction;
            track.largeChangeCount = 1;
        }

        // Hold the current size for isolated large jumps. A real approach or
        // retreat persists for several frames and then enters fast EMA mode.
        if (track.largeChangeCount >= BBOX_FAST_CONFIRM_FRAMES) {
            filtered.width += BBOX_FAST_ALPHA * (raw.width - filtered.width);
            filtered.height += BBOX_FAST_ALPHA * (raw.height - filtered.height);
            track.largeChangeCount = 0;
            track.largeChangeDirection = 0;
            track.fastFramesRemaining = BBOX_FAST_MODE_FRAMES - 1;
            sizeWasUpdated = true;
        }
    } else {
        float alpha = (track.fastFramesRemaining > 0)
                      ? BBOX_FAST_ALPHA
                      : BBOX_SIZE_ALPHA;

        if (fabsf(raw.width - filtered.width) >= BBOX_SIZE_DEADBAND) {
            filtered.width += alpha * (raw.width - filtered.width);
            sizeWasUpdated = true;
        }
        if (fabsf(raw.height - filtered.height) >= BBOX_SIZE_DEADBAND) {
            filtered.height += alpha * (raw.height - filtered.height);
            sizeWasUpdated = true;
        }

        track.largeChangeCount = 0;
        track.largeChangeDirection = 0;
        if (track.fastFramesRemaining > 0) {
            track.fastFramesRemaining--;
        }
    }

    // During accepted updates, do not let smoothing leave the box much smaller
    // than the current person detection. Unconfirmed outliers bypass this guard.
    if (sizeWasUpdated) {
        float minimumWidth = raw.width * BBOX_MIN_RAW_COVERAGE;
        float minimumHeight = raw.height * BBOX_MIN_RAW_COVERAGE;
        if (filtered.width < minimumWidth) filtered.width = minimumWidth;
        if (filtered.height < minimumHeight) filtered.height = minimumHeight;
    }

    // Center and size use different EMA speeds, so the center can otherwise
    // move before the height catches up and cut off the head or feet. For a
    // trusted size measurement, expand only the necessary edges to cover the
    // padded raw box. Oversized edges move inward gradually, independently,
    // so the box becomes tighter without suddenly cutting off the person.
    bool trustedSizeMeasurement = reliableDetection &&
                                  (!largeChange ||
                                   track.fastFramesRemaining > 0 ||
                                   sizeWasUpdated);
    if (trustedSizeMeasurement) {
        float filteredLeft = filtered.cx - filtered.width * 0.5f;
        float filteredTop = filtered.cy - filtered.height * 0.5f;
        float filteredRight = filtered.cx + filtered.width * 0.5f;
        float filteredBottom = filtered.cy + filtered.height * 0.5f;

        float horizontalPadding = raw.width * BBOX_HORIZONTAL_PADDING_RATIO;
        float verticalPadding = raw.height * BBOX_VERTICAL_PADDING_RATIO;
        float protectedLeft = raw.cx - raw.width * 0.5f - horizontalPadding;
        float protectedTop = raw.cy - raw.height * 0.5f - verticalPadding;
        float protectedRight = raw.cx + raw.width * 0.5f + horizontalPadding;
        float protectedBottom = raw.cy + raw.height * 0.5f + verticalPadding;

        if (filteredLeft > protectedLeft + BBOX_EDGE_GUARD_TOLERANCE) {
            filteredLeft = protectedLeft;
        }
        if (filteredTop > protectedTop + BBOX_EDGE_GUARD_TOLERANCE) {
            filteredTop = protectedTop;
        }
        if (filteredRight < protectedRight - BBOX_EDGE_GUARD_TOLERANCE) {
            filteredRight = protectedRight;
        }
        if (filteredBottom < protectedBottom - BBOX_EDGE_GUARD_TOLERANCE) {
            filteredBottom = protectedBottom;
        }

        if (filteredLeft < protectedLeft - BBOX_EDGE_GUARD_TOLERANCE) {
            filteredLeft += BBOX_OVERSIZE_SHRINK_ALPHA *
                            (protectedLeft - filteredLeft);
        }
        if (filteredTop < protectedTop - BBOX_EDGE_GUARD_TOLERANCE) {
            filteredTop += BBOX_OVERSIZE_SHRINK_ALPHA *
                           (protectedTop - filteredTop);
        }
        if (filteredRight > protectedRight + BBOX_EDGE_GUARD_TOLERANCE) {
            filteredRight += BBOX_OVERSIZE_SHRINK_ALPHA *
                             (protectedRight - filteredRight);
        }
        if (filteredBottom > protectedBottom + BBOX_EDGE_GUARD_TOLERANCE) {
            filteredBottom += BBOX_OVERSIZE_SHRINK_ALPHA *
                              (protectedBottom - filteredBottom);
        }

        filtered.cx = (filteredLeft + filteredRight) * 0.5f;
        filtered.cy = (filteredTop + filteredBottom) * 0.5f;
        filtered.width = filteredRight - filteredLeft;
        filtered.height = filteredBottom - filteredTop;
    }

    clampNormalizedBox(filtered);
    track.score = score;
    track.matchedThisFrame = true;
    track.missingFrames = 0;
}

void normalizedBoxToPixels(
    const NormalizedBox &box,
    int frameWidth,
    int frameHeight,
    int &xmin,
    int &ymin,
    int &xmax,
    int &ymax
)
{
    xmin = (int)((box.cx - box.width * 0.5f) * frameWidth);
    xmax = (int)((box.cx + box.width * 0.5f) * frameWidth);
    ymin = (int)((box.cy - box.height * 0.5f) * frameHeight);
    ymax = (int)((box.cy + box.height * 0.5f) * frameHeight);

    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax >= frameWidth) xmax = frameWidth - 1;
    if (ymax >= frameHeight) ymax = frameHeight - 1;
    if (xmax < xmin) xmax = xmin;
    if (ymax < ymin) ymax = ymin;
}

// Return estimated person height in tenths of a centimetre. Integer exchange
// keeps the callback/web shared state simple; the fitted equation itself uses
// float math once per update and adds no NPU work.
int calculateEstimatedHeightX10(int distanceMmValue, int calibrationHeightX10)
{
    if (distanceMmValue <= 0 || calibrationHeightX10 <= 0) {
        return -1;
    }

    float distanceCm = distanceMmValue / 10.0f;
    float calibrationHeightPx = calibrationHeightX10 / 10.0f;
    float estimatedHeightCm = calibrationHeightPx *
        (HEIGHT_CALIBRATION_SLOPE * distanceCm + HEIGHT_CALIBRATION_INTERCEPT);

    if (estimatedHeightCm <= 0.0f) {
        return -1;
    }
    return (int)(estimatedHeightCm * 10.0f + 0.5f);
}

bool heightDistanceInCalibrationRange(int distanceMmValue)
{
    float distanceCm = distanceMmValue / 10.0f;
    return distanceCm >= HEIGHT_CALIBRATION_MIN_DISTANCE_CM &&
           distanceCm <= HEIGHT_CALIBRATION_MAX_DISTANCE_CM;
}

void resetUltrasonicMedianFilter()
{
    ultrasonicSampleCount = 0;
    ultrasonicSampleIndex = 0;
}

int addUltrasonicMedianSample(int rawDistanceMm)
{
    ultrasonicSamplesMm[ultrasonicSampleIndex] = rawDistanceMm;
    ultrasonicSampleIndex =
        (ultrasonicSampleIndex + 1) % ULTRASONIC_MEDIAN_WINDOW;
    if (ultrasonicSampleCount < ULTRASONIC_MEDIAN_WINDOW) {
        ultrasonicSampleCount++;
    }

    int sortedSamples[ULTRASONIC_MEDIAN_WINDOW];
    for (int i = 0; i < ultrasonicSampleCount; i++) {
        sortedSamples[i] = ultrasonicSamplesMm[i];
    }

    // Insertion sort is inexpensive for a five-sample fixed window.
    for (int i = 1; i < ultrasonicSampleCount; i++) {
        int value = sortedSamples[i];
        int j = i - 1;
        while (j >= 0 && sortedSamples[j] > value) {
            sortedSamples[j + 1] = sortedSamples[j];
            j--;
        }
        sortedSamples[j + 1] = value;
    }

    return sortedSamples[ultrasonicSampleCount / 2];
}

// YOLO result callback. Raw detections remain untouched; filtered track boxes
// are used only for selecting and drawing the displayed person.
void objectDetectionCallback(std::vector<ObjectDetectionResult> results)
{
    int candidateCount = 0;

    for (int i = 0; i < BBOX_MAX_PERSON_TRACKS; i++) {
        personTracks[i].matchedThisFrame = false;
    }

    for (unsigned int i = 0; i < results.size(); i++) {
        if (results[i].type() != PERSON_CLASS_ID) {
            continue;
        }
        candidateCount++;
        NormalizedBox raw = makeNormalizedBox(
            results[i].xMin(),
            results[i].yMin(),
            results[i].xMax(),
            results[i].yMax()
        );
        if (raw.width <= 0.0f || raw.height <= 0.0f) {
            continue;
        }

        int matchedIndex = -1;
        float bestIou = 0.0f;
        int nearestIndex = -1;
        float nearestCenterDistanceSquared = BBOX_MATCH_CENTER_THRESHOLD *
                                             BBOX_MATCH_CENTER_THRESHOLD;

        for (int trackIndex = 0; trackIndex < BBOX_MAX_PERSON_TRACKS; trackIndex++) {
            PersonBoxTrack &track = personTracks[trackIndex];
            if (!track.active || track.matchedThisFrame) {
                continue;
            }

            float iou = boxIntersectionOverUnion(track.filtered, raw);
            if (iou > bestIou) {
                bestIou = iou;
                matchedIndex = trackIndex;
            }

            float dx = raw.cx - track.filtered.cx;
            float dy = raw.cy - track.filtered.cy;
            float distanceSquared = dx * dx + dy * dy;
            if (distanceSquared < nearestCenterDistanceSquared) {
                nearestCenterDistanceSquared = distanceSquared;
                nearestIndex = trackIndex;
            }
        }

        if (bestIou < BBOX_MATCH_IOU_THRESHOLD) {
            matchedIndex = nearestIndex;
        }

        if (matchedIndex < 0) {
            // Prefer an unused slot. If all slots are active, recycle the
            // stalest track that was not already assigned this frame.
            int stalestIndex = -1;
            int largestMissingCount = -1;
            for (int trackIndex = 0; trackIndex < BBOX_MAX_PERSON_TRACKS; trackIndex++) {
                PersonBoxTrack &track = personTracks[trackIndex];
                if (!track.active) {
                    matchedIndex = trackIndex;
                    break;
                }
                if (!track.matchedThisFrame && track.missingFrames > largestMissingCount) {
                    largestMissingCount = track.missingFrames;
                    stalestIndex = trackIndex;
                }
            }
            if (matchedIndex < 0) {
                matchedIndex = stalestIndex;
            }
        }

        if (matchedIndex >= 0) {
            PersonBoxTrack &track = personTracks[matchedIndex];
            if (track.active) {
                updatePersonTrack(track, raw, results[i].score());
            } else {
                initializePersonTrack(track, raw, results[i].score());
            }
        }
    }

    for (int i = 0; i < BBOX_MAX_PERSON_TRACKS; i++) {
        PersonBoxTrack &track = personTracks[i];
        if (!track.active || track.matchedThisFrame) {
            continue;
        }
        track.missingFrames++;
        if (track.missingFrames >= BBOX_MISSING_RESET_FRAMES) {
            track.active = false;
            track.largeChangeCount = 0;
            track.fastFramesRemaining = 0;
            if (primaryTrackIndex == i) {
                primaryTrackIndex = -1;
            }
        }
    }

    int largestMatchedIndex = -1;
    float largestMatchedArea = 0.0f;
    for (int i = 0; i < BBOX_MAX_PERSON_TRACKS; i++) {
        if (!personTracks[i].active || !personTracks[i].matchedThisFrame) {
            continue;
        }
        float area = personTracks[i].filtered.width * personTracks[i].filtered.height;
        if (area > largestMatchedArea) {
            largestMatchedArea = area;
            largestMatchedIndex = i;
        }
    }

    if (largestMatchedIndex >= 0) {
        bool primaryIsMatched = primaryTrackIndex >= 0 &&
                                personTracks[primaryTrackIndex].active &&
                                personTracks[primaryTrackIndex].matchedThisFrame;
        if (!primaryIsMatched) {
            primaryTrackIndex = largestMatchedIndex;
        } else if (largestMatchedIndex != primaryTrackIndex) {
            float primaryArea = personTracks[primaryTrackIndex].filtered.width *
                                personTracks[primaryTrackIndex].filtered.height;
            if (largestMatchedArea > primaryArea * BBOX_PRIMARY_SWITCH_RATIO) {
                primaryTrackIndex = largestMatchedIndex;
            }
        }
    }

    int displayTrackIndex = -1;
    if (primaryTrackIndex >= 0 &&
        personTracks[primaryTrackIndex].active &&
        personTracks[primaryTrackIndex].matchedThisFrame) {
        displayTrackIndex = primaryTrackIndex;
    }

    OSD.createBitmap(VIDEO_CHANNEL);
    OSD.createBitmap(JPEG_CHANNEL);
    personCandidateCount = candidateCount;

    if (displayTrackIndex >= 0) {
        PersonBoxTrack &personTrack = personTracks[displayTrackIndex];
        int frameWidth = videoConfig.width();
        int frameHeight = videoConfig.height();
        int xmin, ymin, xmax, ymax;
        normalizedBoxToPixels(
            personTrack.filtered,
            frameWidth,
            frameHeight,
            xmin,
            ymin,
            xmax,
            ymax
        );

        int heightPx = ymax - ymin;
        if (heightPx < 0) heightPx = 0;

        bboxXMin = xmin;
        bboxYMin = ymin;
        bboxXMax = xmax;
        bboxYMax = ymax;
        bboxHeightPx = heightPx;
        int calibrationHeightX10 = (int)(
            personTrack.filtered.height * HEIGHT_CALIBRATION_IMAGE_HEIGHT_PX * 10.0f +
            0.5f
        );
        if (calibrationHeightX10 < 0) calibrationHeightX10 = 0;
        if (calibrationHeightX10 > (int)(HEIGHT_CALIBRATION_IMAGE_HEIGHT_PX * 10.0f)) {
            calibrationHeightX10 = (int)(HEIGHT_CALIBRATION_IMAGE_HEIGHT_PX * 10.0f);
        }
        bboxCalibrationHeightX10 = calibrationHeightX10;
        bboxHeightPercentX10 = (heightPx * 1000) / frameHeight;
        personScore = personTrack.score;
        personDetected = true;

        // Draw exactly one bounding box on the RTSP video.
        OSD.drawRect(
            VIDEO_CHANNEL,
            xmin,
            ymin,
            xmax,
            ymax,
            3,
            OSD_COLOR_WHITE
        );

        char label[72];
        int distanceSnapshot = distanceMm;
        int estimatedHeightX10 = -1;
        if (candidateCount == 1) {
            estimatedHeightX10 = calculateEstimatedHeightX10(
                distanceSnapshot,
                calibrationHeightX10
            );
        }
        if (estimatedHeightX10 >= 0) {
            snprintf(
                label,
                sizeof(label),
                "Person Conf:%d%% H:%dpx Est:%d.%dcm",
                personScore,
                heightPx,
                estimatedHeightX10 / 10,
                estimatedHeightX10 % 10
            );
        } else {
            snprintf(
                label,
                sizeof(label),
                "Person Conf:%d%% H:%dpx",
                personScore,
                heightPx
            );
        }
        int textY = ymin - OSD.getTextHeight(VIDEO_CHANNEL);
        if (textY < 0) textY = 0;
        OSD.drawText(
            VIDEO_CHANNEL,
            xmin,
            textY,
            label,
            OSD_COLOR_CYAN
        );

        // Scale the same smoothed normalized box to the browser JPEG channel.
        int jpegWidth = jpegConfig.width();
        int jpegHeight = jpegConfig.height();
        int jpegXMin, jpegYMin, jpegXMax, jpegYMax;
        normalizedBoxToPixels(
            personTrack.filtered,
            jpegWidth,
            jpegHeight,
            jpegXMin,
            jpegYMin,
            jpegXMax,
            jpegYMax
        );

        OSD.drawRect(
            JPEG_CHANNEL,
            jpegXMin,
            jpegYMin,
            jpegXMax,
            jpegYMax,
            3,
            OSD_COLOR_WHITE
        );
        int jpegTextY = jpegYMin - OSD.getTextHeight(JPEG_CHANNEL);
        if (jpegTextY < 0) jpegTextY = 0;
        OSD.drawText(
            JPEG_CHANNEL,
            jpegXMin,
            jpegTextY,
            label,
            OSD_COLOR_CYAN
        );
    } else {
        // No person in this frame. createBitmap() plus update() clears old OSD.
        personDetected = false;
        personScore = 0;
        bboxHeightPx = 0;
        bboxCalibrationHeightX10 = 0;
        bboxHeightPercentX10 = 0;
        bboxXMin = 0;
        bboxYMin = 0;
        bboxXMax = 0;
        bboxYMax = 0;
    }

    OSD.update(VIDEO_CHANNEL);
    OSD.update(JPEG_CHANNEL);
}

void updateUltrasonic()
{
    static unsigned long lastMeasurementMs = 0;
    unsigned long now = millis();

    if ((now - lastMeasurementMs) < DISTANCE_INTERVAL_MS) {
        return;
    }
    lastMeasurementMs = now;

    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    unsigned long durationUs = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
    if (durationUs == 0) {
        resetUltrasonicMedianFilter();
        distanceMm = -1;
        Serial.println("Ultrasonic: no echo / out of range");
    } else {
        // distance(mm) = duration(us) * 0.343 / 2
        int rawDistanceMm = (int)((durationUs * 343UL) / 2000UL);
        int medianDistanceMm = addUltrasonicMedianSample(rawDistanceMm);
        distanceMm = medianDistanceMm + ULTRASONIC_DISTANCE_OFFSET_MM;
        int distanceSnapshot = distanceMm;
        bool detectedSnapshot = personDetected;
        int candidateSnapshot = personCandidateCount;
        int calibrationHeightSnapshot = bboxCalibrationHeightX10;
        Serial.print("Ultrasonic distance (median + 5.0 cm): ");
        Serial.print(distanceSnapshot / 10.0f, 1);
        Serial.print(" cm");

        if (detectedSnapshot && candidateSnapshot == 1 && calibrationHeightSnapshot > 0) {
            int estimatedHeightX10 = calculateEstimatedHeightX10(
                distanceSnapshot,
                calibrationHeightSnapshot
            );
            Serial.print(" | bbox@720p: ");
            Serial.print(calibrationHeightSnapshot / 10.0f, 1);
            Serial.print(" px | estimated height: ");
            Serial.print(estimatedHeightX10 / 10.0f, 1);
            Serial.print(" cm");
            if (!heightDistanceInCalibrationRange(distanceSnapshot)) {
                Serial.print(" (outside 180-300 cm calibration range)");
            }
        } else if (detectedSnapshot && candidateSnapshot > 1) {
            Serial.print(" | height disabled: multiple people");
        }
        Serial.println();
    }
}

void streamCameraTask(void *parameter)
{
    (void)parameter;
    const char *boundary = "amb82frame";

    while (true) {
        WiFiClient client = streamServer.available();
        if (!client) {
            delay(5);
            continue;
        }

        String currentLine = "";
        bool headersComplete = false;
        while (client.connected() && !headersComplete) {
            if (!client.available()) {
                delay(1);
                continue;
            }

            char c = client.read();
            if (c == '\n') {
                if (currentLine.length() == 0) {
                    headersComplete = true;
                } else {
                    currentLine = "";
                }
            } else if (c != '\r') {
                currentLine += c;
            }
        }

        if (headersComplete) {
            client.println("HTTP/1.1 200 OK");
            client.println("Access-Control-Allow-Origin: *");
            client.println("Cache-Control: no-store, no-cache, must-revalidate");
            client.println("Pragma: no-cache");
            client.print("Content-Type: multipart/x-mixed-replace; boundary=");
            client.println(boundary);
            client.println("Connection: close");
            client.println();

            while (client.connected()) {
                uint32_t imageAddress = 0;
                uint32_t imageLength = 0;
                Camera.getImage(JPEG_CHANNEL, &imageAddress, &imageLength);

                if (imageAddress == 0 || imageLength == 0) {
                    delay(20);
                    continue;
                }

                client.print("--");
                client.println(boundary);
                client.println("Content-Type: image/jpeg");
                client.print("Content-Length: ");
                client.println(imageLength);
                client.println();

                uint8_t *image = (uint8_t *)imageAddress;
                uint32_t sent = 0;
                while (sent < imageLength && client.connected()) {
                    uint32_t remaining = imageLength - sent;
                    uint32_t blockSize = remaining > 1024 ? 1024 : remaining;
                    client.write(image + sent, blockSize);
                    sent += blockSize;
                }
                client.print("\r\n");
                delay(80);
            }
        }

        client.stop();
    }
}

void handleWebClient()
{
    WiFiClient client = webServer.available();
    if (!client) {
        return;
    }

    // Follow the request-reading pattern used by the official Ameba HTTP
    // example. WiFiClient::available() is blocking in this SDK, so do not try
    // to drain the socket after the request: respond as soon as the blank
    // header line is received.
    String requestLine = "";
    String currentLine = "";
    bool firstLineComplete = false;
    bool responseSent = false;

    while (client.connected() && !responseSent) {
        if (client.available()) {
            char c = client.read();

            if (!firstLineComplete) {
                if (c == '\n') {
                    firstLineComplete = true;
                } else if (c != '\r') {
                    requestLine += c;
                }
            }

            if (c == '\n') {
                if (currentLine.length() == 0) {
                    if (requestLine.startsWith("GET /data ")) {
                        sendJsonData(client);
                    } else if (requestLine.startsWith("GET / ") ||
                               requestLine.startsWith("GET /index.html ")) {
                        sendDashboard(client);
                    } else {
                        sendNotFound(client);
                    }
                    responseSent = true;
                } else {
                    currentLine = "";
                }
            } else if (c != '\r') {
                currentLine += c;
            }
        }
    }

    // Give the network stack time to transmit queued data, then explicitly
    // close the connection. Without this close, browser fetch() keeps waiting
    // because these responses intentionally do not use Content-Length.
    delay(10);
    client.stop();
}

void sendJsonData(WiFiClient &client)
{
    int distanceSnapshot = distanceMm;
    bool detectedSnapshot = personDetected;
    int candidateSnapshot = personCandidateCount;
    int scoreSnapshot = personScore;
    int heightSnapshot = bboxHeightPx;
    int calibrationHeightSnapshot = bboxCalibrationHeightX10;
    int heightPercentSnapshot = bboxHeightPercentX10;
    int xminSnapshot = bboxXMin;
    int yminSnapshot = bboxYMin;
    int xmaxSnapshot = bboxXMax;
    int ymaxSnapshot = bboxYMax;
    int estimatedHeightX10 = -1;
    if (detectedSnapshot && candidateSnapshot == 1) {
        estimatedHeightX10 = calculateEstimatedHeightX10(
            distanceSnapshot,
            calibrationHeightSnapshot
        );
    }
    bool heightInRange = heightDistanceInCalibrationRange(distanceSnapshot);
    bool heightReliable = estimatedHeightX10 >= 0 && heightInRange &&
                          scoreSnapshot >= BBOX_RELIABLE_SCORE;

    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: application/json; charset=utf-8");
    client.println("Cache-Control: no-store");
    client.println("Connection: close");
    client.println();

    client.print("{\"distanceCm\":");
    if (distanceSnapshot < 0) {
        client.print("null");
    } else {
        client.print(distanceSnapshot / 10.0f, 1);
    }
    client.print(",\"personDetected\":");
    client.print(detectedSnapshot ? "true" : "false");
    client.print(",\"personCandidates\":");
    client.print(candidateSnapshot);
    client.print(",\"confidence\":");
    client.print(scoreSnapshot);
    client.print(",\"bboxHeightPx\":");
    client.print(heightSnapshot);
    client.print(",\"bboxHeightCalibrationPx\":");
    client.print(calibrationHeightSnapshot / 10.0f, 1);
    client.print(",\"bboxHeightPercent\":");
    client.print(heightPercentSnapshot / 10.0f, 1);
    client.print(",\"estimatedHeightCm\":");
    if (estimatedHeightX10 < 0) {
        client.print("null");
    } else {
        client.print(estimatedHeightX10 / 10.0f, 1);
    }
    client.print(",\"heightDistanceInRange\":");
    client.print(heightInRange ? "true" : "false");
    client.print(",\"heightReliable\":");
    client.print(heightReliable ? "true" : "false");
    client.print(",\"bbox\":{");
    client.print("\"xmin\":");
    client.print(xminSnapshot);
    client.print(",\"ymin\":");
    client.print(yminSnapshot);
    client.print(",\"xmax\":");
    client.print(xmaxSnapshot);
    client.print(",\"ymax\":");
    client.print(ymaxSnapshot);
    client.println("}}");
}

void sendDashboard(WiFiClient &client)
{
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/html; charset=utf-8");
    client.println("Cache-Control: no-store");
    client.println("Connection: close");
    client.println();

    client.print(
        "<!doctype html><html lang='zh-Hant'><head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>AMB82 人物監測</title>"
        "<style>"
        "body{margin:0;background:#0b1220;color:#e5e7eb;font-family:Arial,sans-serif}"
        ".wrap{max-width:820px;margin:auto;padding:28px 18px}"
        "h1{font-size:28px;margin:0 0 8px}.sub{color:#94a3b8;margin-bottom:22px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}"
        ".card{background:#172033;border:1px solid #29344b;border-radius:16px;padding:20px}"
        ".label{color:#94a3b8;font-size:14px}.value{font-size:34px;font-weight:700;margin-top:8px}"
        ".ok{color:#4ade80}.no{color:#f87171}.warn{color:#fbbf24}"
        ".info{margin-top:16px;color:#cbd5e1;line-height:1.7}code{color:#67e8f9}"
        ".video-card{padding:12px;margin-bottom:14px}.video-title{padding:4px 6px 10px;color:#cbd5e1}"
        ".camera{display:block;width:100%;height:auto;aspect-ratio:16/9;object-fit:contain;background:#020617;border-radius:10px}"
        "</style></head><body><main class='wrap'>"
        "<h1>AMB82 人物監測</h1>"
        "<div class='sub'>YOLOv7-tiny 最大人物框 + HC-SR04</div>"
        "<div class='card video-card'><div class='video-title'>即時鏡頭（含人物框）</div>"
        "<img id='camera' class='camera' src='http://"
    );
    client.print(boardIp);
    client.print(
        ":81/stream' alt='AMB82 即時鏡頭' onerror=\"this.alt='鏡頭串流連線失敗，請重新整理頁面'\"></div>"
        "<section class='grid'>"
        "<div class='card'><div class='label'>超聲波距離</div><div id='distance' class='value'>--</div></div>"
        "<div class='card'><div class='label'>Bounding box 高度</div><div id='height' class='value'>--</div><div id='heightPct' class='label'></div><div id='calibrationHeight' class='label'></div></div>"
        "<div class='card'><div class='label'>估算含鞋身高</div><div id='estimatedHeight' class='value'>--</div><div id='heightStatus' class='label'></div></div>"
        "<div class='card'><div class='label'>人物狀態</div><div id='person' class='value'>等待中</div><div id='confidence' class='label'></div></div>"
        "</section>"
        "<div class='card info'>"
        "畫面只顯示面積最大的單一人物框。候選人物數：<span id='count'>0</span><br>"
        "框座標：<code id='bbox'>--</code><br>"
        "身高校正範圍：<code>180–300 cm</code>；使用換算至 1280×720 的平滑人物框，超聲波需對準同一人。<br>"
        "瀏覽器影片：<code>http://"
    );
    client.print(boardIp);
    client.print(
        ":81/stream</code><br>"
        "RTSP 位址：<code>rtsp://"
    );
    client.print(boardIp);
    client.print(":");
    client.print(rtspPort);
    client.print(
        "</code></div>"
        "</main><script>"
        "async function refresh(){try{const r=await fetch('/data',{cache:'no-store'});const d=await r.json();"
        "document.getElementById('distance').textContent=d.distanceCm===null?'無回波':d.distanceCm.toFixed(1)+' cm';"
        "document.getElementById('height').textContent=d.personDetected?d.bboxHeightPx+' px':'--';"
        "document.getElementById('heightPct').textContent=d.personDetected?d.bboxHeightPercent.toFixed(1)+'% 畫面高度':'';"
        "document.getElementById('calibrationHeight').textContent=d.personDetected?'校正用 720p：'+d.bboxHeightCalibrationPx.toFixed(1)+' px':'';"
        "const eh=document.getElementById('estimatedHeight'),hs=document.getElementById('heightStatus');"
        "eh.textContent=d.estimatedHeightCm===null?'--':d.estimatedHeightCm.toFixed(1)+' cm';"
        "eh.className='value '+(d.heightReliable?'ok':(d.estimatedHeightCm===null?'':'warn'));"
        "if(d.personCandidates>1){hs.textContent='多人時無法確認超聲波對應目標';}"
        "else if(!d.personDetected){hs.textContent='等待人物框';}"
        "else if(d.distanceCm===null){hs.textContent='等待超聲波回波';}"
        "else if(!d.heightDistanceInRange){hs.textContent='距離超出校正範圍，結果僅為外插';}"
        "else if(d.confidence<55){hs.textContent='偵測信心較低，結果僅供參考';}"
        "else{hs.textContent='校正範圍內；身高包含鞋子';}"
        "const p=document.getElementById('person');p.textContent=d.personDetected?'偵測到':'未偵測到';p.className='value '+(d.personDetected?'ok':'no');"
        "document.getElementById('confidence').textContent=d.personDetected?'信心分數 '+d.confidence+'%':'';"
        "document.getElementById('count').textContent=d.personCandidates;"
        "document.getElementById('bbox').textContent=d.personDetected?'('+d.bbox.xmin+', '+d.bbox.ymin+') - ('+d.bbox.xmax+', '+d.bbox.ymax+')':'--';"
        "}catch(e){document.getElementById('person').textContent='連線中斷';}}"
        "refresh();setInterval(refresh,500);"
        "</script></body></html>"
    );
}

void sendNotFound(WiFiClient &client)
{
    client.println("HTTP/1.1 404 Not Found");
    client.println("Content-Type: text/plain; charset=utf-8");
    client.println("Connection: close");
    client.println();
    client.println("Not found");
}
