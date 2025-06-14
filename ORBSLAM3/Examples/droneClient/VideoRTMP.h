#ifndef _VIDEO_RTMP_H_
#define _VIDEO_RTMP_H_

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <atomic>

class VideoRTMP {
private:
    cv::VideoCapture cap;
    std::string rtmpUrl;
    float vx, vy, vz;
    double droneLocationLat, droneLocationLng;
    bool isConnected;
    
    // Frame buffer members
    std::queue<std::pair<cv::Mat, std::chrono::steady_clock::time_point>> frameBuffer;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread captureThread;
    bool stopThread;
    static const int BUFFER_SIZE = 10;  // Reduced buffer size for lower latency
    
    // Frame rate control
    std::chrono::steady_clock::time_point lastFrameTime;
    std::chrono::steady_clock::time_point lastProcessedFrameTime;
    const int targetFPS = 30;
    const std::chrono::microseconds frameInterval{1000000 / targetFPS};
    std::atomic<double> currentFPS{0.0};
    std::atomic<double> processingFPS{0.0};
    
    // Server processing metrics
    std::atomic<double> avgProcessingTime{0};  // Average server processing time
    std::atomic<double> avgServerDelay{0};     // Average delay including server processing
    std::atomic<int> serverProcessingCount{0}; // Count of frames processed by server
    std::atomic<bool> serverBusy{false};       // Indicates if server is still processing
    
    // Performance metrics
    std::atomic<int> droppedFrames{0};
    std::atomic<int> capturedFrames{0};
    std::atomic<double> bufferLatency{0};  // Time spent in buffer
    
    void captureLoop();
    bool shouldDropFrame() const;
    bool Frame() const;
    void updateMetrics(double processingTime);
    bool isFrameValid(const cv::Mat& frame) const;
    void updateFPS();
    double calculateOptimalDropRate() const;

public:
    VideoRTMP(const std::string& url);
    ~VideoRTMP();
    
    bool connect();
    void disconnect();
    void stop();
    cv::Mat getImage();
    // bool shouldDropFrame() const;
    
    // Server processing feedback
    void updateServerProcessingTime(double processingTime);
    void setServerBusy(bool busy) { serverBusy = busy; }
    bool isServerBusy() const { return serverBusy; }
    
    // Performance monitoring
    int getDroppedFrames() const { return droppedFrames; }
    int getCapturedFrames() const { return capturedFrames; }
    double getAvgProcessingTime() const { return avgProcessingTime; }
    double getAvgServerDelay() const { return avgServerDelay; }
    int getBufferSize() const { return frameBuffer.size(); }
    double getCurrentFPS() const { return currentFPS; }
    double getProcessingFPS() const { return processingFPS; }
    double getBufferLatency() const { return bufferLatency; }
    
    // Maintain the same interface as VideoSocketUDP
    float getvx() { return vx; }
    float getvy() { return vy; }
    float getvz() { return vz; }
    double getLatitude() { return droneLocationLat; }
    double getLongitude() { return droneLocationLng; }
    bool isStreamConnected() { return isConnected; }
};

#endif