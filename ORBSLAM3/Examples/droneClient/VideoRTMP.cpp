#include "VideoRTMP.h"
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

// Reduced logging in production
#ifdef DEBUG
#define LOG_DEBUG(x) std::cout << x << std::endl
#else
#define LOG_DEBUG(x)
#endif

#define LOG_ERROR(x) std::cerr << x << std::endl

VideoRTMP::VideoRTMP(const std::string& url) : 
    rtmpUrl(url), 
    vx(0), vy(0), vz(0),
    droneLocationLat(0), droneLocationLng(0),
    isConnected(false),
    stopThread(false),
    lastFrameTime(std::chrono::steady_clock::now()),
    lastProcessedFrameTime(std::chrono::steady_clock::now()),
    currentFPS(0),
    processingFPS(0),
    adaptiveFPS(30),
    consecutiveDrops(0),
    consecutiveSuccesses(0),
    bufferUtilization(0),
    avgProcessingTime(0),
    avgServerDelay(0),
    serverProcessingCount(0),
    serverBusy(false),
    droppedFrames(0),
    capturedFrames(0),
    bufferLatency(0) {
    
    if (url.empty() || url.substr(0, 7) != "rtmp://") {
        throw std::invalid_argument("[VideoRTMP] Invalid RTMP URL format");
    }
    
    if(!connect()) {
        throw std::runtime_error("[VideoRTMP] Failed to establish initial connection");
    }
    captureThread = std::thread(&VideoRTMP::captureLoop, this);
}

VideoRTMP::~VideoRTMP() {
    stop();
}

bool VideoRTMP::shouldDropFrame() {
    // Calculate buffer utilization
    double bufferFullness = static_cast<double>(frameBuffer.size()) / BUFFER_SIZE;
    bufferUtilization = bufferFullness;

    // Only drop frames if buffer is critically full
    if (bufferFullness >= 0.98) {
        return true;
    }
    
    // Only drop frames if server is very busy
    if (serverBusy && bufferFullness > 0.8) {
        return true;
    }
    
    // Calculate drop rate based on server processing time - Much less aggressive
    if (serverProcessingCount > 0) {
        double dropProbability = 0.0;
        
        // Only drop frames if server delay is extremely high
        if (avgServerDelay > 1200.0) {
            dropProbability = 0.2;  // Drop only 10% of frames
        } else if (avgServerDelay > 800.0) {
            dropProbability = 0.1;  // Drop only 5% of frames
        }
        
        if (bufferFullness > 0.9) {
            dropProbability += 0.05;  // Very small increase
        }

        // Adaptive drop rate based on consecutive drops
        if (consecutiveDrops > 5) {
            dropProbability *= 0.3; // Significantly reduce drop rate if we've been dropping too much
        }

        // Random drop based on probability
        bool shouldDrop = ((double)rand() / RAND_MAX) < dropProbability;
        
        // Update consecutive counters
        if (shouldDrop) {
            consecutiveDrops++;
            consecutiveSuccesses = 0;
        } else {
            consecutiveSuccesses++;
            consecutiveDrops = 0;
        }
        
        return shouldDrop;
    }
    
    return false;
}

void VideoRTMP::updateMetrics(double processingTime) {
    capturedFrames++;
    avgProcessingTime = avgProcessingTime * 0.95 + processingTime * 0.05;
}

bool VideoRTMP::isFrameValid(const cv::Mat& frame) const {
    if (frame.empty()) return false;
    
    try {
        // Check for completely black or white frames
        cv::Scalar mean = cv::mean(frame);
        if (mean[0] < 5 || mean[0] > 250) return false;
        
        // Check for reasonable frame size
        if (frame.rows < 100 || frame.cols < 100) return false;
        
        // Check for corrupted pixel values
        double minVal, maxVal;
        cv::minMaxLoc(frame, &minVal, &maxVal);
        if (std::isnan(minVal) || std::isnan(maxVal)) return false;
        
        return true;
    } catch (const cv::Exception& e) {
        LOG_ERROR("[VideoRTMP] Error validating frame: " << e.what());
        return false;
    }
}

void VideoRTMP::updateFPS() {
    static int frameCount = 0;
    static auto lastFPSUpdate = std::chrono::steady_clock::now();
    frameCount++;
    
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - lastFPSUpdate);
    
    if (duration.count() >= 1) {
        currentFPS = frameCount / duration.count();
        frameCount = 0;
        lastFPSUpdate = now;
        
        // Update adaptive frame rate
        updateAdaptiveFrameRate();
    }
}

void VideoRTMP::updateAdaptiveFrameRate() {
    // Adaptive frame rate based on performance - Less aggressive
    int newFPS = adaptiveFPS;
    
    // If we're dropping too many frames, reduce FPS slightly
    if (consecutiveDrops > 8) {
        newFPS = std::max(25, adaptiveFPS - 2);  // Don't go below 25 FPS
        consecutiveDrops = 0;
    }
    // If we're doing well, gradually increase FPS
    else if (consecutiveSuccesses > 15 && bufferUtilization < 0.7) {
        newFPS = std::min(30, adaptiveFPS + 1);  // Increase more slowly
        consecutiveSuccesses = 0;
    }
    // If server is struggling, reduce FPS slightly
    else if (avgServerDelay > 500.0) {
        newFPS = std::max(26, adaptiveFPS - 1);  // Don't reduce too much
    }
    // If buffer is getting full, reduce FPS slightly
    else if (bufferUtilization > 0.9) {
        newFPS = std::max(27, adaptiveFPS - 1);  // Don't reduce too much
    }
    
    adaptiveFPS = newFPS;
}

double VideoRTMP::calculateOptimalDropRate() const {
    double bufferFullness = static_cast<double>(frameBuffer.size()) / BUFFER_SIZE;
    double processingRatio = processingFPS / currentFPS;
    
    // If processing is keeping up, no need to drop frames
    if (processingRatio >= 0.9) return 0.0;  // Reduced from 0.95
    
    // Calculate drop rate based on buffer fullness and processing speed
    double dropRate = 0.0;
    
    if (bufferFullness > 0.9) {  // Increased from 0.8
        // Buffer getting full, increase drop rate gradually
        dropRate = 0.3 + (bufferFullness - 0.9) * 2.0;  // Reduced from 0.5 and 2.5
    } else if (processingRatio < 0.6) {  // Increased from 0.5
        // Processing falling behind, drop frames based on ratio
        dropRate = (1.0 - processingRatio) * 0.5;  // Added 0.5 multiplier to reduce drop rate
    }
    
    return std::min(0.5, dropRate); // Never drop more than 50% of frames (reduced from 80%)
}

void VideoRTMP::updateServerProcessingTime(double processingTime) {
    // Update average processing time with exponential moving average
    avgProcessingTime = avgProcessingTime * 0.7 + processingTime * 0.3;
    
    // Update server delay (includes both processing and buffer time)
    double totalDelay = processingTime + bufferLatency;
    avgServerDelay = avgServerDelay * 0.7 + totalDelay * 0.3;
    
    serverProcessingCount++;
}

void VideoRTMP::captureLoop() {
    int reconnectAttempts = 0;
    const int maxReconnectAttempts = 10;
    auto minFrameInterval = std::chrono::milliseconds(1000 / adaptiveFPS); // Use adaptive FPS
    
    cv::Mat frame;
    frame.reserve(1088 * 720 * 3);
    
    while (!stopThread) {
        if (!isConnected) {
            if (reconnectAttempts >= maxReconnectAttempts) {
                LOG_ERROR("[VideoRTMP] Max reconnection attempts reached");
                break;
            }
            if (!connect()) {
                reconnectAttempts++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            reconnectAttempts = 0;
        }

        auto frameStart = std::chrono::steady_clock::now();
        auto timeSinceLastFrame = std::chrono::duration_cast<std::chrono::milliseconds>(
            frameStart - lastFrameTime);
        
        // Use adaptive frame interval
        minFrameInterval = std::chrono::milliseconds(1000 / adaptiveFPS);
        if (timeSinceLastFrame < minFrameInterval) {
            std::this_thread::sleep_for(minFrameInterval - timeSinceLastFrame);
            continue;
        }

        try {
            if (!cap.read(frame)) {
                LOG_ERROR("[VideoRTMP] Failed to read frame");
                isConnected = false;
                continue;
            }
        } catch (const cv::Exception& e) {
            LOG_ERROR("[VideoRTMP] Error reading frame: " << e.what());
            isConnected = false;
            continue;
        }

        if (frame.empty() || !isFrameValid(frame)) {
            continue;
        }

        // Resize if needed
        if (frame.size() != cv::Size(640, 480)) {
            try {
                cv::resize(frame, frame, cv::Size(640, 480), 0, 0, cv::INTER_LINEAR);
            } catch (const cv::Exception& e) {
                LOG_ERROR("[VideoRTMP] Error resizing frame: " << e.what());
                continue;
            }
        }

        updateFPS();
        
        // Check if we should drop this frame
        if (shouldDropFrame()) {
            droppedFrames++;
            continue;
        }

        {
            std::unique_lock<std::mutex> lock(mtx);
            
            // Remove oldest frame if buffer is full
            while (frameBuffer.size() >= BUFFER_SIZE) {
                frameBuffer.pop();
                droppedFrames++;
            }
            
            // Store frame with timestamp
            frameBuffer.push({frame.clone(), frameStart});
            lastFrameTime = frameStart;
            capturedFrames++;
            
            lock.unlock();
            cv.notify_one();
        }

        // Update metrics less frequently
        if (capturedFrames % 30 == 0) {  // Reduced from 100 to 30 for more frequent updates
            auto now = std::chrono::steady_clock::now();
            auto processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - frameStart).count();
            updateMetrics(static_cast<double>(processingTime));
            
            LOG_DEBUG("[VideoRTMP] Stats: FPS=" << currentFPS 
                      << " AdaptiveFPS=" << adaptiveFPS
                      << " ProcessingFPS=" << processingFPS
                      << " Dropped=" << droppedFrames
                      << " Buffer=" << frameBuffer.size() << "/" << BUFFER_SIZE
                      << " BufferUtil=" << (bufferUtilization * 100) << "%"
                      << " ServerDelay=" << avgServerDelay << "ms"
                      << " ConsecDrops=" << consecutiveDrops
                      << " ConsecSuccess=" << consecutiveSuccesses);
        }
    }
}

bool VideoRTMP::connect() {
    if (isConnected) return true;

    try {
        cap.release();
        
        // Try direct RTMP with optimized settings
        cap.open(rtmpUrl, cv::CAP_FFMPEG);
        if (!cap.isOpened()) {
            // Enhanced FFmpeg pipeline with ultra-low latency optimizations
            std::string ffmpeg_pipeline = 
                "ffmpeg -loglevel error"  // Reduced logging
                " -fflags nobuffer"       // Disable input buffering
                " -flags low_delay"       // Minimize latency
                " -thread_queue_size 2048" // Increased queue size for better buffering
                " -rtmp_live live"        // Enable live streaming mode
                " -rtmp_buffer 1000"      // Ultra-low RTMP buffer size
                " -probesize 32"          // Minimal probe size
                " -analyzeduration 0"     // No analysis delay
                " -i " + rtmpUrl + 
                " -vcodec rawvideo"
                " -pix_fmt bgr24"
                " -f rawvideo pipe:1";
            cap.open(ffmpeg_pipeline, cv::CAP_FFMPEG);
        }
        
        if (!cap.isOpened()) {
            return false;
        }

        // Optimize capture settings
        cap.set(cv::CAP_PROP_BUFFERSIZE, BUFFER_SIZE);
        cap.set(cv::CAP_PROP_FPS, targetFPS);
        
        // Quick connection test
        cv::Mat testFrame;
        if (!cap.read(testFrame) || testFrame.empty()) {
            cap.release();
            return false;
        }
        
        isConnected = true;
        return true;
    }
    catch (const std::exception& e) {
        LOG_ERROR("[VideoRTMP] Connection error: " << e.what());
        cap.release();
        return false;
    }
}

void VideoRTMP::disconnect() {
    if (isConnected) {
        cap.release();
        isConnected = false;
        std::unique_lock<std::mutex> lock(mtx);
        while (!frameBuffer.empty()) {
            frameBuffer.pop();
        }
    }
}

cv::Mat VideoRTMP::getImage() {
    if (!isConnected) {
        return cv::Mat();
    }

    std::unique_lock<std::mutex> lock(mtx);
    
    if (frameBuffer.empty()) {
        if (cv.wait_for(lock, std::chrono::milliseconds(100)) == std::cv_status::timeout) {
            return cv::Mat();
        }
    }
    
    if (frameBuffer.empty()) {
        return cv::Mat();
    }
    
    // Get frame and its timestamp
    auto [frame, timestamp] = frameBuffer.front();
    frameBuffer.pop();
    
    // Calculate time spent in buffer
    auto now = std::chrono::steady_clock::now();
    bufferLatency = std::chrono::duration_cast<std::chrono::milliseconds>(now - timestamp).count();
    
    // Final validation
    if (!isFrameValid(frame)) {
        LOG_ERROR("[VideoRTMP] Invalid frame detected in getImage");
        return cv::Mat();
    }
    
    return frame.clone();
}

void VideoRTMP::stop() {
    if (!stopThread) {
        stopThread = true;
        cv.notify_all();
        if(captureThread.joinable()) {
            captureThread.join();
        }
        disconnect();
    }
}
