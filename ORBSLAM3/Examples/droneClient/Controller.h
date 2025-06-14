#ifndef _CONTROLLER_H_
#define _CONTROLLER_H_

#define _WINSOCK_DEPRECATED_NO_WARNINGS 1

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <sys/stat.h>
#include <chrono>
#include <thread>

// #include <opencv/cv.h>
// #include <opencv/highgui.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#if defined(_WIN32) || defined(WIN32)
#include <Ws2tcpip.h> //For Windows.
#include <windows.h>  //For Windows.
#include <winsock2.h> //For Windows.
#pragma comment (lib, "Ws2_32.lib")    //For Windows.
#pragma comment (lib, "Mswsock.lib")   //For Windows.
#pragma comment (lib, "AdvApi32.lib")  //For Windows.

#elif defined(__unix__) || defined(__linux__)
#include <sys/types.h>  //For linux.
#include <sys/socket.h> //For linux.
#include <netinet/in.h> //For linux.
#include <netinet/ip.h> //For linux.
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>

#endif

#include "VideoRTMP.h"

using namespace std;
using namespace cv;

// Command types for unified control interface
enum class CommandType {
    CUSTOM,             // Direct velocity control
    TAKEOFF_LANDING,    // Takeoff or landing
    MOTOR_TOGGLE,       // Turn motors on/off
    FORWARD,           // Move forward
    BACKWARD,          // Move backward
    LEFT,              // Move left
    RIGHT,             // Move right
    UP,                // Move up
    DOWN,              // Move down
    TURN_LEFT,         // Turn left
    TURN_RIGHT         // Turn right
};

// Unified command structure
struct DroneCommand {
    CommandType type;
    double vx;  // Forward/backward velocity
    double vy;  // Left/right velocity
    double vz;  // Up/down velocity
    double vr;  // Rotation velocity
    bool speed_limit;  // Whether to apply speed limits

    // Constructor for custom velocity control
    DroneCommand(double _vx, double _vy, double _vz, double _vr, bool _speed_limit = true)
        : type(CommandType::CUSTOM), vx(_vx), vy(_vy), vz(_vz), vr(_vr), speed_limit(_speed_limit) {}

    // Constructor for predefined commands
    DroneCommand(CommandType _type, double _unit = 0.0)
        : type(_type), vx(0), vy(0), vz(0), vr(0), speed_limit(true) {
        switch (_type) {
            case CommandType::FORWARD:
                vy = _unit;
                break;
            case CommandType::BACKWARD:
                vy = -_unit;
                break;
            case CommandType::LEFT:
                vx = -_unit;
                break;
            case CommandType::RIGHT:
                vx = _unit;
                break;
            case CommandType::UP:
                vz = _unit;
                break;
            case CommandType::DOWN:
                vz = -_unit;
                break;
            case CommandType::TURN_LEFT:
                vr = -_unit;
                break;
            case CommandType::TURN_RIGHT:
                vr = _unit;
                break;
            default:
                break;
        }
    }
};

class Controller {
private:
    VideoRTMP* cap;
    int controlfd;
    struct sockaddr_in controlInfo;
    socklen_t control_addrlen;
    std::string phone_ip;
    bool rtmp_busy;
    double rtmp_processing_time;

    void initControlSocket();
    void connectToControlSignalServer();
    double setLimitation(double value, double min, double max);
    void sendControlSignal(const DroneCommand& cmd);

public:
    Controller(const std::string& ip = "192.168.0.158");
    ~Controller();

    // Unified control interface
    void executeCommand(const DroneCommand& cmd);
    
    // Convenience methods for common operations
    void takeoff() { executeCommand(DroneCommand(CommandType::TAKEOFF_LANDING)); }
    void land() { executeCommand(DroneCommand(CommandType::TAKEOFF_LANDING)); }
    void stopMotors() { executeCommand(DroneCommand(CommandType::MOTOR_TOGGLE)); }
    
    // Legacy methods that now use the unified interface
    void sendCommand(double vx, double vy, double vz, double vr, bool speed_limit = true) {
        executeCommand(DroneCommand(vx, vy, vz, vr, speed_limit));
    }
    
    void keyboardControl(int key, float unit) {
        switch(key) {
            case ' ': executeCommand(DroneCommand(CommandType::TAKEOFF_LANDING)); break;
            case 'm': case 'M': executeCommand(DroneCommand(CommandType::MOTOR_TOGGLE)); break;
            case 'i': case 'I': executeCommand(DroneCommand(CommandType::FORWARD, unit)); break;
            case 'k': case 'K': executeCommand(DroneCommand(CommandType::BACKWARD, unit)); break;
            case 'j': case 'J': executeCommand(DroneCommand(CommandType::LEFT, unit)); break;
            case 'l': case 'L': executeCommand(DroneCommand(CommandType::RIGHT, unit)); break;
            case 'w': case 'W': executeCommand(DroneCommand(CommandType::UP, unit)); break;
            case 's': case 'S': executeCommand(DroneCommand(CommandType::DOWN, unit)); break;
            case 'a': case 'A': executeCommand(DroneCommand(CommandType::TURN_LEFT, unit)); break;
            case 'd': case 'D': executeCommand(DroneCommand(CommandType::TURN_RIGHT, unit)); break;
        }
    }

    void keyboardControl();
    Mat getImage();
    float getvx();
    float getvy();
    float getvz();
    double getLatitude();
    double getLongitude();
    void setRTMPUrl(const std::string& url);
    void stop();
    bool isConnected();
    bool initializeConnection();

    // New RTMP-specific methods
    void setRTMPBusy(bool busy) {
        if (cap) {
            cap->setServerBusy(busy);
        }
    }
    
    void updateRTMPProcessingTime(double processingTime) {
        if (cap) {
            cap->updateServerProcessingTime(processingTime);
        }
    }
    
    double getRTMPServerDelay() const {
        return cap ? cap->getAvgServerDelay() : 0.0;
    }
    
    int getRTMPDroppedFrames() const {
        return cap ? cap->getDroppedFrames() : 0;
    }
};

#endif
