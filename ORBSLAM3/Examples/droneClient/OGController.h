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

#include "VideoSocketUDP.h"

using namespace std;
using namespace cv;

class Controller {
    int controlfd = 0;                                  
    struct sockaddr_in controlInfo;                     
    //int control_addrlen = sizeof(controlInfo);       
    socklen_t control_addrlen = sizeof(controlInfo);

    VideoSocketUDP* cap;
    
public:
	Controller();
	~Controller();
	void keyboardControl();
    void keyboardControl(int key, float unit=0.2);
    void takeoff();
    void land();
    void sendCommand(double vx, double vy, double vz, double vr, bool speed_limit = true);
    void sendCommand(Mat vel, bool speed_limit = true);
    Mat getImage();
    float getvx();
    float getvy();
    float getvz();
    double getLatitude();
    double getLongitude();

protected:
    void limitSpeed(Mat& vel);
    double setLimitation(double value, double min, double max);

private:
	void initControlSocket();
    void connectToControlSignalServer();
};

#endif