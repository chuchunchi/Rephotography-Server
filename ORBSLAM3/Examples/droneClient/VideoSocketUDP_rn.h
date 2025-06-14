#ifndef _VIDEOSOCKETUDP_H_
#define _VIDEOSOCKETUDP_H_

#pragma once

//#define _WIN32_WINNT 0x0A00
#define _WINSOCK_DEPRECATED_NO_WARNINGS 1

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <sys/stat.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"

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

#include <chrono>
#include <string>


using namespace std;
using namespace cv;


class VideoSocketUDP {
	int i_lastFrameIndex = 0; 
	int sock = 0;   
	float vx, vy, vz;
	sockaddr_in local_addr;

public:
	VideoSocketUDP(void);
	~VideoSocketUDP(void);
	Mat getImage();
	float getvx();
	float getvy();
	float getvz();
	void getImageInfo(unsigned char* buffer);

private:
	Mat decodeAndroidData(unsigned char* buffer_video, int size);
};

#endif
