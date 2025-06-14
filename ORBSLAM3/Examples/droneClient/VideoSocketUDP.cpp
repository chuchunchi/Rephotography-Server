#pragma once
//#include "stdafx.h"
#include "VideoSocketUDP.h"

#define MCAST_PORT 11006
#define MCAST_ADDR "224.0.0.88"
// #define LOCAL_ADDR "192.168.1.134"
#define MCAST_SIZE 100000     // 40000

#if defined(__unix__)  || defined(__linux__)
#define INVALID_SOCKET -1 //For linux.
#define SOCKET_ERROR -1  // For linux.
#endif

VideoSocketUDP::VideoSocketUDP(void) {
    
#if defined(_WIN32) || defined(WIN32)
    /*Start--For using socket in Windows*/
	WSADATA wsaVideo;
	int iResult = WSAStartup(MAKEWORD(2, 2), &wsaVideo); 
	if (iResult != 0) {
		cout << "Error creating WSA" << endl;
		return;
	}
	/*End--For using socket in Windows*/
#endif

	//Initial receiver socket:
	sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock == INVALID_SOCKET) {
		perror("Fail to create a socket.");
		exit(EXIT_FAILURE);
	}
	else cout << "UDP Connection established " << sock << endl;

	//Initial socket的連線資訊:
#if defined(__unix__) || defined(__linux__)
    bzero(&local_addr,sizeof(local_addr));    //For linux.
#elif defined(_WIN32) || defined(WIN32)
	memset(&local_addr, 0, sizeof(local_addr)); //For Windows.
#endif
	local_addr.sin_family = AF_INET;
	//local_addr.sin_addr.s_addr = inet_addr(LOCAL_ADDR);
	//local_addr.sin_addr.s_addr = inet_addr("192.168.1.134");
	local_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	local_addr.sin_port = htons(MCAST_PORT);
	int one = 1;
	setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
	setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &one, sizeof(one));


	if (bind(sock, (struct sockaddr*)&local_addr, sizeof(local_addr)) < 0) {
		perror("bind error");
		return;
	}

	/*int ttl = 0; 
	if (setsockopt(sock, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) < 0) {
		perror("IP_MULTICAST_TTL");
		return;
	}*/

	/*bool loop = 1;
	if (setsockopt(sock, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop)) < 0) {
		perror("set sock error");
		return;
	}*/

	
	struct ip_mreq mreq;
#if defined(_WIN32) || defined(WIN32)
	mreq.imr_multiaddr.S_un.S_addr = inet_addr(MCAST_ADDR);  // For Windows
#elif defined(__linux__) || defined(__unix__)
	mreq.imr_multiaddr.s_addr = inet_addr(MCAST_ADDR);  // For linux
#endif 
	//mreq.imr_interface.s_addr = inet_addr(LOCAL_ADDR);
	mreq.imr_interface.s_addr = htonl(INADDR_ANY);
	if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
		cout << "set sock error2" << endl;
		return;
	}
	cout << "ip: " << local_addr.sin_addr.s_addr << " : " << local_addr.sin_port << endl;
}

//Start a UDP server for receiving image data from Android client.
Mat VideoSocketUDP::getImage() {
	cout << "Inside get image\n";
	unsigned char buffer[MCAST_SIZE]; 
	int size;                          
	int i_lastFrameIndex = -1;         
	socklen_t addr_len = 0;
    Mat img;
	while (1) {
		addr_len = sizeof(local_addr);
		memset(buffer, 0, MCAST_SIZE);
		cout << "start receive" << endl;
		size = recvfrom(sock, (char*)buffer, MCAST_SIZE, 0, (struct sockaddr*)&local_addr, &addr_len);
		cout << "Receive size: " << size << endl;
		if (size >= 6000) {
			int t1 = (int)buffer[4]; 
			int t2 = (int)buffer[3];
			int t3 = (int)buffer[2];
			int t4 = (int)buffer[1];
			int t5 = (int)buffer[0];
			int i_frameIndex = t5 * 10000 + t4 * 1000 + t3 * 100 + t2 * 10 + t1;        
			cout << "******frameIndex=\n" << i_frameIndex << endl;

			if (i_frameIndex > i_lastFrameIndex) { 
				i_lastFrameIndex = i_frameIndex;
				
				int t6 = (int)buffer[5]; 
				int t7 = (int)buffer[6];
				int t8 = (int)buffer[7];
				int t9 = (int)buffer[8];
				int t10 = (int)buffer[9];
				float xSpeed = t7 * 10 + t8 + t9 * 0.1 + t10 * 0.01; 
				if (t6 == 1) { 
					xSpeed = xSpeed * -1;
				}
				vx = xSpeed;
				//if (xSpeed!=0) cout << "******xSpeed=" << xSpeed << endl;

				int t11 = (int)buffer[10]; 
				int t12 = (int)buffer[11];
				int t13 = (int)buffer[12];
				int t14 = (int)buffer[13];
				int t15 = (int)buffer[14];
				float ySpeed = t12 * 10 + t13 + t14 * 0.1 + t15 * 0.01; 
				if (t11 == 1) { 
					ySpeed = ySpeed * -1;
				}
				vy = ySpeed;
				//if (ySpeed != 0) cout << "******ySpeed=" << ySpeed << endl;

				int t16 = (int)buffer[15]; 
				int t17 = (int)buffer[16];
				int t18 = (int)buffer[17];
				int t19 = (int)buffer[18];
				int t20 = (int)buffer[19];
				float zSpeed = t17 * 10 + t18 + t19 * 0.1 + t20 * 0.01; 
				if (t16 == 1) { 
					zSpeed = zSpeed * -1;
				}
				vz = zSpeed;
				//if (zSpeed != 0) cout << "******zSpeed=" << zSpeed << endl;

				int t21 = (int)buffer[20]; //從buffer轉移frame的第21~29個byte的緯度資料到t21~t29
				int t22 = (int)buffer[21];
				int t23 = (int)buffer[22];
				int t24 = (int)buffer[23];
				int t25 = (int)buffer[24];
				int t26 = (int)buffer[25];
				int t27 = (int)buffer[26];
				int t28 = (int)buffer[27];
				int t29 = (int)buffer[28];
				droneLocationLat = t22 * 10 + t23 + t24 * 0.1 + t25 * 0.01 + t26 * 0.001 + t27 * 0.0001 + t28 * 0.00001 + t29 * 0.000001;
				if (t21 == 1) { //1表示ySpeed為負數
					droneLocationLat = droneLocationLat * -1;
				}
				//if (droneLocationLat != 0) cout << "******droneLocationLat=" << setprecision(10) << droneLocationLat << endl;

				int t30 = (int)buffer[29]; //從buffer轉移frame的第30~39個byte的經度資料到t30~t39
				int t31 = (int)buffer[30];
				int t32 = (int)buffer[31];
				int t33 = (int)buffer[32];
				int t34 = (int)buffer[33];
				int t35 = (int)buffer[34];
				int t36 = (int)buffer[35];
				int t37 = (int)buffer[36];
				int t38 = (int)buffer[37];
				int t39 = (int)buffer[38];
				droneLocationLng = t31 * 100 + t32 * 10 + t33 + t34 * 0.1 + t35 * 0.01 + t36 * 0.001 + t37 * 0.0001 + t38 * 0.00001 + t39 * 0.000001;
				if (t30 == 1) { //1表示ySpeed為負數
					droneLocationLng = droneLocationLng * -1;
				}
				//if (droneLocationLng != 0) cout << "******droneLocationLng=" << setprecision(10) << droneLocationLng << endl;

				unsigned char * buffer_video = new unsigned char[size]; //宣告buffer_video儲存實際的影像資料
				memcpy((void*)buffer_video, (void*)buffer, size);       //從buffer轉移全部資料到buffer_video
				memmove(buffer_video, buffer_video + 39, size);         //將第40個byte後的影像資料移到buffer_video (buffer_video只保留影像資料)        
				cout<<"Examples/UDP"<<endl;
                img = decodeAndroidData(buffer_video, size);
				
                // if (!isEmpty(img)) break; // PJ 2020/09/17
                return img;
			}
		}
	}
    return img;
}

VideoSocketUDP::~VideoSocketUDP(void) {
#if defined(__uxix__) || defined(__linux__)
    close(sock);     //For linux.
#elif defined(_WIN32) || defined(WIN32)
	closesocket(sock);  // For Windows
#endif
}

Mat VideoSocketUDP::decodeAndroidData(unsigned char* buffer_video, int size) {
	Mat matVideo = Mat(1, size, CV_8UC1, buffer_video);
	if (matVideo.empty()) {
		cout << "Empty Mat" << endl;
		//return;
	}
	Mat frame_video = imdecode(matVideo, CV_LOAD_IMAGE_COLOR);
	if (frame_video.empty()) {
		cout << "Empty frame!" << endl;
		//return;
	}
	free(buffer_video);
    return frame_video;
}

bool VideoSocketUDP::isEmpty(Mat img) {
	Mat gray_img;
	if(img.channels() == 3 || img.channels() == 4) cvtColor(img, gray_img, CV_BGR2GRAY);
	else return true;
	int black_count = 0;
	for (int r = 0; r < gray_img.rows; r++) {
		for (int c = 0; c < gray_img.cols; c++) {
			if (gray_img.at<uchar>(r, c) < 15) black_count++;
		}
	}
	if (black_count > (img.rows/2)*(img.cols/2)) return true;
	return false;
	// Mat checke_mpty_frame = (gray_img != Mat::zeros(img.rows, img.cols, CV_8U));
	// return countNonZero(checke_mpty_frame) < (img.rows/5)*(img.cols/5);
}

float VideoSocketUDP::getvx() {
	return vx;
}

float VideoSocketUDP::getvy() {
	return vy;
}

float VideoSocketUDP::getvz() {
	return vz;
}

double VideoSocketUDP::getLatitude() {
	return droneLocationLat;
}

double VideoSocketUDP::getLongitude() {
	return droneLocationLng;
}
