#pragma once
#include "stdafx.h"
#include "VideoSocketUDP_rn.h"

#define MCAST_PORT 11006
#define MCAST_ADDR "224.0.0.88"
#define LOCAL_ADDR "192.168.1.134"
#define MCAST_SIZE 90000

#if defined(__unix__)  || defined(__linux__)
#define INVALID_SOCKET -1 //For linux.
#define SOCKET_ERROR -1  // For linux.
#endif

VideoSocketUDP::VideoSocketUDP(void) {
    
#if defined(_WIN32) || defined(WIN32)
	WSADATA wsaVideo;
	int iResult = WSAStartup(MAKEWORD(2, 2), &wsaVideo); 
	if (iResult != 0) {
		cout << "Error creating WSA" << endl;
		return;
	}
#endif

	//Initial receiver socket:
	sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock == INVALID_SOCKET) {
		perror("Fail to create a socket.");
		exit(EXIT_FAILURE);
	}
	else cout << "UDP Connection established:" << sock << endl;

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

	

	if (bind(sock, (struct sockaddr*)&local_addr, sizeof(local_addr)) < 0) {
		perror("bind error");
		return;
	}

	
	struct ip_mreq mreq;
#if defined(_WIN32) || defined(WIN32)
	mreq.imr_multiaddr.S_un.S_addr = inet_addr(MCAST_ADDR);  // For Windows
#elif defined(__linux__) || defined(__unix__)
	mreq.imr_multiaddr.s_addr = inet_addr(MCAST_ADDR);  // For linux
#endif 
	//mreq.imr_interface.s_addr = inet_addr(LOCAL_ADDR);
	mreq.imr_interface.s_addr = htonl(INADDR_ANY);
#if defined(__uxix__) || defined(__linux__)
	if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {  // For linux
#elif defined(_WIN32) || defined(WIN32)
	if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, reinterpret_cast<char FAR *>(&mreq), sizeof(mreq)) < 0) { // For Windows
#endif
		cout << "set sock error2" << endl;
		return;
	}
}

//Start a UDP server for receiving image data from Android client.
Mat VideoSocketUDP::getImage() {

	unsigned char buffer[MCAST_SIZE]; 
	int size;                          
	int i_lastFrameIndex = -1;     
	socklen_t addr_len = 0;
    Mat img;
	//while (1) {
		addr_len = sizeof(local_addr);
		memset(buffer, 0, MCAST_SIZE);
		size = recvfrom(sock, (char*)buffer, MCAST_SIZE, 0, (struct sockaddr*)&local_addr, &addr_len);
		if (size >= 6000) {
			int t1 = (int)buffer[0]; //從buffer轉移前9個byte的frameIndex資料到t1~t9
			int t2 = (int)buffer[1];
			int t3 = (int)buffer[2];
			int t4 = (int)buffer[3];
			int t5 = (int)buffer[4];
			int t6 = (int)buffer[5];
			int t7 = (int)buffer[6];
			int t8 = (int)buffer[7];
			int t9 = (int)buffer[8];
			//int i_frameIndex = t1 * 100000000 + t2 * 10000000 + t3 * 1000000 + t4 * 100000 + t5 * 10000 + t6 * 1000 + t7 * 100 + t8 * 10 + t9; //目前frame的frameIndex
			int i_frameIndex = t2 * 10000000 + t3 * 1000000 + t4 * 100000 + t5 * 10000 + t6 * 1000 + t7 * 100 + t8 * 10 + t9; //目前frame的frameIndex
			// cout << "******i_frameIndex=" << i_frameIndex << endl;

			if (i_frameIndex > i_lastFrameIndex) { //過濾掉之前的frame
				i_lastFrameIndex = i_frameIndex;

				getImageInfo(buffer);

				unsigned char * buffer_video = new unsigned char[size]; //宣告buffer_video儲存實際的影像資料
				memcpy((void*)buffer_video, (void*)buffer, size);       //從buffer轉移全部資料到buffer_video
				memmove(buffer_video, buffer_video + 47, size);         //將第48個byte後的影像資料移到buffer_video (buffer_video只保留影像資料)
				// *********************** 2020/05/06 ***********************          
				img = decodeAndroidData(buffer_video, size);
                //break;
			}
		}
	//}
    return img;
}

void VideoSocketUDP::getImageInfo(unsigned char* buffer) {

				int t10 = (int)buffer[9]; //從buffer轉移frame的第10~14個byte的"xSpeed"資料到t10~t14
				int t11 = (int)buffer[10];
				int t12 = (int)buffer[11];
				int t13 = (int)buffer[12];
				int t14 = (int)buffer[13];
				float xSpeed = t11 * 10 + t12 + t13 * 0.1 + t14 * 0.01; //xSpeed
				if (t10 == 1) { //1表示xSpeed為負數
					xSpeed = xSpeed * -1;
				}
				vx = xSpeed;
				//if (xSpeed!=0) cout << "******xSpeed=" << xSpeed << endl;

				int t15 = (int)buffer[14]; //從buffer轉移frame的第15~19個byte的"ySpeed"資料到t15~t19
				int t16 = (int)buffer[15];
				int t17 = (int)buffer[16];
				int t18 = (int)buffer[17];
				int t19 = (int)buffer[18];
				float ySpeed = t16 * 10 + t17 + t18 * 0.1 + t19 * 0.01; //ySpeed
				if (t15 == 1) { //1表示ySpeed為負數
					ySpeed = ySpeed * -1;
				}
				vy = ySpeed;
				//if (ySpeed != 0) cout << "******ySpeed=" << ySpeed << endl;

				int t20 = (int)buffer[19]; //從buffer轉移frame的第20~24個byte的"zSpeed"資料到t20~t24
				int t21 = (int)buffer[20];
				int t22 = (int)buffer[21];
				int t23 = (int)buffer[22];
				int t24 = (int)buffer[23];
				float zSpeed = t21 * 10 + t22 + t23 * 0.1 + t24 * 0.01; //zSpeed
				if (t20 == 1) { //1表示ySpeed為負數
					zSpeed = zSpeed * -1;
				}
				vz = zSpeed;
				//if (zSpeed != 0) cout << "******zSpeed=" << zSpeed << endl;
				// *********************** 2019/03/18 ***********************
				int t25 = (int)buffer[24]; //從buffer轉移frame的第25~33個byte的緯度資料到t25~t33
				int t26 = (int)buffer[25];
				int t27 = (int)buffer[26];
				int t28 = (int)buffer[27];
				int t29 = (int)buffer[28];
				int t30 = (int)buffer[29];
				int t31 = (int)buffer[30];
				int t32 = (int)buffer[31];
				int t33 = (int)buffer[32];

				string str26 = to_string(t26);
				string str27 = to_string(t27);
				string str28 = to_string(t28);
				string str29 = to_string(t29);
				string str30 = to_string(t30);
				string str31 = to_string(t31);
				string str32 = to_string(t32);
				string str33 = to_string(t33);
				string s_droneLocationLat = str26 + str27 + "." + str28 + str29 + str30 + str31 + str32 + str33;

				if (t25 == 1) { //1表示緯度為負數
					s_droneLocationLat = "-" + s_droneLocationLat;
				}
				//cout << "******s_droneLocationLat=" << s_droneLocationLat << endl;

				int t34 = (int)buffer[33]; //從buffer轉移frame的第34~43個byte的經度資料到t34~t43
				int t35 = (int)buffer[34];
				int t36 = (int)buffer[35];
				int t37 = (int)buffer[36];
				int t38 = (int)buffer[37];
				int t39 = (int)buffer[38];
				int t40 = (int)buffer[39];
				int t41 = (int)buffer[40];
				int t42 = (int)buffer[41];
				int t43 = (int)buffer[42];

				string str35 = to_string(t35);
				string str36 = to_string(t36);
				string str37 = to_string(t37);
				string str38 = to_string(t38);
				string str39 = to_string(t39);
				string str40 = to_string(t40);
				string str41 = to_string(t41);
				string str42 = to_string(t42);
				string str43 = to_string(t43);
				string s_droneLocationLng = str35 + str36 + str37 + "." + str38 + str39 + str40 + str41 + str42 + str43;

				if (t34 == 1) { //1表示經度為負數
					s_droneLocationLng = "-" + s_droneLocationLng;
				}
				//cout << "******s_droneLocationLng=" << s_droneLocationLng << endl;
				// *********************** 2019/03/18 ***********************

				// *********************** 2020/05/06 ***********************
				int t44 = (int)buffer[43]; //從buffer轉移frame的第44~47個byte的"heading"資料到t44~t47
				int t45 = (int)buffer[44];
				int t46 = (int)buffer[45];
				int t47 = (int)buffer[46];
				int heading = t45 * 100 + t46 * 10 + t47; //heading
				if (t44 == 1) { //1表示xSpeed為負數
					heading = heading * -1;
				}
				//if (heading!=0) cout << "******heading=" << heading << endl;

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
	// Mat frame_video = imdecode(matVideo, IMREAD_COLOR);
	Mat frame_video = imdecode(matVideo, CV_LOAD_IMAGE_COLOR);
	if (frame_video.empty()) {
		cout << "1Empty frame!" << endl;
		//return;
	}
	free(buffer_video);
    return frame_video;
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