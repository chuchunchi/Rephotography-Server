#include "stdafx.h"
#include "Controller.h"

#if defined(__unix__)  || defined(__linux__)
#define INVALID_SOCKET -1 //For linux.
#define SOCKET_ERROR -1  // For linux.
#endif

Controller::Controller() {
    /*controlfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (controlfd == -1) {
		perror("Fail to create a socket.");
		//WSACleanup();
		//exit(EXIT_FAILURE);
	}
	else cout << "TCP client Connection established" << endl;

    //bzero(&controlInfo,control_addrlen);    //For linux.
	memset(&controlInfo, 0, control_addrlen); //For Windows.

	controlInfo.sin_family = PF_INET;
    //mobile_serverInfo.sin_addr.s_addr = inet_addr("192.168.0.175");
	if (inet_pton(AF_INET, "192.168.1.30", &(controlInfo.sin_addr)) == 0) { // IPv4  
		perror("socket");
	}
    controlInfo.sin_port = htons(8080);*/

	cap = new VideoSocketUDP(); 

// 	controlfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
//     if (controlfd == -1) {
// 		perror("Fail to create a socket.");
// 	}
// 		// else cout << "TCP client Connection established: " << controlfd << endl;

// #if defined(__unix__) || defined(__linux__)
// 	bzero(&controlInfo,control_addrlen);    //For linux.
// #elif defined(_WIN32) || defined(WIN32)
// 	memset(&controlInfo, 0, control_addrlen); //For Windows.
// #endif

// 	controlInfo.sin_family = PF_INET;
// 	if (inet_pton(AF_INET, "192.168.1.224", &(controlInfo.sin_addr)) == 0) {  
// 		perror("socket");
// 	}
// 	controlInfo.sin_port = htons(8080);

// 	if (connect(controlfd, (struct sockaddr *)&controlInfo, control_addrlen) == SOCKET_ERROR) {
// 		cout << "fail to connect to ControlSignal Server" << endl;
// #if defined(__unix__) || defined(__linux__)
// 		close(controlfd); // For linux.
// #elif defined(_WIN32) || defined(WIN32)
// 		closesocket(controlfd); //For Windows.
// #endif
// 	}
}

Controller::~Controller() {
#if defined(__unix__) || defined(__linux__)
        close(controlfd); // For linux.
#elif defined(_WIN32) || defined(WIN32)
		closesocket(controlfd); //For Windows.
#endif
}

void Controller::initControlSocket() {
    controlfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (controlfd == -1) {
		perror("Fail to create a socket.");
	}
	// else cout << "TCP client Connection established: " << controlfd << endl;

#if defined(__unix__) || defined(__linux__)
    bzero(&controlInfo,control_addrlen);    //For linux.
#elif defined(_WIN32) || defined(WIN32)
	memset(&controlInfo, 0, control_addrlen); //For Windows.
#endif

	controlInfo.sin_family = PF_INET;
	if (inet_pton(AF_INET, "192.168.0.158", &(controlInfo.sin_addr)) == 0) {  
		perror("socket");
	}
    controlInfo.sin_port = htons(8080);
}

void Controller::connectToControlSignalServer() {
	// cout << "connect " << controlfd << '\n';
	if (connect(controlfd, (struct sockaddr *)&controlInfo, control_addrlen) == SOCKET_ERROR) {
		cout << "fail to connect to ControlSignal Server" << endl;
#if defined(__unix__) || defined(__linux__)
		close(controlfd); // For linux.
#elif defined(_WIN32) || defined(WIN32)
		closesocket(controlfd); //For Windows.
#endif
	}
	// else printf("Connect to ControlSignal Server successfully...\n");
}

void Controller::sendCommand(double vx, double vy, double vz, double vr, bool speed_limit) {  // vx : FB   vy : RL   vz : UD   vr : rot
    if (speed_limit) {
    	vx = setLimitation(vx, 0, 0.6);
    	vy = setLimitation(vy, 0, 0.5);
    	vz = setLimitation(vz, 0, 0.3);
    	vr = setLimitation(vr, 0, 8);
    }
    if (0.06 <= vx && vx <= 0.1) vx = 0.1;
    if (0.06 <= vy && vy <= 0.1) vy = 0.1;
    if (0.06 <= vz && vz <= 0.1) vz = 0.1;
    string controlSignal = "2,Custom," + to_string(vy) + "," + to_string(vx) + "," + to_string(vr) + "," + to_string(vz);
    std::cout << "controlSignal: "  << controlSignal << endl;
    initControlSocket();
	connectToControlSignalServer();
	int iResult = send(controlfd, controlSignal.c_str(), (int)strlen(controlSignal.c_str()), 0);
#if defined(__unix__) || defined(__linux__)
	close(controlfd); // For linux.
#elif defined(_WIN32) || defined(WIN32)
	closesocket(controlfd); //For Windows.
#endif
	//this_thread::sleep_for(chrono::milliseconds(150));
}


void Controller::keyboardControl() {
	Mat img = imread("0.jpg");
	int key = 255;    // key = -1
	if (img.empty()) {
		cout << "control image is not open!" << endl; 
	} else {
		imshow ("control", img);
        key = waitKey(33);
        if (key != 255) {   // key == -1
        	while (1) {
        		if (key == 'v' || key == 'V') break;
				keyboardControl(key);
        		imshow ("control", img);
        		key = waitKey(33);
        	}
        }
	}
}

void Controller::keyboardControl(int key, float unit) {
	string controlSignal = "Empty";
    if (key == ' ') 
	    controlSignal = "1,TakeoffOrLanding,0,0,0,0";     //space = takeoff or land
    else if (key == 'm' || key == 'M')    
	    controlSignal = "1,turnOnOrTurnOffMotors,0,0,0,0";     //M = turn on or turn off Motors.
    else if (key == 'i' || key == 'I')    
	    controlSignal = "1,pitchDown,0," + to_string(unit) + ",0,0";     //I = move forward
    else if (key == 'k' || key == 'K')   
	    controlSignal = "1,pitchUp,0,-" + to_string(unit) + ",0,0";     //K = move backwards
    else if (key == 'j' || key == 'J')    
	    controlSignal = "1,rollLeft,-" + to_string(unit) + ",0,0,0";     //J = move left
    else if (key == 'l' || key == 'L')    
	    controlSignal = "1,rollRight," + to_string(unit) + ",0,0,0";     //L = move right
    else if (key == 'w' || key == 'W')    
	    controlSignal = "1,up,0,0,0," + to_string(unit);     //W = move up
    else if (key == 's' || key == 'S')    
	    controlSignal = "1,down,0,0,0,-" + to_string(unit);     //S = move down
    else if (key == 'a' || key == 'A')    
	    controlSignal = "1,yawLeft,0,0,-" + to_string(unit) + ",0";     //A = turn left
    else if (key == 'd' || key == 'D')    
	    controlSignal = "1,yawRight,0,0," + to_string(unit) + ",0";     //D = turn right


    if (controlSignal != "Empty") {
	    initControlSocket();
	    connectToControlSignalServer();
	    cout << controlSignal << endl;
	    int iResult = send(controlfd, controlSignal.c_str(), (int)strlen(controlSignal.c_str()), 0); 

	    if (iResult == SOCKET_ERROR) {
#if defined(_WIN32) || defined(WIN32)
            wprintf(L"send failed with error: %d\n", WSAGetLastError());
		    closesocket(controlfd); 
		    WSACleanup();           
#elif defined(__linux__) || defined(__unix__)
		    close(controlfd);     
#endif
	    }
#if defined(__unix__) || defined(__linux__)
        close(controlfd); 
#elif defined(_WIN32) || defined(WIN32)
	    closesocket(controlfd); 
#endif
	}
}

void Controller::limitSpeed(Mat& vel) {
    // for (int i = 0; i < 3; i++) {
    // 	if (i == 0) vel.at<double>(i, 0) = setLimitation(vel.at<double>(i, 0), 0, 0.8);   // forward
    // 	else if(i == 1) vel.at<double>(i, 0) = setLimitation(vel.at<double>(i, 0), 0, 0.5);  
    // }
    vel.at<double>(0, 0) = setLimitation(vel.at<double>(0, 0), 0, 0.8);
    vel.at<double>(1, 0) = setLimitation(vel.at<double>(1, 0), 0, 0.5);
    vel.at<double>(2, 0) = setLimitation(vel.at<double>(2, 0), 0, 0.3);
    vel.at<double>(3, 0) = setLimitation(vel.at<double>(3, 0), 0, 15);
}

double Controller::setLimitation(double value, double min, double max) {
    if (abs(value) < min)
        return 0;
    else if (abs(value) > max)
        return max * (abs(value) / value);
    else
        return value;	
}

void Controller::takeoff() {
	string controlSignal = "1,TakeoffOrLanding,0,0,0,0";
    cout << "takeoff ..." << endl;
    initControlSocket();
	connectToControlSignalServer();
	int iResult = send(controlfd, controlSignal.c_str(), (int)strlen(controlSignal.c_str()), 0);
#if defined(__unix__) || defined(__linux__)
	close(controlfd); // For linux.
#elif defined(_WIN32) || defined(WIN32)
	closesocket(controlfd); //For Windows.
#endif
}

void Controller::land() {
	string controlSignal = "1,TakeoffOrLanding,0,0,0,0";
    cout << "land ..." << endl;
    initControlSocket();
	connectToControlSignalServer();
	int iResult = send(controlfd, controlSignal.c_str(), (int)strlen(controlSignal.c_str()), 0);
#if defined(__unix__) || defined(__linux__)
	close(controlfd); // For linux.
#elif defined(_WIN32) || defined(WIN32)
	closesocket(controlfd); //For Windows.
#endif
}

Mat Controller::getImage() {
	return cap->getImage();
}


float Controller::getvx() {
	return getvx();
}

float Controller::getvy() {
	return getvy();
}

float Controller::getvz() {
	return getvz();
}

double Controller::getLatitude() {
	return getLatitude();
}

double Controller::getLongitude() {
	return getLongitude();
}