#include "stdafx.h"
#include "Controller.h"
#include "VideoSocketUDP.h"

#if defined(__unix__)  || defined(__linux__)
#define INVALID_SOCKET -1 //For linux.
#define SOCKET_ERROR -1  // For linux.
#endif

Controller::Controller(const std::string& ip) : cap(nullptr), controlfd(-1), phone_ip(ip) {
}

// Controller::Controller() {
//     /*controlfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
//     if (controlfd == -1) {
// 		perror("Fail to create a socket.");
// 		//WSACleanup();
// 		//exit(EXIT_FAILURE);
// 	}
// 	else cout << "TCP client Connection established" << endl;

//     //bzero(&controlInfo,control_addrlen);    //For linux.
// 	memset(&controlInfo, 0, control_addrlen); //For Windows.

// 	controlInfo.sin_family = PF_INET;
//     //mobile_serverInfo.sin_addr.s_addr = inet_addr("192.168.0.175");
// 	if (inet_pton(AF_INET, "192.168.1.30", &(controlInfo.sin_addr)) == 0) { // IPv4  
// 		perror("socket");
// 	}
//     controlInfo.sin_port = htons(8080);*/

// 	cap = new VideoSocketUDP(); 

// // 	controlfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
// //     if (controlfd == -1) {
// // 		perror("Fail to create a socket.");
// // 	}
// // 		// else cout << "TCP client Connection established: " << controlfd << endl;

// // #if defined(__unix__) || defined(__linux__)
// // 	bzero(&controlInfo,control_addrlen);    //For linux.
// // #elif defined(_WIN32) || defined(WIN32)
// // 	memset(&controlInfo, 0, control_addrlen); //For Windows.
// // #endif

// // 	controlInfo.sin_family = PF_INET;
// // 	if (inet_pton(AF_INET, "192.168.1.224", &(controlInfo.sin_addr)) == 0) {  
// // 		perror("socket");
// // 	}
// // 	controlInfo.sin_port = htons(8080);

// // 	if (connect(controlfd, (struct sockaddr *)&controlInfo, control_addrlen) == SOCKET_ERROR) {
// // 		cout << "fail to connect to ControlSignal Server" << endl;
// // #if defined(__unix__) || defined(__linux__)
// // 		close(controlfd); // For linux.
// // #elif defined(_WIN32) || defined(WIN32)
// // 		closesocket(controlfd); //For Windows.
// // #endif
// // 	}
// }

Controller::~Controller() {
    if (controlfd != -1) {
#if defined(__unix__) || defined(__linux__)
        close(controlfd);
#elif defined(_WIN32) || defined(WIN32)
        closesocket(controlfd);
#endif
        controlfd = -1;
    }
}


void Controller::setRTMPUrl(const std::string& url) {
    // Clean up existing cap if any
    if (cap != nullptr) {
        cap->stop();
        delete cap;
        cap = nullptr;
    }
    
    try {
        cap = new VideoRTMP(url);
    } catch (const std::exception& e) {
        std::cerr << "[Controller] Failed to create RTMP connection: " << e.what() << std::endl;
        cap = nullptr;
    }
}

void Controller::initControlSocket() {
    // Only create socket if not already connected
    if (controlfd != -1) {
        return;
    }

    controlfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (controlfd == -1) {
        std::cerr << "[Controller] Failed to create control socket" << std::endl;
        return;
    }

#if defined(__unix__) || defined(__linux__)
    bzero(&controlInfo, control_addrlen);    //For linux.
#elif defined(_WIN32) || defined(WIN32)
    memset(&controlInfo, 0, control_addrlen); //For Windows.
#endif

    controlInfo.sin_family = PF_INET;
    // Connect to phone's IP for control commands
    if (inet_pton(AF_INET, phone_ip.c_str(), &(controlInfo.sin_addr)) == 0) {  
        std::cerr << "[Controller] Invalid phone IP address: " << phone_ip << std::endl;
#if defined(__unix__) || defined(__linux__)
        close(controlfd);
#elif defined(_WIN32) || defined(WIN32)
        closesocket(controlfd);
#endif
        controlfd = -1;
        return;
    }
    controlInfo.sin_port = htons(8080);  // Phone's control port

    // Set socket options for better reliability
    struct timeval timeout;
    timeout.tv_sec = 3;  // 3 seconds timeout
    timeout.tv_usec = 0;
    if (setsockopt(controlfd, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)) < 0) {
        std::cerr << "[Controller] Failed to set socket receive timeout" << std::endl;
    }
    if (setsockopt(controlfd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout, sizeof(timeout)) < 0) {
        std::cerr << "[Controller] Failed to set socket send timeout" << std::endl;
    }
}

void Controller::connectToControlSignalServer() {
    // This function establishes TCP connection to Android app for sending control commands
    if (connect(controlfd, (struct sockaddr *)&controlInfo, control_addrlen) == SOCKET_ERROR) {
        std::cerr << "[Controller] Failed to connect to Android app for control commands" << std::endl;
#if defined(__unix__) || defined(__linux__)
        close(controlfd);
#elif defined(_WIN32) || defined(WIN32)
        closesocket(controlfd);
#endif
        throw std::runtime_error("Failed to establish control connection with Android app");
    }
    std::cout << "[Controller] Successfully connected to Android app for control commands" << std::endl;
}

void Controller::sendCommand(double vx, double vy, double vz, double vr, bool speed_limit) {
    vz *= 2;
    if (speed_limit) {
        vx = setLimitation(vx, 0, 0.6);
        vy = setLimitation(vy, 0, 0.6);
        vz = setLimitation(vz, 0, 0.6);
        vr = setLimitation(vr, 0, 8);
    }

    string controlSignal = "2,Custom," + to_string(vy) + "," + to_string(vx) + "," + to_string(vr) + "," + to_string(vz);
    std::cout << "[Controller] Sending control signal: " << controlSignal << std::endl;

    // If not connected, try to connect
    if (!isConnected()) {
        initControlSocket();
        try {
            connectToControlSignalServer();
        } catch (const std::exception& e) {
            std::cerr << "[Controller] Failed to establish connection: " << e.what() << std::endl;
            return;
        }
    }

    // Try to send command on existing connection
    try {
        int iResult = send(controlfd, controlSignal.c_str(), (int)strlen(controlSignal.c_str()), 0);
        if (iResult == SOCKET_ERROR) {
            throw std::runtime_error("Send failed");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[Controller] Error sending command: " << e.what() << std::endl;
        // Only close the socket if there's an error
        if (controlfd != -1) {
#if defined(__unix__) || defined(__linux__)
            close(controlfd);
#elif defined(_WIN32) || defined(WIN32)
            closesocket(controlfd);
#endif
            controlfd = -1;
        }
    }
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
        try {
            // Use existing connection if possible
            if (!isConnected()) {
                initControlSocket();
                connectToControlSignalServer();
            }
            std::cout << "[Controller] Sending keyboard command: " << controlSignal << std::endl;
            int iResult = send(controlfd, controlSignal.c_str(), (int)strlen(controlSignal.c_str()), 0);
            if (iResult == SOCKET_ERROR) {
                throw std::runtime_error("Failed to send keyboard command");
            }
        }
        catch (const std::exception& e) {
            std::cerr << "[Controller] Error in keyboard control: " << e.what() << std::endl;
            // Only close connection on error
            if (controlfd != -1) {
#if defined(__unix__) || defined(__linux__)
                close(controlfd);
#elif defined(_WIN32) || defined(WIN32)
                closesocket(controlfd);
#endif
                controlfd = -1;
            }
        }
    }
}

void Controller::limitSpeed(Mat& vel) {
    // for (int i = 0; i < 3; i++) {
    // 	if (i == 0) vel.at<double>(i, 0) = setLimitation(vel.at<double>(i, 0), 0, 0.8);   // forward
    // 	else if(i == 1) vel.at<double>(i, 0) = setLimitation(vel.at<double>(i, 0), 0, 0.5);  
    // }
    vel.at<double>(0, 0) = setLimitation(vel.at<double>(0, 0), 0, 0.8);
    vel.at<double>(1, 0) = setLimitation(vel.at<double>(1, 0), 0, 0.8);
    vel.at<double>(2, 0) = setLimitation(vel.at<double>(2, 0), 0, 0.8);
    vel.at<double>(3, 0) = setLimitation(vel.at<double>(3, 0), 0, 10);
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
    std::cout << "[Controller] Initiating takeoff..." << std::endl;
    try {
        initControlSocket();
        connectToControlSignalServer();
        int iResult = send(controlfd, controlSignal.c_str(), (int)strlen(controlSignal.c_str()), 0);
        if (iResult == SOCKET_ERROR) {
            throw std::runtime_error("Failed to send takeoff command");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[Controller] Takeoff error: " << e.what() << std::endl;
    }
#if defined(__unix__) || defined(__linux__)
    close(controlfd);
#elif defined(_WIN32) || defined(WIN32)
    closesocket(controlfd);
#endif
    controlfd = -1;
}

void Controller::land() {
    string controlSignal = "1,TakeoffOrLanding,0,0,0,0";
    std::cout << "[Controller] Initiating landing..." << std::endl;
    try {
        initControlSocket();
        connectToControlSignalServer();
        int iResult = send(controlfd, controlSignal.c_str(), (int)strlen(controlSignal.c_str()), 0);
        if (iResult == SOCKET_ERROR) {
            throw std::runtime_error("Failed to send landing command");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[Controller] Landing error: " << e.what() << std::endl;
    }
#if defined(__unix__) || defined(__linux__)
    close(controlfd);
#elif defined(_WIN32) || defined(WIN32)
    closesocket(controlfd);
#endif
    controlfd = -1;
}

void Controller::stop() {
    if (cap != nullptr) {
        cap->stop();
        delete cap;
        cap = nullptr;
    }
    
    // Close control socket if open
    if (controlfd != -1) {
#if defined(__unix__) || defined(__linux__)
        close(controlfd);
#elif defined(_WIN32) || defined(WIN32)
        closesocket(controlfd);
#endif
        controlfd = -1;
    }
}

bool Controller::isConnected() {
    if (controlfd == -1) {
        return false;
    }
    
    // Check if the socket is still valid by trying to peek at incoming data
    char buf;
    int result = recv(controlfd, &buf, 1, MSG_PEEK | MSG_DONTWAIT);
    
    if (result == 0) {  // Connection closed by peer
        close(controlfd);
        controlfd = -1;
        return false;
    }
    else if (result < 0) {
        // Check if error is because of no data (which is fine) or a real error
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            close(controlfd);
            controlfd = -1;
            return false;
        }
    }
    
    return true;
}

bool Controller::initializeConnection() {
    try {
        initControlSocket();
        connectToControlSignalServer();
        return isConnected();
    }
    catch (const std::exception& e) {
        std::cerr << "[Controller] Connection initialization failed: " << e.what() << std::endl;
        return false;
    }
}

Mat Controller::getImage() {
	return cap->getImage();
}


float Controller::getvx() {
	return cap->getvx();
}

float Controller::getvy() {
	return cap->getvy();
}

float Controller::getvz() {
	return cap->getvz();
}

double Controller::getLatitude() {
	return getLatitude();
}

double Controller::getLongitude() {
	return getLongitude();
}