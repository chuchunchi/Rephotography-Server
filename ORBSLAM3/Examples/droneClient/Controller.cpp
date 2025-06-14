#include "stdafx.h"
#include "Controller.h"

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
        std::cerr << "[Controller] Socket already exists, closing old socket" << std::endl;
#if defined(__unix__) || defined(__linux__)
        close(controlfd);
#elif defined(_WIN32) || defined(WIN32)
        closesocket(controlfd);
#endif
        controlfd = -1;
    }

    controlfd = socket(AF_INET, SOCK_STREAM, 0);  // Changed IPPROTO_TCP to 0
    if (controlfd == -1) {
        std::cerr << "[Controller] Failed to create control socket: " << strerror(errno) << std::endl;
        return;
    }

    // Set socket to non-blocking mode
    int flags = fcntl(controlfd, F_GETFL, 0);
    if (flags == -1) {
        std::cerr << "[Controller] Failed to get socket flags: " << strerror(errno) << std::endl;
        close(controlfd);
        controlfd = -1;
        return;
    }
    if (fcntl(controlfd, F_SETFL, flags | O_NONBLOCK) == -1) {
        std::cerr << "[Controller] Failed to set non-blocking mode: " << strerror(errno) << std::endl;
        close(controlfd);
        controlfd = -1;
        return;
    }

    // Initialize address structure
    memset(&controlInfo, 0, sizeof(controlInfo));  // Changed from control_addrlen to sizeof(controlInfo)
    controlInfo.sin_family = AF_INET;  // Changed from PF_INET to AF_INET
    controlInfo.sin_port = htons(8080);

    // Connect to phone's IP for control commands
    if (inet_pton(AF_INET, phone_ip.c_str(), &(controlInfo.sin_addr)) <= 0) {  // Changed == 0 to <= 0
        std::cerr << "[Controller] Invalid phone IP address: " << phone_ip << std::endl;
        close(controlfd);
        controlfd = -1;
        return;
    }

    // Set socket options for better reliability
    int opt = 1;
    if (setsockopt(controlfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "[Controller] Failed to set SO_REUSEADDR: " << strerror(errno) << std::endl;
    }

    struct timeval timeout;
    timeout.tv_sec = 3;  // 3 seconds timeout
    timeout.tv_usec = 0;
    if (setsockopt(controlfd, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout, sizeof(timeout)) < 0) {
        std::cerr << "[Controller] Failed to set socket receive timeout: " << strerror(errno) << std::endl;
    }
    if (setsockopt(controlfd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout, sizeof(timeout)) < 0) {
        std::cerr << "[Controller] Failed to set socket send timeout: " << strerror(errno) << std::endl;
    }
}

void Controller::connectToControlSignalServer() {
    if (controlfd == -1) {
        std::cerr << "[Controller] Socket not initialized, initializing now" << std::endl;
        initControlSocket();
        if (controlfd == -1) {
            throw std::runtime_error("Failed to initialize socket");
        }
    }

    // This function establishes TCP connection to Android app for sending control commands
    int ret = connect(controlfd, (struct sockaddr *)&controlInfo, sizeof(controlInfo));  // Changed control_addrlen to sizeof(controlInfo)
    if (ret == SOCKET_ERROR) {
        if (errno == EINPROGRESS) {
            // Connection is in progress (non-blocking socket)
            fd_set write_fds;
            struct timeval timeout;
            
            FD_ZERO(&write_fds);
            FD_SET(controlfd, &write_fds);
            timeout.tv_sec = 5;  // 5 second timeout
            timeout.tv_usec = 0;
            
            ret = select(controlfd + 1, NULL, &write_fds, NULL, &timeout);
            if (ret == 0) {
                std::cerr << "[Controller] Connection timeout" << std::endl;
                close(controlfd);
                controlfd = -1;
                throw std::runtime_error("Connection timeout");
            } else if (ret < 0) {
                std::cerr << "[Controller] Select error: " << strerror(errno) << std::endl;
                close(controlfd);
                controlfd = -1;
                throw std::runtime_error("Select error");
            }
            
            // Check if connection was successful
            int error = 0;
            socklen_t len = sizeof(error);
            if (getsockopt(controlfd, SOL_SOCKET, SO_ERROR, &error, &len) < 0) {
                std::cerr << "[Controller] Getsockopt error: " << strerror(errno) << std::endl;
                close(controlfd);
                controlfd = -1;
                throw std::runtime_error("Getsockopt error");
            }
            if (error != 0) {
                std::cerr << "[Controller] Connection error: " << strerror(error) << std::endl;
                close(controlfd);
                controlfd = -1;
                throw std::runtime_error("Connection error");
            }
        } else {
            std::cerr << "[Controller] Failed to connect to Android app for control commands" << std::endl;
            std::cerr << "[Controller] Error code: " << errno << " - " << strerror(errno) << std::endl;
            std::cerr << "[Controller] Attempting to connect to IP: " << phone_ip << " on port 8080" << std::endl;
            
            close(controlfd);
            controlfd = -1;
            throw std::runtime_error("Failed to establish control connection with Android app");
        }
    }
    
    // Set socket back to blocking mode
    int flags = fcntl(controlfd, F_GETFL, 0);
    if (flags != -1) {
        fcntl(controlfd, F_SETFL, flags & ~O_NONBLOCK);
    }
    
    std::cout << "[Controller] Successfully connected to Android app for control commands" << std::endl;
}

void Controller::sendControlSignal(const DroneCommand& cmd) {
    string controlSignal;
    
    // Format control signal based on command type
    if (cmd.type == CommandType::CUSTOM) {
        // Apply speed limits if requested
        double vx = cmd.vx, vy = cmd.vy, vz = cmd.vz * 2, vr = cmd.vr;  // Note: vz is doubled
        if (cmd.speed_limit) {
            vx = setLimitation(vx, 0, 0.6);
            vy = setLimitation(vy, 0, 0.6);
            vz = setLimitation(vz, 0, 0.6);
            vr = setLimitation(vr, 0, 8);
        }
        controlSignal = "2,Custom," + to_string(vy) + "," + to_string(vx) + "," + to_string(vr) + "," + to_string(vz);
    } else {
        // Handle predefined commands
        switch (cmd.type) {
            case CommandType::TAKEOFF_LANDING:
                controlSignal = "1,TakeoffOrLanding,0,0,0,0";
                break;
            case CommandType::MOTOR_TOGGLE:
                controlSignal = "1,turnOnOrTurnOffMotors,0,0,0,0";
                break;
            case CommandType::FORWARD:
                controlSignal = "1,pitchDown,0," + to_string(cmd.vy) + ",0,0";
                break;
            case CommandType::BACKWARD:
                controlSignal = "1,pitchUp,0," + to_string(cmd.vy) + ",0,0";
                break;
            case CommandType::LEFT:
                controlSignal = "1,rollLeft," + to_string(cmd.vx) + ",0,0,0";
                break;
            case CommandType::RIGHT:
                controlSignal = "1,rollRight," + to_string(cmd.vx) + ",0,0";
                break;
            case CommandType::UP:
                controlSignal = "1,up,0,0,0," + to_string(cmd.vz);
                break;
            case CommandType::DOWN:
                controlSignal = "1,down,0,0,0," + to_string(cmd.vz);
                break;
            case CommandType::TURN_LEFT:
                controlSignal = "1,yawLeft,0,0," + to_string(cmd.vr) + ",0";
                break;
            case CommandType::TURN_RIGHT:
                controlSignal = "1,yawRight,0,0," + to_string(cmd.vr) + ",0";
                break;
        }
    }

    std::cout << "[Controller] Sending control signal: " << controlSignal << std::endl;

    // Ensure connection is established
    if (!isConnected()) {
        initControlSocket();
        try {
            connectToControlSignalServer();
        } catch (const std::exception& e) {
            std::cerr << "[Controller] Failed to establish connection: " << e.what() << std::endl;
            return;
        }
    }

    // Send the command
    try {
        int iResult = send(controlfd, controlSignal.c_str(), (int)strlen(controlSignal.c_str()), 0);
        if (iResult == SOCKET_ERROR) {
            throw std::runtime_error("Send failed");
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[Controller] Error sending command: " << e.what() << std::endl;
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

void Controller::executeCommand(const DroneCommand& cmd) {
    sendControlSignal(cmd);
}

void Controller::keyboardControl() {
    Mat img = imread("0.jpg");
    int key = 255;    // key = -1
    if (img.empty()) {
        cout << "control image is not open!" << endl; 
    } else {
        imshow("control", img);
        key = waitKey(33);
        if (key != 255) {   // key == -1
            while (1) {
                if (key == 'v' || key == 'V') break;
                keyboardControl(key, 0.3);  // Default unit value
                imshow("control", img);
                key = waitKey(33);
            }
        }
    }
}

double Controller::setLimitation(double value, double min, double max) {
    if (abs(value) < min)
        return 0;
    else if (abs(value) > max)
        return max * (abs(value) / value);
    else
        return value;	
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