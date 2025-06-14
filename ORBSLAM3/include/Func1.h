#ifndef FUNC_H
#define FUNC_H

#define _BSD_SOURCE
#include <cstring>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <strings.h>
#include <string>
#include <math.h>

#include <iostream>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Data.h"

using namespace std;

cv::Mat GetRelative(cv::Mat &origin, cv::Mat &result){
    cv::Mat relativeRT = origin.inv()*result;
    return relativeRT;
}

cv::Mat TurnToWorldCoord(cv::Mat &pos){
    cv::Mat Tvec(3,1,CV_32F);
    cv::Mat Rvec(3,3,CV_32F);
    Rvec = pos.rowRange(0,3).colRange(0,3).t();
    Tvec = -Rvec*pos.rowRange(0,3).col(3);
    
    cv::Mat Twc(4,4,CV_32F);

    Twc.at<float>(0,0) = Rvec.at<float>(0,0);
    Twc.at<float>(0,1) = Rvec.at<float>(0,1);
    Twc.at<float>(0,2) = Rvec.at<float>(0,2);
    Twc.at<float>(0,3) = Tvec.at<float>(0,0);

    Twc.at<float>(1,0) = Rvec.at<float>(1,0);
    Twc.at<float>(1,1) = Rvec.at<float>(1,1);
    Twc.at<float>(1,2) = Rvec.at<float>(1,2);
    Twc.at<float>(1,3) = Tvec.at<float>(1,0);
    
    Twc.at<float>(2,0) = Rvec.at<float>(2,0);
    Twc.at<float>(2,1) = Rvec.at<float>(2,1);
    Twc.at<float>(2,2) = Rvec.at<float>(2,2);
    Twc.at<float>(2,3) = Tvec.at<float>(2,0);
    
    Twc.at<float>(3,0) = 0;
    Twc.at<float>(3,1) = 0;
    Twc.at<float>(3,2) = 0;
    Twc.at<float>(3,3) = 1;

    return Twc.clone();
}


float GetDist(cv::Mat &currentPos, cv::Mat &lastPos){
    cv::Mat Tvec(3,1,CV_32F);
    Tvec = currentPos.rowRange(0,3).col(3) - lastPos.rowRange(0,3).col(3);
    float dist = Tvec.at<float>(0,0)*Tvec.at<float>(0,0)
               + Tvec.at<float>(1,0)*Tvec.at<float>(1,0)
               + Tvec.at<float>(2,0)*Tvec.at<float>(2,0);
    return sqrt(dist);
}

float GetUnitDist(cv::Mat &pos){
    cv::Mat Tvec(3,1,CV_32F);
    Tvec = pos.rowRange(0,3).col(3);
    float dist = Tvec.at<float>(0,0)*Tvec.at<float>(0,0)
               + Tvec.at<float>(1,0)*Tvec.at<float>(1,0)
               + Tvec.at<float>(2,0)*Tvec.at<float>(2,0);
    return sqrt(dist);
}

int Max(cv::Mat &col){
    float c1 = fabs(col.at<float>(0));
    float c2 = fabs(col.at<float>(1));
    float c3 = fabs(col.at<float>(2));
    if((c1>c2) && (c1>c3)) return 1;
    else if((c2>c1) && (c2>c3)) return 2;
    else return 3;
}

cv::Vec3f getRotationDegree(cv::Mat &R){
    float sy = sqrt(R.at<float>(0,0)*R.at<float>(0,0) 
                  + R.at<float>(1,0)*R.at<float>(1,0));
    bool singular = sy < 1e-6;
    float x, y, z;
    if(!singular){
        x = atan2(R.at<float>(2,1), R.at<float>(2,2));
        y = atan2(R.at<float>(2,0), sy);
        z = atan2(R.at<float>(1,0), R.at<float>(0,0));
    }
    else{
        x = atan2(-R.at<float>(1,2), R.at<float>(1,1));
        y = atan2(-R.at<float>(2,0), sy);
        z = 0;
    }
    x = x*180/3.14;
    y = y*180/3.14;
    z = z*180/3.14;
    return cv::Vec3f(x, y, z);
}

cv::Mat CheckPositionFixing(int label, cv::Mat &globalPos, cv::Mat &predictPos, int inlier, int lastInlier, int dev_range, int dev_x, float scale){
    
    bool globalFix = true;
    for(int i=0 ; i<3 ; i++){
        for(int j=0 ; j<3 ; j++){
            if(globalPos.at<float>(i,j)*predictPos.at<float>(i,j)<0)
                globalFix = false;
        }
    }
    
    cv::Mat r_global = globalPos.rowRange(0,3).colRange(0,3);
    cv::Mat r_predict = predictPos.rowRange(0,3).colRange(0,3);
    
    cv::Vec3f degree_global = getRotationDegree(r_global);
    cv::Vec3f degree_predict = getRotationDegree(r_predict);
    
    bool checkR1 = fabs(degree_global[0] - degree_predict[0]) < 10;
    bool checkR2 = fabs(degree_global[1] - degree_predict[1]) < 10;
    bool checkR3 = fabs(degree_global[2] - degree_predict[2]) < 10;
    
    float dist = GetDist(globalPos, predictPos);
    dist = dist / scale;
    //bool checkT = (dist < 1.2); //outdoor
    
    //bool checkTEasy = (dist < 4.5); //outdoor
    bool checkTEasy = (dist < 2.25);
    bool checkT = (dist < 0.6); //indoor
    
    /*
    if(lastInlier == 0){
        label = Global;
        return globalPos;
    }
    */
    float image_area = 640*320;
    float dev_ellipse = (float)dev_range * 3.14;

    cout << "---pose choosing---" << endl;
    if(inlier < 5){
        cout << "\n---case0---\n" << endl; 
        
        return predictPos;
    }
    
    else if(dev_ellipse < (image_area/8) && dev_x < 40){
        label = Predict;
        cout << "\n---case1---\n" << endl; 
        return predictPos;
    } 

    else if(inlier > 20 && inlier < 250){  // 40
    //else if(inlier > 20){
        label = Global;
        cout << "\n---case2---\n" << endl; 
                
        /*if(checkTEasy)
            return globalPos;
        else
            return predictPos;*/
        
        return globalPos;
    }
    else if (inlier >= 250) {
        label = Predict;
        cout << "\n---case 1000---\n" << endl; 
        return predictPos;
    }
    else if(inlier < 15){
        label = Predict;
        cout << "\n---case3---\n" << endl; 
        return predictPos;
    }
    else if(checkR1 && checkR2 && checkR3 &&
       checkT && globalFix){ 
       // check if the dist between global pose and predicted pose not too far
        label = Global;
        cout << "\n---case4---\n" << endl; 
        return globalPos;
    }
    else{
        label = Predict;
        cout << "\n---case5---\n" << endl; 
        return predictPos.clone();
    }    
}
/*
void connection(int &sockfd, const char *server_ip){
    struct sockaddr_in dest;
    // create socket
    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    // initialize value in dest
    bzero(&dest, sizeof(dest));
    dest.sin_family = PF_INET;
    dest.sin_port = htons(8889);
    dest.sin_addr.s_addr = inet_addr(server_ip);
    
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 3000;
    // set timeout for the socket receiving
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    
    // Connecting to server
    connect(sockfd, (struct sockaddr*)&dest, sizeof(dest));
    cout << "TCP connection is built!" << endl;
}*/
void connection(int &sockfd, const char *server_ip, int port){
    struct sockaddr_in dest;
    // create socket
    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    // initialize value in dest
    bzero(&dest, sizeof(dest));
    dest.sin_family = PF_INET;
    dest.sin_port = htons(port);   //dest.sin_port = htons(8889);
    dest.sin_addr.s_addr = inet_addr(server_ip);
    
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 30000;     // tv.tv_usec = 3000;
    // set timeout for the socket receiving
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));    //***********************************
    
    // Connecting to server
    /*while (connect(sockfd, (struct sockaddr*)&dest, sizeof(dest)) == -1 ) {
        cout << "wait to connect to server ..." << endl;
    }*/
    int isConnect = connect(sockfd, (struct sockaddr*)&dest, sizeof(dest));
    if (isConnect == -1) cout << "ip: " << server_ip << " fail to connect !" << endl;
    
}

int recvInt(int sockfd){
    char buf[10];
    recv(sockfd, buf, sizeof(buf), 0);
    return atoi(buf);
}

int recvImg(int sockfd,cv::Mat& img) {
    // Receive image size
    int img_size = recvInt(sockfd);
    if (img_size<=0) return 0;
    // Receive image data
    std::vector<uchar> buffer(img_size);
    int bytes_received = 0;
    while (bytes_received < img_size) {
        int len = recv(sockfd, buffer.data() + bytes_received, 4096, 0);
        if (len < 0) {
            std::cerr << "Read error" << std::endl;
            return 0;
        }
        bytes_received += len;
    }
    // Convert image data to cv::Mat
    img = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    cout<<"finish received."<<endl;
    return 1;
}

void sendImg(int sockfd, cv::Mat &img){
    cv::Mat streamImg = img.clone();
    vector<uchar> imgBuf;
    imgBuf.clear();
    imencode(".jpg", streamImg, imgBuf);
    int bufSize = imgBuf.size();
    string sizeStr = to_string(bufSize);

    //////////////////////////////////////////////// original
    // char size[10];
    // strcpy(size, sizeStr.c_str());
    ////////////////////////////////////////////////

    ////////////////////////////////////////////////    11/1 PJ test
    char size[11];
    string temp_size = to_string(bufSize);
    string tmp_size;
    for (int i = 0; i < 10-temp_size.length(); ++i)
    {
        tmp_size += "0";
    }
    tmp_size += temp_size;
    // cout << "tmp_size!!!!!!!" << tmp_size << endl;
    strcpy(size, tmp_size.c_str());
    ////////////////////////////////////////////////
            
    send(sockfd, size, sizeof(size), 0); // send image size
    send(sockfd, &imgBuf[0], imgBuf.size(), 0); // send image
            
}

bool receiveDataFromServer(int sockfd, cv::Mat& img, string& text) {
    uint8_t flags;
    uint32_t image_length, text_length;

    // 读取1字节的标志位
    if (read(sockfd, &flags, 1) != 1) {
        // 错误处理
        return false;
    }

    // 读取图片和字符串的长度
    uint32_t lengths[2];
    size_t len_received = 0;
    while (len_received < sizeof(lengths)) {
        ssize_t n = read(sockfd, reinterpret_cast<char*>(lengths) + len_received, sizeof(lengths) - len_received);
        if (n <= 0) {
            // 错误处理
            return false;
        }
        len_received += n;
    }
    image_length = ntohl(lengths[0]);
    text_length = ntohl(lengths[1]);
    cout<< "image_length: "<<to_string(image_length)<<endl;
    // 根据标志位判断是否有图片和字符串
    if (flags & 0x01) {
        // 接收图片数据
        std::vector<uchar> image_data(image_length);
        size_t received = 0;
        while (received < image_length) {
            // cout<<'received:' << to_string(received)<<endl;
            ssize_t n = read(sockfd, image_data.data() + received, image_length - received);
            if (n <= 0) {
                // 错误处理
                return false;
            }
            received += n;
        }
        // 使用OpenCV解码16位深度图
        img = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            std::cerr << "Failed to decode image" << std::endl;
            return false;
        }
    }

    if (flags & 0x02) {
        // 接收字符串数据
        std::vector<char> text_data(text_length + 1, '\0'); // +1用于字符串结束符
        size_t received = 0;
        while (received < text_length) {
            ssize_t n = read(sockfd, text_data.data() + received, text_length - received);
            if (n <= 0) {
                // 错误处理
                return false;
            }
            received += n;
        }
        text=text_data.data();

    }
    cout<<"finish received."<<endl;
    return true;
}

void sendPose(int poseSockfd, cv:: Mat tempPos) {
    if (tempPos.empty()) return;
    float pos[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            pos[4*i+j] = tempPos.at<float>(i, j);
        }
    }
    //cout << "tempPos: " << tempPos << endl;
    //cout << "start sending pose" << endl;
    /*for (int i = 0; i < 4; i++) {
        send(poseSockfd, tempPos.ptr<float>(i), sizeof(float)*4, 0);
    }*/
    send(poseSockfd, pos, sizeof(pos), 0);
    //cout << "finish sending pose" << endl;
}

void sendState(int poseSockfd, cv:: Mat tempPos, float vx, float vy, float vz) {
    if (tempPos.empty()) return;
    float state[19];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[4*i+j] = tempPos.at<float>(i, j);
        }
    }
    state[16] = vx;
    state[17] = vy;
    state[18] = vz;
    //cout << "tempPos: " << tempPos << endl;
    //cout << "start sending pose" << endl;
    /*for (int i = 0; i < 4; i++) {
        send(poseSockfd, tempPos.ptr<float>(i), sizeof(float)*4, 0);
    }*/
    send(poseSockfd, state, sizeof(state), 0);
    //cout << "finish sending pose" << endl;
}

cv::Mat CrossProduct(cv::Mat &a, cv::Mat &b)
{
    cv::Mat c(3, 1, CV_32F);
    c.at<float>(0,0) = a.at<float>(0,1) * b.at<float>(0,2) - a.at<float>(0,2) * b.at<float>(0,1);
    c.at<float>(0,1) = a.at<float>(0,2) * b.at<float>(0,0) - a.at<float>(0,0) * b.at<float>(0,2);
    c.at<float>(0,2) = a.at<float>(0,0) * b.at<float>(0,1) - a.at<float>(0,1) * b.at<float>(0,0);

    return c;
}

double DotProduct(cv::Mat &a, cv::Mat &b)
{
    double result;
    result = a.at<float>(0,0) * b.at<float>(0,0) + a.at<float>(0,1) * b.at<float>(0,1) + a.at<float>(0,2) * b.at<float>(0,2);
    return result;
}

double Normalize(cv::Mat &v)
{
    double result;
    result = sqrt(v.at<float>(0,0) * v.at<float>(0,0) + v.at<float>(0,1) * v.at<float>(0,1) + v.at<float>(0,2) * v.at<float>(0,2));
    return result;
}

cv::Mat RotationMatrix(double angle, cv::Mat &u)
{
    double norm = Normalize(u);
    cv::Mat rotatinMatrix(3, 3, CV_32F);
    
    float u0 = u.at<float>(0,0) / norm;
    float u1 = u.at<float>(0,1) / norm;
    float u2 = u.at<float>(0,2) / norm;

    rotatinMatrix.at<float>(0,0) = cos(angle) + u0 * u0 * (1 - cos(angle));
    rotatinMatrix.at<float>(0,1) = u0 * u1 * (1 - cos(angle) - u2 * sin(angle));
    rotatinMatrix.at<float>(0,2) = u1 * sin(angle) + u0 * u2 * (1 - cos(angle));

    rotatinMatrix.at<float>(1,0) = u2 * sin(angle) + u0 * u1 * (1 - cos(angle));
    rotatinMatrix.at<float>(1,1) = cos(angle) + u1 * u1 * (1 - cos(angle));
    rotatinMatrix.at<float>(1,2) = -u0 * sin(angle) + u1 * u2 * (1 - cos(angle));
      
    rotatinMatrix.at<float>(2,0) = -u1 * sin(angle) + u0 * u2 * (1 - cos(angle));
    rotatinMatrix.at<float>(2,1) = u0 * sin(angle) + u1 * u2 * (1 - cos(angle));
    rotatinMatrix.at<float>(2,2) = cos(angle) + u2 * u2 * (1 - cos(angle));

    return rotatinMatrix;
}

cv::Mat Calculation(cv::Mat &vectorBefore, cv::Mat &vectorAfter)
{
    cv::Mat rotationAxis;
    double rotationAngle;
    cv::Mat rotationMatrix;
    rotationAxis = CrossProduct(vectorBefore, vectorAfter);
    rotationAngle = acos(DotProduct(vectorBefore, vectorAfter) / Normalize(vectorBefore) / Normalize(vectorAfter));
    rotationMatrix = RotationMatrix(rotationAngle, rotationAxis);
    return rotationMatrix;
}


void trajectoryRefinement(vector<imgInfo> &trackingRecord, int fixingPoint, int lastFixingPoint){
    // scale up the section + rotation
    cout << "\n$$$$$$$ start refinement $$$$$$$" << endl << endl;
    cout << "trajectory refinement" << endl;
    cout << "fixingPoint: " << fixingPoint << endl;
    cout << "lastFixingPoint: " << lastFixingPoint << endl;
    
    vector<cv::Mat> RelativeRT;

    float totalDist = GetDist(trackingRecord[fixingPoint].pos,
                              trackingRecord[lastFixingPoint].pos);
    float realDist = GetDist(trackingRecord[fixingPoint-1].pos,
                             trackingRecord[lastFixingPoint].pos);
    // get scale
    float scale = totalDist / realDist;
    
    // store relative poses and multiply scale
    for(int i=lastFixingPoint+1 ; i<fixingPoint ; i++){
        cv::Mat current = trackingRecord[i].pos.clone();
        cv::Mat last = trackingRecord[i-1].pos.clone();
        
        if(current.empty() || last.empty()){
            cout << "pose is empty!" << endl;
            continue;
        }
        cv::Mat relative = trackingRecord[i-1].pos.inv() * trackingRecord[i].pos;
        relative.rowRange(0,3).col(3) *= scale;

        RelativeRT.push_back(relative);
    }

    // apply new relative poses
    for(int i=lastFixingPoint+1, k=0 ; i<fixingPoint ; i++, k++){
        cv::Mat goal;
        
        goal = trackingRecord[i-1].pos * RelativeRT[k];
        trackingRecord[i].pos = goal.clone();
    }

    cv::Mat vectorAfter(1, 3, CV_32F);
    cv::Mat vectorBefore(1, 3, CV_32F);
    vectorAfter.at<float>(0,0) = trackingRecord[fixingPoint].pos.at<float>(0,3) - trackingRecord[lastFixingPoint].pos.at<float>(0,3);
    vectorAfter.at<float>(0,1) = trackingRecord[fixingPoint].pos.at<float>(1,3) - trackingRecord[lastFixingPoint].pos.at<float>(1,3);
    vectorAfter.at<float>(0,2) = trackingRecord[fixingPoint].pos.at<float>(2,3) - trackingRecord[lastFixingPoint].pos.at<float>(2,3);
    
    vectorBefore.at<float>(0,0) = trackingRecord[fixingPoint-1].pos.at<float>(0,3) - trackingRecord[lastFixingPoint].pos.at<float>(0,3);
    vectorBefore.at<float>(0,1) = trackingRecord[fixingPoint-1].pos.at<float>(1,3) - trackingRecord[lastFixingPoint].pos.at<float>(1,3);
    vectorBefore.at<float>(0,2) = trackingRecord[fixingPoint-1].pos.at<float>(2,3) - trackingRecord[lastFixingPoint].pos.at<float>(2,3);
    
    cv::Mat rotatinMatrix = Calculation(vectorBefore, vectorAfter);
    
    vector<cv::Mat> RT;

    for(int i=lastFixingPoint+1 ; i<fixingPoint ; i++ ){
        cv::Mat temp = trackingRecord[i-1].pos.inv() * trackingRecord[i].pos;
        RT.push_back(temp.clone());
    }

    for(int i=lastFixingPoint+1, k=0 ; i<fixingPoint ; i++, k++){
        cv::Mat goal(3, 1, CV_32F);
        cv::Point3f start( trackingRecord[i].pos.at<float>(0,3),
                           trackingRecord[i].pos.at<float>(1,3),
                           trackingRecord[i].pos.at<float>(2,3));

        cv::Mat temp(3, 1, CV_32F);
        temp.at<float>(0,0) = trackingRecord[i].pos.at<float>(0,3) - trackingRecord[lastFixingPoint].pos.at<float>(0,3);
        temp.at<float>(0,1) = trackingRecord[i].pos.at<float>(1,3) - trackingRecord[lastFixingPoint].pos.at<float>(1,3);
        temp.at<float>(0,2) = trackingRecord[i].pos.at<float>(2,3) - trackingRecord[lastFixingPoint].pos.at<float>(2,3);
        
        goal = rotatinMatrix * temp;
        
        trackingRecord[i].pos.at<float>(0,3) = goal.at<float>(0,0) + trackingRecord[lastFixingPoint].pos.at<float>(0,3);
        trackingRecord[i].pos.at<float>(1,3) = goal.at<float>(0,1) + trackingRecord[lastFixingPoint].pos.at<float>(1,3);
        trackingRecord[i].pos.at<float>(2,3) = goal.at<float>(0,2) + trackingRecord[lastFixingPoint].pos.at<float>(2,3);

    }
    cout << "$$$$$$$ refinement end $$$$$$$\n" << endl;
}

void saveTrajectory(string filename, vector<imgInfo> &trackingRecord){

    ofstream poseFile;
    poseFile.open(filename);
    for(int i=0 ; i<trackingRecord.size() ; i++){
        if(trackingRecord[i].pos.empty())
            continue;
        else{
            // store index and Tvec
            poseFile << trackingRecord[i].index 
            << " " << trackingRecord[i].pos.at<float>(0,3)  
            << " " << trackingRecord[i].pos.at<float>(1,3)
            << " " << trackingRecord[i].pos.at<float>(2,3) 
            // store Rvec
            << " " << trackingRecord[i].pos.at<float>(0,0) 
            << " " << trackingRecord[i].pos.at<float>(0,1) 
            << " " << trackingRecord[i].pos.at<float>(0,2) 
            << " " << trackingRecord[i].pos.at<float>(1,0) 
            << " " << trackingRecord[i].pos.at<float>(1,1) 
            << " " << trackingRecord[i].pos.at<float>(1,2) 
            << " " << trackingRecord[i].pos.at<float>(2,0) 
            << " " << trackingRecord[i].pos.at<float>(2,1) 
            << " " << trackingRecord[i].pos.at<float>(2,2) 
            << endl;
        }
    }
    poseFile.close();
}
 
void getRealScale(string filename, float &real_scale){
    ifstream scaleFile;
    scaleFile.open(filename);
    scaleFile >> real_scale;
    scaleFile.close();
}

void setPath(string pathFile, vector<cv::Mat> &move_path) {
    ifstream fin(pathFile);
    if (!fin) {
        cout << "path file is not open!" << endl;
    }
    string line;
    while (getline(fin, line)) {
        if (line.empty()) break;
        stringstream ss(line);
        cv::Mat keyPoint(4, 1, CV_32F);
        for (int i = 0; i < 4; i++) {
            ss >> keyPoint.at<float>(i, 0);
        }
        move_path.push_back(keyPoint); 
    }
    fin.close();

    cout << "************ path ************" << endl;
    for (size_t i = 0; i < move_path.size(); i++) {
        cout << move_path[i] << endl;
    }
}

#endif
