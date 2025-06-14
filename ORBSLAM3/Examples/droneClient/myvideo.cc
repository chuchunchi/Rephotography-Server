// ./client ../../test_video/test7.mp4 ./camera_params_new.yaml

#include <opencv2/opencv.hpp>
// ORB-SLAM的系统接口
#include "System.h"

#include <string>
#include <chrono>   // for time stamp
#include <iostream>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <sys/socket.h>
#include <netinet/in.h>
#include <Func1.h>

#include <thread>
#include <atomic>

#include "VideoSocketUDP.h"
#include "Controller.h"

using namespace std;

// 参数文件与字典文件
// 如果你系统上的路径不同，请修改它
string vocFile = "../../Vocabulary/ORBvoc.txt";
string parameter_file = "";
// 视频文件
string videoFile = "";
std::string receivedString;//KADFP结果

Controller * controller = new Controller();

void saveTrajectory(std::ofstream& file,double timestamp, Sophus::SE3f pose);
void sendFrame2server(int sockfd, int frame_idx, cv::Mat frame_img);


int main(int argc, char **argv) {

    if(argc != 3)
    {
        std::cout << "[Usage]: ./client ./video/path ./parameter/File " << endl; // ./client ../../test_video/lab.mp4 ./TUM1.yaml
        return 1;
    }
    videoFile = string(argv[1]);
    parameter_file = string(argv[2]);
    // ORB_SLAM3::System SLAM(vocFile, parameterFile, ORB_SLAM3::System::MONOCULAR, true);// 声明 ORB-SLAM3 系统
    ORB_SLAM3::System SLAM(vocFile, parameter_file, ORB_SLAM3::System::RGBD, true);// 声明 ORB-SLAM3 系统
    float imageScale = SLAM.GetImageScale();

    
    cv::VideoCapture video(videoFile);// 获取视频图像
    
    auto start = chrono::system_clock::now();// 记录系统时间

    bool mainloop_running_flag=true;
    
    int counter_loop = 0;//总循环计数器
    int counter_inputframe = 0;//输入视频帧数计数器
    int frame_idx = 0;
    int flag_match = 0;
    bool userControl=false;
    Eigen::Vector3f translation1(0.0f, 0.0f, 0.0f);
    Eigen::Quaternionf rotation1(1.0f, 0.0f, 0.0f, 0.0f); //单位四元数，表示无旋转
    Sophus::SE3f init_se3(rotation1, translation1);
  
    std::vector<cv::Mat> trajectorys_video;
    std::ofstream file("./logs/trajectorys.txt");

    //server端初始化
    int sockfd;
    connection(sockfd, "140.113.195.240", 9999);
    
    while (mainloop_running_flag){
        std::cout<< "total loop: " << ++counter_loop <<endl;
        cv::Mat imRGB;

        //决定图像来源 ————> 0:读取input影片一帧到frame，1：读取realtime无人机画面的一帧到frame
        video>>imRGB;
        
        if (imRGB.data == nullptr){//说明input影片的所有frame pose定位完成,flag设置为1
            std::cout<<"finish scanned. <------------------------------"<<endl;
            file.close();//关闭trajectorys.yaml写入
            break;
        }
        frame_idx++;
        cv::resize(imRGB, imRGB, cv::Size(640,480));
        
        if(counter_loop%3==0){
            // 传输到server端进行单目深度估计
            sendFrame2server(sockfd, frame_idx, imRGB);
            // 接收处理好的深度图
            cv::Mat imD;
            std::cout<<"start recv depth image."<<endl;
            usleep(1000);
            receiveDataFromServer(sockfd,imD,receivedString);
            // while(1) if(recvImg(sockfd,imD)==1) break;
            
            if(!imD.empty()){
                cv::resize(imD, imD, cv::Size(640,480));
                cv::imwrite("./logs/depth/f_"+ to_string(frame_idx) +".png", imD);
                
                // 对当前frame画面进行定位
                auto now = chrono::system_clock::now();
                auto timestamp = chrono::duration_cast<chrono::milliseconds>(now - start);
                double ts = double(timestamp.count())/1000.0;
                // Sophus::SE3f Tcw = SLAM.TrackMonocular(imRGB, ts);
                
                Sophus::SE3f Tcw = SLAM.TrackRGBD(imRGB, imD,ts);

                // Sophus::SE3f Twc = Tcw.inverse();
                saveTrajectory(file,ts, Tcw);
            }else{
                    cout<< "imD is empty." <<endl;
            }
        }
    }
    std::cout<<"Save Map."<<endl;
    close(sockfd);
    SLAM.Shutdown();
    return 0;
}

void saveTrajectory(std::ofstream& file, double timestamp, Sophus::SE3f pose){
    if (file.is_open()) {
        // 保存四元数和位移
        Eigen::Quaternionf q = pose.unit_quaternion();  // 获取四元数
        Eigen::Vector3f t = pose.translation();         // 获取平移向量
        
        // 按照四元数的 w, x, y, z 顺序保存
        file << to_string(timestamp) << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " << t.x() << " " << t.y() << " " << t.z()<< std::endl;
    } else {
        std::cerr << "Unable to open file: " << "./logs/trajectorys.txt" << std::endl;
    }
}

void sendFrame2server(int sockfd,int frame_idx,cv::Mat frame_img){
    std::cout << "client start sending..." << endl;
    std::cout<<"frame_idx: ===============>"<<frame_idx<<endl;
    // set index from int to chararray
    char frameIdx[11];
    // string temp_index = to_string(temp.index);
    string temp_index = to_string(frame_idx); 
    string tmp_idx;
    for (int i = 0; i < 10-temp_index.length(); ++i)
    {
        tmp_idx += "0";
    }
    tmp_idx += temp_index;
    // cout << frameIdx[0] << frameIdx[1] << endl;
    // std::cout << tmp_idx <<"<============================="<< endl;
    strcpy(frameIdx, tmp_idx.c_str());
    int sendIndex = send(sockfd, frameIdx, sizeof(frameIdx), 0);  // image index
    // compress image to a smaller vector
    sendImg(sockfd, frame_img);
    cout << endl << "finish sending..." << endl << "index: " << frameIdx << endl << endl;
}
