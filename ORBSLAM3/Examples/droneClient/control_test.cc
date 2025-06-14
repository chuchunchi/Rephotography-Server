// ./control_test trajectorys_1201_1 ./camera_params2.yaml

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
#include <arpa/inet.h> // ntohl, ntohs
#include <unistd.h>    // read, write
#include <Func1.h>

#include <thread>
#include <atomic>

#include "VideoSocketUDP.h"
#include "Controller.h"
#include "pid.h"

using namespace std;

// 参数文件与字典文件
// 如果你系统上的路径不同，请修改它
string vocFile = "../../Vocabulary/ORBvoc.txt";
string TrajFileName = "./logs/trajs/";
string parameter_file = "";
// 视频文件
string videoFile = "";
int KADFP_mode = 0;
int input_mode = 0;

Controller * controller = new Controller();

void connection(int& sockfd, const char* server_ip, int server_port, struct sockaddr_in servaddr);
int UserControl(int key, float v, float a);
void sendFrame2server(int sockfd, int frame_idx, cv::Mat frame_img);
void recv_depth(int sockfd,int frame_idx,cv::Mat& imD);
bool recvall(int sockfd, char *buf, int len);
bool sendall(int sockfd, const char *buf, int len);
bool recvImgData(int sockfd, std::vector<uchar>& buf, int len);


int main(int argc, char **argv) {
    if(argc != 3){
        std::cout << "[Usage]: ./control_test ./trajectorys/file/name ./parameter/File " << endl;
        return 1;
    }
    parameter_file = string(argv[2]);

    double fx = 517.417475;
    double fy = 585.870258;
    double cx = 336.769936;
    double cy = 207.547167;

    // 畸變係數
    double k1 = -0.18169;
    double k2 = 0.05021;
    double p1 = -0.00039;
    double p2 = 0.007179;
    double k3 = -0.03024;

    // 創建相機內參數矩陣
    cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // 創建畸變係數向量
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);

    ORB_SLAM3::System SLAM(vocFile, parameter_file, ORB_SLAM3::System::RGBD, true);// 声明 ORB-SLAM3 系统
    float imageScale = SLAM.GetImageScale();
    
    int curr_idx = -1;
    bool userControl=false;
  
    std::map<std::string, int> fly_command={
        {"original",-1},
        {"left",'j'}, {"right",'l'},//左j，右l
        {"forward",'i'}, {"backward",'k'},//前i，後k
        {"up",'w'}, {"down",'s'},//上w，下s
        {"lf",'u'},{"lb",'n'},{"rf",'o'},{"rb",'m'},//左前u，左後n，右前o，右後m
        {"lu",'q'},{"ld",'z'},{"ru",'e'},{"rd",'c'},//左上q，左下z，右上e，右下c
        {"fu",'t'},{"fd",'g'},{"bu",'f'},{"bd",'h'},//前上t，前下g，後上f，後下h
        {"turnleft",'a'}, {"turnright",'d'}//左轉a，右轉d
    };

    //socket初始化
    const char *server_ip = "140.113.195.240";
    int server_port = 8888;
    int sockfd;
    struct sockaddr_in servaddr;
    connection(sockfd, server_ip, server_port, servaddr);
    
    bool start_flag = false;
    bool end_flag = false;
    auto start = chrono::system_clock::now();// 记录系统时间
    //主循环
    while (1) {
        try{
            while(start_flag == false){
                if(SLAM.GetKey()==' '){
                    start_flag =true;
                    cout<<"[system] Mission start."<<endl;
                    usleep(50000);
                    auto start = chrono::system_clock::now();// 记录系统时间
                } 
            }
            cv::Mat im;
            cv::Mat imRGB;
            // 载入一张无人机畫面
            im = controller->getImage();
            if(im.empty()){
                std::cout <<"[system] ====> frame empty." << endl;
                continue;
            }
            // 去畸變
            cv::undistort(im, imRGB, K, distCoeffs);
            // cv::resize(imRGB, imRGB, cv::Size(640,480));
            curr_idx++;

            // 传输到server端进行单目深度估计
            sendFrame2server(sockfd, curr_idx, imRGB);

            // 接收处理好的深度图
            cv::Mat imD;
            recv_depth(sockfd, curr_idx, imD);
            if(!imD.empty()){
                auto now = chrono::system_clock::now();
                auto timestamp = chrono::duration_cast<chrono::milliseconds>(now - start);
                double ts = double(timestamp.count())/1000.0;
                Sophus::SE3f Tcw = SLAM.TrackRGBD(imRGB, imD, ts);
            }
            int key = -1;
            key = SLAM.GetKey();
            if(key == 27){ //char 27 = 'esc' 緊急情況按住esc及時退出系統控制改用鍵盤控制，防止炸機
                userControl = true;
                std::cout<<"[system] Switch to keyboard control mode."<<endl;
            }
            if(userControl==false){
                cout<<"[controller] get frame"<< curr_idx <<endl;
                usleep(50000);
                // 系統无人机控制
                if(curr_idx<100) controller->sendCommand(0.0,-0.1,0.0,0.0,true); //x,y,z,r,limit:前后,左右,上下,旋转,限速
                else if(curr_idx<150) controller->sendCommand(0.1,0.0,0.0,0.0,true);
                else if(curr_idx<250) controller->sendCommand(0.0,0.1,0.0,0.0,true);
                else if(curr_idx<300) controller->sendCommand(0.1,0.0,0.0,0.0,true);
                else if(curr_idx<350) controller->sendCommand(0.0,-0.1,0.0,0,true);
                else if(curr_idx<360) controller->sendCommand(0.0,0.0,-0.1,0.0,true);
                else if(curr_idx<420) controller->sendCommand(-0.1,0.0,0.0,0.0,true);
                else if(curr_idx>=420) break;

            }else{
                cout<<"[system] 按住 [ ` ] 手動結束任務。 ====>>> 結束前記得先按空格降落！！！ <<<===="<<endl;
                if(key =='`') break;
                if(key != -1) {
                    UserControl(key, 0.4, 3.0);//鍵盤控制
                }
            }
        }catch(const std::exception& e){
            std::cout << "Caught exception: " << e.what() << std::endl;
            break;
        }
    }
    usleep(50000);
    SLAM.SaveTrajectoryTUM("./logs/trajs/"+string(argv[1])+".txt");
    SLAM.SaveKeyFrameTrajectoryTUM("./logs/trajs/"+string(argv[1])+"_kf.txt");

    close(sockfd);
    controller->land();
    SLAM.Shutdown();
    std::cout<<"[system] Mission complete!"<<endl;
    return 0;
}

void connection(int& sockfd, const char* server_ip, int server_port,struct sockaddr_in servaddr){
    // 创建套接字
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0){
        perror("[socket] 套接字创建失败");
        exit(EXIT_FAILURE);
    }

    // 设置服务器地址
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(server_port);

    if (inet_pton(AF_INET, server_ip, &servaddr.sin_addr) <= 0){
        perror("[socket] 无效的地址/地址不支持");
        exit(EXIT_FAILURE);
    }

    // 连接服务器
    if (connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0){
        perror("[socket] 连接失败");
        exit(EXIT_FAILURE);
    }

    cout << "[socket] 已连接服务器" << endl;
}

void sendFrame2server(int sockfd,int frame_idx,cv::Mat frame_img){
        // 将图像编码为PNG格式字节流
        vector<uchar> buf;
        cv::imencode(".png", frame_img, buf);
        string image_data(buf.begin(), buf.end());
        uint32_t image_size = htonl(image_data.size());
        uint32_t idx = htonl(frame_idx);
        // 发送图像大小
        if (!sendall(sockfd, (char *)&image_size, 4)){
            cerr << "[socket] 发送图像大小失败" << endl;
        }
        // 发送图像idx
        if (!sendall(sockfd, (char *)&idx, 4)){
            cerr << "[socket] 发送图像idx失败" << endl;
        }
        // 发送图像数据
        if (!sendall(sockfd, image_data.c_str(), image_data.size())){
            cerr << "[socket] 发送图像数据失败" << endl;
        }
        // cout << "[socket] 发送图像 " << frame_idx << "，大小为 " << image_data.size() << endl;
}

void recv_depth(int sockfd, int frame_idx, cv::Mat& imD){
    uint32_t depth_size_net, idx_net;
    if (!recvall(sockfd, (char *)&depth_size_net, 4)){
        cerr << "[socket] 接收深度图大小失败" << endl;
    }
    if (!recvall(sockfd, (char *)&idx_net, 4)){
        cerr << "[socket] 接收idx失败" << endl;
    }

    uint32_t image_size = ntohl(depth_size_net);
    uint32_t idx = ntohl(idx_net);
    // cout<<"[socket] 接收深度圖: 當前frame idx / 接收到的idx: "<<frame_idx<<" / "<<idx<<endl;
    // 接收深度图数据
    std::vector<uchar> image_data(image_size);
    if (!recvImgData(sockfd, image_data, image_size)){
        cerr << "[socket] 接收深度图数据失败" << endl;
    }
    else{
    // cout << "[socket] 收到深度图，大小为 " << image_size << endl;
    imD = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);
    cv::resize(imD, imD, cv::Size(640,480));
    // cv::imwrite("./logs/depth/f_"+ to_string(frame_idx) +".png", imD);
    image_data.clear();
    }
}

bool recvall(int sockfd, char *buf, int len){
    int total = 0;
    int n;
    while (total < len){
        n = recv(sockfd, buf + total, len - total, 0);
        if (n <= 0){
            return false;
        }
        total += n;
    }
    return true;
}

bool recvImgData(int sockfd, std::vector<uchar>& buf, int len){
    int total = 0;
    int n;
    while (total < len){
        n = recv(sockfd, buf.data() + total, len - total, 0);
        if (n <= 0){
            return false;
        }
        total += n;
    }
    return true;
}

bool sendall(int sockfd, const char *buf, int len){
    int total = 0;
    int n;
    while (total < len){
        n = send(sockfd, buf + total, len - total, 0);
        if (n <= 0)
        {
            return false;
        }
        total += n;
    }
    return true;
}

int UserControl(int key, float v, float a){//控制器:移动速度，旋转速度
    //左j，右l，前i，後，上w，下s
    //左前u，左後n，右前o，右後m
    //左上q，左下z，右上e，右下c
    //前上t，前下g，後上f，後下h
    //左轉a，右轉d
    switch(key){
        case 'w':
            cout << "[controller] up---> w" << endl;
            controller->keyboardControl(key, v);
        break;
        case 's':
            cout << "[controller] down---> s" << endl;
            controller->keyboardControl(key, v);
        break;
        case 'j':
            cout << "[controller] left---> j" << endl;
            controller->keyboardControl(key, v);
        break;
        case 'k':
            cout << "[controller] back---> k" << endl;
            controller->keyboardControl(key, v);
        break;
        case 'l':
            cout << "[controller] right---> l" << endl;
            controller->keyboardControl(key, v);
        break;
        case 'i':
            cout << "[controller] forward---> i" << endl;
            controller->keyboardControl(key, v);
        break;
        case 'a':
            cout << "[controller] turn left---> a" << endl;
            controller->keyboardControl(key, a);
        break;
        case 'd':
            cout << "[controller] turn right---> d" << endl;
            controller->keyboardControl(key, a);
        break;
        case 'u':
            cout << "[controller] left forward" << endl;
            controller->sendCommand(v,-v,0,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 'o':
            cout << "[controller] right forward" << endl;
            controller->sendCommand(v,v,0,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 'n':
            cout << "[controller] left backward" << endl;
            controller->sendCommand(-v,-v,0,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 'm':
            cout << "[controller] right backward" << endl;
            controller->sendCommand(-v,v,0,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 'q':
            cout << "[controller] left up" << endl;
            controller->sendCommand(0,-v,v,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 'z':
            cout << "[controller] left down" << endl;
            controller->sendCommand(0,-v,-v,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 'e':
            cout << "[controller] right up" << endl;
            controller->sendCommand(0,v,v,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 'c':
            cout << "[controller] right down" << endl;
            controller->sendCommand(0,v,-v,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 't':
            cout << "[controller] forward up" << endl;
            controller->sendCommand(v,0,v,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 'g':
            cout << "[controller] forward down" << endl;
            controller->sendCommand(v,0,-v,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 'f':
            cout << "[controller] backward up" << endl;
            controller->sendCommand(-v,0,v,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case 'h':
            cout << "[controller] backward down" << endl;
            controller->sendCommand(-v,0,-v,0,true);//x,y,z,r,limit:前后,左右,上下,旋转,限速
        break;
        case ' ':
            cout << "[controller] take off or landing" << endl;
            controller->keyboardControl(key, 0);
            sleep(3);
        break;
    }
}
