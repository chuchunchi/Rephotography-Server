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
#include <algorithm>
#include <cmath>

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

void UserControl(int key, float v, float a);
void sendFrame2server(int sockfd, int frame_idx, cv::Mat frame_img);
void recv_depth(int sockfd,int frame_idx,cv::Mat& imD);
bool recvall(int sockfd, char *buf, int len);
bool sendall(int sockfd, const char *buf, int len);
bool recvImgData(int sockfd, std::vector<uchar>& buf, int len);
bool recv_float(int sockfd, float &value);
void recv_command(int sockfd, string& receivedString);
std::vector<float> getDirection(Sophus::SE3f Twc_curr,Sophus::SE3f Twc_video,float threshold);
float quaternion_to_euler_yaw(const Eigen::Quaternionf& q);
float normalize_angle(float angle);
float getRotation(const Sophus::SE3f& Twc_curr, const Sophus::SE3f& Twc_video);
std::string compareDirection(string dir1, string dir2,bool merge_state,bool align_state);



int main(int argc, char **argv) {
    if(argc != 6){
        std::cout << "[Usage]: ./client ./video/path ./trajectorys/file/name ./parameter/File KADFP_mode(關閉: 0 / 開啓: 1) intputmode(無人機畫面: 0 / 視頻模擬輸入: 1)" << endl;
        return 1;
    }
    videoFile = string(argv[1]);
    TrajFileName = TrajFileName + string(argv[2])+".txt";
    parameter_file = string(argv[3]);
    KADFP_mode = atoi(argv[4]);
    input_mode = atoi(argv[5]);

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
    
    cv::VideoCapture video(videoFile);// 获取视频图像
  
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

    string orb_dir = "original";
    //socket初始化
    const char *server_ip = "140.113.195.240";
    int server_port = 8888;
    int sockfd;
    struct sockaddr_in servaddr;
    connection(sockfd, server_ip, server_port, servaddr);
   
    bool start_flag = false;
    bool align_state = false;
    int counter_loop = 0;//总循环计数器
    int curr_idx = -1;
    int compare_counter = 0;
    bool userControl=false;
    std::vector<float> orbv_curr(3,0.0);
    float orba_curr = 0.0;
    int kf_pose_num =0;
    auto start = chrono::system_clock::now();
    std::vector<ORB_SLAM3::KeyFrame*> vpKFs={};
    int traj_len = 0;
    if(KADFP_mode == 1){
        vpKFs = SLAM.GetAllKeyFramesFromAtlas();
        sort(vpKFs.begin(),vpKFs.end(),ORB_SLAM3::KeyFrame::lId);
        traj_len = vpKFs.size();
    }

    //主循环
    while (1) {
        try{
            while(start_flag == false){
                if(SLAM.GetKey()==' '){
                    start_flag =true;
                    cout<<"[system] Mission start."<<endl;
                    start = chrono::system_clock::now(); // 记录系统时间
                } 
            }

            std::cout<< "[system] ================== [total loop]: " << counter_loop++ <<" =================="<<endl;
            cv::Mat im;
            cv::Mat imRGB;
            if(input_mode == 0){ // 由終端參數決定輸入來源爲無人機 0 或使用影片模擬輸入 1
                // 载入一张无人机畫面
                for(int i=0;i<12;i++)
                    im = controller->getImage();
                if(im.empty()){
                    std::cout <<"[system] loop["<< counter_loop << "] ====> frame empty." << endl;
                    continue;
                }
                // 去畸變
                cv::undistort(im, imRGB, K, distCoeffs);
            }else{
                // 模拟输入
                video>>imRGB; 
                if (imRGB.data == nullptr){//说明input影片的所有frame pose定位完成
                    std::cout<<"[system] 影片結束，輸入任意鍵保存並結束任務。 <------------------------------"<<endl;
                    int key = -1;
                    while(key==-1) key = SLAM.GetKey();
                    break;
                }
            }
            auto now = chrono::system_clock::now();
            auto timestamp = chrono::duration_cast<chrono::milliseconds>(now - start);
            double ts = double(timestamp.count())/1000.0;
            curr_idx++;

            // 传输到server端进行单目深度估计
            sendFrame2server(sockfd, curr_idx, imRGB);

            // 接收处理好的深度图
            cv::Mat imD;
            Sophus::SE3f Twc_curr;
            recv_depth(sockfd, curr_idx, imD);
            if(!imD.empty()){
                Sophus::SE3f Tcw = SLAM.TrackRGBD(imRGB, imD, ts);
                Twc_curr = Tcw.inverse();
            }

            if(KADFP_mode == 1){
                // 获取地图中的所有关键帧
                if(SLAM.getMergeState()){
                    cout<<"[system] merge_state = 1"<<endl;
                    vpKFs = SLAM.GetAllKeyFramesFromMergedMap();
                    sort(vpKFs.begin(),vpKFs.end(),ORB_SLAM3::KeyFrame::lId);
                    
                }else{
                    cout<<"[system] merge_state = 0"<<endl;
                }
                
                cout<<"[system] kf_pose_num / traj_len: "<<kf_pose_num<<" / "<<traj_len<<endl;
                Sophus::SE3f Twc_video=vpKFs[kf_pose_num]->GetPoseInverse();
                int v_idx = vpKFs[kf_pose_num]->mnFrameId;
                
                // 发送此pose的idx
                uint32_t v_idx_data = htonl(v_idx);
                if (!sendall(sockfd, (char *)&v_idx_data, 4)){
                    cerr << "[socket] 发送idx失败" << endl;
                }
                std::string receivedString;
                recv_command(sockfd, receivedString);
                // 处理KADFP指令
                std::istringstream iss(receivedString);
                std::string kadfp_dir;
                float kadfp_error = 0.0;
                iss >> kadfp_dir >> kadfp_error;
                // if(!recv_float(sockfd, kadfp_error)){ // 接收kadfp error
                //     cerr << "[socket] 接收KADFP error失败" << endl;
                // // }
                // cout<<"[socket] kadfp_error: "<<kadfp_error<<endl;
                compare_counter++;

                float trans_threshold = 0.02;
                orbv_curr = getDirection(Twc_curr, Twc_video, trans_threshold); // 計算orb相機位姿得到移動方向
                orba_curr = 0.0;
                if(curr_idx>100) orba_curr = getRotation(Twc_curr, Twc_video);// orb位姿判斷旋轉角度
                if(kadfp_error<15 ||(fabs(orbv_curr[0]+orbv_curr[1]+orbv_curr[2])<trans_threshold && orba_curr==0.0))
                    align_state = true;
                else
                    align_state = false;
                
                if(align_state || compare_counter>=8){
                    cout<<"[system] 比较次数: "<<compare_counter<<", align_state = "<<align_state<<" , 进入下一个关键帧(curr:"<< v_idx <<")"<<endl;
                    compare_counter = 0;
                    kf_pose_num++;
                }

                if(kf_pose_num>=traj_len){
                    cout << "[system] 影片结束，任務完成，交由使用者手動控制。 " << std::endl;
                    cout<<"[system] 按住 [ ` ] 手動結束任務。"<<endl;
                    auto stop = chrono::system_clock::now();
                    auto stoptime = chrono::duration_cast<chrono::milliseconds>(stop - start);
                    double worktime = double(stoptime.count())/1000.0;
                    int hours = int(worktime) / 3600;
                    int minutes = (int(worktime) % 3600) / 60;
                    int seconds = int(worktime) % 60;
                    cout << "[system] total cost: " << hours << " : " << minutes << " : " << seconds << "" << endl;
                    while(1){
                        int key = -1;
                        key = SLAM.GetKey();
                        if(key =='`') {
                            if(input_mode == 0) controller->land();
                            break;
                        }
                        if(key != -1) {
                            if(input_mode == 0) UserControl(key, 0.4, 3.0);//鍵盤控制
                        }
                    }
                    break;
                }

                int key = -1;
                key = SLAM.GetKey();
                if(key == 27){ //char 27 = 'esc' 緊急情況按住esc及時退出系統控制改用鍵盤控制，防止炸機
                    userControl = true;
                    std::cout<<"[system] Switch to keyboard control mode."<<endl;
                }
                if(userControl==false){
                    cout<<"[controller] curr frame / video frame: "<< curr_idx <<" / "<< v_idx <<endl;
                    // 使用遙控器手控扫描环境，等待重定位和子地图融合
                    // if(!merge_state){
                    //         cout<<"[system] wait Map Merge................."<<endl;
                    // }else{
                        if(!align_state){
                            cout<<"[controller] "<<orbv_curr[0]<<" "<<orbv_curr[1]<<" "<<orbv_curr[2]<<" "<<orba_curr<<endl;
                            if(counter_loop%5==0 && orba_curr==0.0){
                                if(kadfp_error/300>0.1) kadfp_error = 0.1;
                                if(input_mode == 0) UserControl(fly_command[kadfp_dir], kadfp_error/500, 0);
                            }else{
                                // for(int i=0;i<int(orbv_curr[3]);i++){
                                //     usleep(45000);
                                    // if(input_mode == 0) controller->sendCommand(orbv_curr[0],orbv_curr[1],orbv_curr[2],orba_curr/orbv_curr[3],true);
                                // }
                                if(input_mode == 0) controller->sendCommand(orbv_curr[0],orbv_curr[1],orbv_curr[2],orba_curr,true);
                            }
                        }
                        // 系統无人机控制end
                    // }
                }else{
                    cout<<"[system] 按住 [ ` ] 手動結束任務。"<<endl;
                    if(key =='`') {
                        if(input_mode == 0) controller->land();
                        break;
                    }
                    if(key != -1) {
                        if(input_mode == 0) UserControl(key, 0.4, 3.0);//鍵盤控制
                    }
                }
            }else{
                int key= SLAM.GetKey();
                if(key == 27 or key =='`'){ //char 27 = 'esc'
                    break;
                }

            }
        }catch(const std::exception& e){
            std::cout << "Caught exception: " << e.what() << std::endl;
            if(input_mode == 0) controller->land();
            break;
        }
    }
    if(KADFP_mode==1){
        SLAM.SaveTrajectoryTUM("./logs/trajs/"+string(argv[2])+"_new.txt");
        SLAM.SaveKeyFrameTrajectoryTUM("./logs/trajs/"+string(argv[2])+"_new_kf.txt");
    }else{
        SLAM.SaveTrajectoryTUM("./logs/trajs/"+string(argv[2])+".txt");
        SLAM.SaveKeyFrameTrajectoryTUM("./logs/trajs/"+string(argv[2])+"_kf.txt");
    }
    close(sockfd);
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

bool recv_float(int sockfd, float &value) {
    uint32_t net_value;
    char buffer[sizeof(uint32_t)];
    int total = 0;
    int bytes_left = sizeof(uint32_t);
    int n;

    while (total < sizeof(uint32_t)) {
        n = recv(sockfd, buffer + total, bytes_left, 0);
        if (n <= 0) {
            return false; // 接收失敗
        }
        total += n;
        bytes_left -= n;
    }

    memcpy(&net_value, buffer, sizeof(uint32_t));
    net_value = ntohl(net_value);
    value = *reinterpret_cast<float*>(&net_value);
    return true; // 接收成功
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

void recv_command(int sockfd, string& receivedString){
    uint32_t  string_size_net, idx_net;
    if (!recvall(sockfd, (char *)&string_size_net, 4)){
        cerr << "[socket] 接收字符串大小失败" << endl;
    }
    uint32_t string_size = ntohl(string_size_net);
    // 接收字符串数据
    char *string_data = new char[string_size + 1]; // +1用于字符串结束符
    if (!recvall(sockfd, string_data, string_size)){
        cerr << "[socket] 接收字符串数据失败" << endl;
    }
    string_data[string_size] = '\0'; // 字符串结束符
    cout << "[socket] 收到字符串: " << string_data << endl;
    receivedString=string_data;
    delete[] string_data;
}

//计算orb-slam位姿误差,判断位移方向
std::vector<float> getDirection(Sophus::SE3f Twc_curr, Sophus::SE3f Twc_video, float threshold) {
    Eigen::Vector3f t_c = Twc_curr.translation();
    Eigen::Vector3f t_v = Twc_video.translation();
    cout<<"[system] twc_curr: "<<t_c.z()<<" "<<t_c.x()<<" "<<t_c.y()<<endl;
    cout<<"[system] twc_vide: "<<t_v.z()<<" "<<t_v.x()<<" "<<t_v.y()<<endl;

    float alpha =3.0;
    std::vector<float> delta = {t_v.x() - t_c.x(), -(t_v.y() - t_c.y()), t_v.z() - t_c.z()};
    cout<<"[system] delta_origin: "<<delta[2]<<" "<<delta[0]<<" "<<delta[1]<<endl;
    
    std::vector<std::string> directions = {"left", "right", "up", "down", "backward", "forward", "original"};
    int axis = std::distance(delta.begin(), std::max_element(delta.begin(), delta.end(), [](float a, float b) { return std::abs(a) < std::abs(b); }));
    string bestdir = delta[axis] < 0 ? directions[axis * 2] : directions[axis * 2 + 1];
    float dis = sqrt(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2])/alpha;
    if(dis<threshold) {
        dis = 0.0;
        axis = 6;
    }
    cout<<"[system] bestDirection: "<<bestdir<< ", dis: "<<delta[axis]<<endl;
    int move_times = 1;
    // if(fabs(dis)<0.1){
        for(int i=0;i<3;i++){
            if(i!=axis) delta[i]=0.0;
            else delta[i] = delta[i] < 0 ? (-dis) : dis;
        }
    // }else{
    //     move_times = int(dis/0.1);
    //     for(int i=0;i<3;i++){
    //         if(i!=axis) delta[i]=0.0;
    //         else delta[i] = delta[i] < 0 ? (-0.1) : 0.1;
    //     }
    // }
    
    std::vector<float> move = {delta[2],delta[0],delta[1],float(move_times)};
    return move;
}

float normalize_angle(float angle) {
    if (angle > 180) {
        angle -= 360;
    } else if (angle < -180) {
        angle += 360;
    }
    return angle;
}

float quaternion_to_euler_yaw(const Eigen::Quaternionf& q) {
    float x = q.x(), y = q.y(), z = q.z(), w = q.w();
    float t3 = +2.0 * (w * z + x * y);
    float t4 = +1.0 - 2.0 * (y * y + z * z);
    float yaw_z = std::atan2(t3, t4);

    return yaw_z;
}

float getRotation(const Sophus::SE3f& Twc_curr, const Sophus::SE3f& Twc_video) {
    Eigen::Quaternionf R_curr = Twc_curr.unit_quaternion();
    Eigen::Quaternionf R_video = Twc_video.unit_quaternion();
    float tx = 3;
    
    Eigen::Quaternionf ang_rel = R_curr.conjugate() * R_video;
    float yaw = quaternion_to_euler_yaw(ang_rel);
    float R_theta = 2 * acos(std::abs(ang_rel.w()));//弧度
    float ang_origin = R_theta * 180.0 / M_PI;// 转换为度数
    float angle = normalize_angle(ang_origin)/tx;
    if(yaw>0) angle = -angle;
    cout<<"[system] angle_origin: "<<angle<<endl;
    if(fabs(angle) < 1.0) angle = 0.0;//旋轉角度閾值，低於該值則認為無需旋轉
    if(angle > 3.0 && angle<8.0) angle = 3.0; //控制旋轉角度在3度以內
    else if(angle < -3.0 && angle>-8.0) angle = -3.0;
    else if(fabs(angle)>8.0) angle = 0.0;// 若超過8度則認定爲異常值，無效
    
    if(angle>0.0) cout<<"[system] rotation: turnright "<< angle <<endl;
    else if(angle<0.0) cout<<"[system] rotation: turnleft "<< angle<<endl;
    else cout<<"[system] rotation: original "<< angle<<endl;
    
    return angle;
}


// float getRotation(const Sophus::SE3f& Twc_curr, const Sophus::SE3f& Twc_video) {
//     Eigen::Quaternionf R_curr = Twc_curr.unit_quaternion();
//     Eigen::Quaternionf R_video = Twc_video.unit_quaternion();
//     float tx = 2;

//     float yaw = quaternion_to_euler_yaw(R_curr) - quaternion_to_euler_yaw(R_video);
    
//     Eigen::Quaternionf ang_rel = R_curr.conjugate() * R_video;
//     float R_theta = 2 * acos(std::abs(ang_rel.w()));//弧度
//     float ang_origin = R_theta * 180.0 / M_PI;// 转换为度数
//     float angle = normalize_angle(ang_origin)/tx;
//     if(yaw<0) angle = -angle;
//     cout<<"[system] angle_origin: "<<angle<<endl;
//     if(fabs(angle) < 1.0) angle = 0.0;//旋轉角度閾值，低於該值則認為無需旋轉
//     if(angle > 3.0 && angle<8.0) angle = 3.0; //控制旋轉角度在3度以內
//     else if(angle < -3.0 && angle>-8.0) angle = -3.0;
//     else if(fabs(angle)>8.0) angle = 0.0;// 若超過8度則認定爲異常值，無效
    
//     if(angle>0.0) cout<<"[system] rotation: turnright "<< angle <<endl;
//     else if(angle<0.0) cout<<"[system] rotation: turnleft "<< angle<<endl;
//     else cout<<"[system] rotation: original "<< angle<<endl;
    
//     return angle;
// }

std::string compareDirection(string dir1,string dir2, bool merge_state,bool align_state){
    /*結合orb和kadfp判斷的方向，選擇最終移動方向
    若重定位和地圖融合失敗，則直接使用kadfp方向
    若orb計算結果與kadfp結果一致，直接確定方向;
    若二者結果相反，則以orb方向優先;
    若結果不同，則組合二者方向
    若一方認爲已到達正確位置，則直接返回original*/ 
    if(!merge_state && !align_state){
        return dir1;
    }else
        return dir2;
    // std::vector<string> dir = {dir1,dir2};
    // std::vector<int> dir_vector = {0,0,0};
    // std::map<vector<int>,string> dir_set = {{{0,0,0},"original"},//不動
    //                         {{-1,0,0},"left"},{{1,0,0},"right"},{{0,1,0},"forward"},{{0,-1,0},"backward"},{{0,0,1},"up"},{{0,0,-1},"down"},//6向基礎移動
    //                         {{-1,1,0},"lf"},{{-1,-1,0},"lb"},{{1,1,0},"rf"},{{1,-1,0},"rb"},//組合移動：左前，左後，右前，右後
    //                         {{-1,0,1},"lu"},{{-1,0,-1},"ld"},{{1,0,1},"ru"},{{1,0,-1},"rd"},//左上，左下，右上，右下
    //                         {{0,1,1},"fu"},{{0,1,-1},"fd"},{{0,-1,1},"bu"},{{0,-1,-1},"bd"}//前上，前下，後上，後下
    // };
    // for(string i : dir){
    //     if(i =="original"){
    //         std::fill(dir_vector.begin(), dir_vector.end(), 0);
    //         break;
    //     }
    //     else if(i == "left")
    //         dir_vector[0]=-1;
    //     else if(i == "right")
    //         dir_vector[0]=1;
    //     else if(i == "forward")
    //         dir_vector[1]=1;
    //     else if(i == "backward")
    //         dir_vector[1]=-1;
    //     else if(i == "up")
    //         dir_vector[2]=1;
    //     else if(i == "down")
    //         dir_vector[2]=-1;
    // }
    // return dir_set[dir_vector];
}


void UserControl(int key, float v, float a){//控制器:移动速度，旋转速度
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