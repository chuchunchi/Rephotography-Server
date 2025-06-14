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
#include <Eigen/Dense>
#include <cmath>

#include "VideoSocketUDP.h"
#include "Controller.h"
#include "pid.h"

using namespace std;

// 参数文件与字典文件
// 如果你系统上的路径不同，请修改它
string vocFile = "../../Vocabulary/ORBvoc.txt";
string target_name = "";
string ex_name = "";
string parameter_file = "";
// 视频文件
string videoFile = "";
int inspection_mode = 0;
int only_kadfp = 0;
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
void get_and_set_params(string parameterFile,string map_name,int inspection_mode);
std::vector<float> getDirection(Sophus::SE3f Twc_curr,Sophus::SE3f Twc_video);
float quaternion_to_euler_yaw(const Eigen::Quaternionf& q);
float normalize_angle(float angle);
float getRotation(const Sophus::SE3f& Twc_curr, const Sophus::SE3f& Twc_video);


int main(int argc, char **argv) {
    if(argc != 8){
        std::cout << "[Usage]: ./client ./video/path ./map/name ./output/trajectorys/name ./parameter/File inspectionmode(關閉: 0 / 開啓: 1) onlykadfp(关闭:0 / 开启:1) intputmode(無人機畫面: 0 / 視頻模擬輸入: 1)" << endl;
        return 1;
    }
    videoFile = string(argv[1]);
    target_name = string(argv[2]);
    ex_name = string(argv[3]);
    parameter_file = string(argv[4]);
    inspection_mode = atoi(argv[5]);
    only_kadfp = atoi(argv[6]);
    input_mode = atoi(argv[7]);

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

    // 误差容忍度，如果室內可以再調小一點
    float trans_threshold = 0.008;
    float rot_threshold = 1.0;
    float kadfp_threshold = 15.0;
    int compare_times_threshold = 8;
    
    //設置相機參數檔中，ORBSLAM3地圖存取模式：如果inspection_mode=1，則地圖模式為load，否則地圖模式為save
    get_and_set_params(parameter_file,target_name,inspection_mode);
    
    ORB_SLAM3::System SLAM(vocFile, parameter_file, ORB_SLAM3::System::RGBD, true);// 声明 ORB-SLAM3 系统
    float imageScale = SLAM.GetImageScale();
    
    cv::VideoCapture video(videoFile);// 获取视频图像
    if (input_mode ==1 && !video.isOpened()) {
        std::cerr << "Failed to open the video file." << std::endl;
        return 1;
    }
  
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
    const char *server_ip = "127.0.0.1";
    int server_port = 8888;
    int sockfd;
    struct sockaddr_in servaddr;
    connection(sockfd, server_ip, server_port, servaddr);
   
    bool start_flag = false;
    bool end_flag = false;
    int counter_loop = 0;//总循环计数器
    int curr_idx = -1;
    int compare_counter = 0;
    
    int kf_pose_num =0;
    int last_idx = 0;
    auto start = chrono::system_clock::now();
    std::vector<ORB_SLAM3::KeyFrame*> vpKFs={};
    int traj_len = 0;

    if(inspection_mode == 1){
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

            std::cout<< "[system] ================== [total loop]: " << counter_loop <<" =================="<<endl;
            cv::Mat im;
            cv::Mat imRGB;
            if(input_mode == 0){ // 由終端參數決定輸入來源爲無人機 0 或使用影片模擬輸入 1
                // 载入一张无人机畫面
                cout<<"[Tips] Step1-1:從 無人機 接收一幀畫面。"<<endl;
                im = controller->getImage();
                if(im.empty()){
                    std::cout <<"[system] loop["<< counter_loop << "] ====> frame empty." << endl;
                    usleep(50000);
                    continue;
                }
                // 去畸變
                cv::undistort(im, imRGB, K, distCoeffs);
            }else{
                // 模拟输入
                // cout<<"[Tips] Step1-1:讀取一幀模擬影片畫面。"<<endl;
                video>>imRGB;
                if (imRGB.data == nullptr){//说明input影片的所有frame pose定位完成
                    std::cout<<"[system] 影片結束，輸入任意鍵保存並結束任務。 <------------------------------"<<endl;
                    int key = -1;
                    while(key==-1) key = SLAM.GetKey();
                    break;
                }
            }
            counter_loop++;
            auto now = chrono::system_clock::now();
            auto timestamp = chrono::duration_cast<chrono::milliseconds>(now - start);
            double ts = double(timestamp.count())/1000.0;
            curr_idx++;

            // 传输到server端进行单目深度估计
            cout<<"[Tips] Step2-1:發送畫面到 server。"<<endl;
            sendFrame2server(sockfd, curr_idx, imRGB);

            // 接收处理好的深度图
            cv::Mat imD;
            Sophus::SE3f Twc_curr;
            cout<<"[Tips] Step2-2:從 server 接收深度圖。"<<endl;
            recv_depth(sockfd, curr_idx, imD);
            if(!imD.empty()){
                Sophus::SE3f Tcw = SLAM.TrackRGBD(imRGB, imD, ts);
                Twc_curr = Tcw.inverse();
            }

            if(inspection_mode == 0){ // 非巡檢模式（錄製模式）到這裏就結束了
                int key= SLAM.GetKey();
                if(key == 27) break; //char 27 = 'esc'
            }else{
                bool orb_align = false;
                bool kadfp_align = false;
                bool align_state = false;
                int compare_times = 0;

                // 获取地图中的所有关键帧
                if(SLAM.getMergeState()){
                    vpKFs = SLAM.GetAllKeyFramesFromMergedMap();
                    sort(vpKFs.begin(),vpKFs.end(),ORB_SLAM3::KeyFrame::lId);
                    traj_len = vpKFs.size();
                }
                cout<<"[system] kf_pose_num / traj_len: "<<kf_pose_num<<" / "<<traj_len<<endl;
                Sophus::SE3f Twc_video=vpKFs[kf_pose_num]->GetPoseInverse();
                int v_idx = vpKFs[kf_pose_num]->mnFrameId;
                // 发送此pose的idx
                cout<<"[Tips] Step3-1:發送關鍵幀的idx到 server。"<<endl;
                uint32_t v_idx_data = htonl(v_idx);
                if (!sendall(sockfd, (char *)&v_idx_data, 4)){
                    cerr << "[socket] 发送idx失败" << endl;
                }

                // 处理KADFP指令
                cout<<"[Tips] Step3-2:從 server 接收KADFP計算的指令。"<<endl;
                std::string receivedString;
                recv_command(sockfd, receivedString);
                std::istringstream iss(receivedString);
                std::string kadfp_dir;
                float kadfp_error = 0.0;
                
                iss >> kadfp_dir >> kadfp_error;
                compare_counter++;

                // 處理orb位姿判斷移動和旋轉
                std::vector<float> orbv_curr(3,0.0);
                orbv_curr = getDirection(Twc_curr, Twc_video); 
                float orba_curr = 0.0;
                if(counter_loop>20 || SLAM.getMergeState()) orba_curr = getRotation(Twc_curr, Twc_video);

                if(fabs(orbv_curr[0]+orbv_curr[1]+orbv_curr[2]) < trans_threshold && orba_curr <rot_threshold)
                    orb_align = true;
            
                if(only_kadfp==1){
                    if(kadfp_error<kadfp_threshold-5) kadfp_align = true;
                    compare_times = 20;
                }else{
                    if(kadfp_error<kadfp_threshold) kadfp_align = true;
                    if (v_idx != last_idx) {
                        compare_times = v_idx - last_idx +1 + compare_times_threshold;
                        last_idx = v_idx;
                    }
                    else{
                        compare_times = compare_times_threshold;
                    }
                }
                cout<<"[system] compare / total:  "<<compare_counter<<"/"<<compare_times<<endl;
                cout<<"[system] curr frame / video frame: "<< curr_idx <<" / "<< v_idx <<endl;
                if(kadfp_align || compare_counter>=compare_times){
                    if(kf_pose_num<traj_len-1){
                        kf_pose_num+=1;
                        cout<<"[system] 進入下一個關鍵幀("<< vpKFs[kf_pose_num]->mnFrameId <<")"<<endl;
                        compare_counter = 0;
                        align_state = true;
                    }else{
                        end_flag = true;
                    }
                }

                // 系統无人机控制
                if(align_state == false && end_flag == false){
                    // cout<<"[Tips] Step1-2: 發送控制指令到 無人機。"<<endl;
                    if(orb_align || only_kadfp==1){
                        float kadfp_v = kadfp_error/300;
                        if(kadfp_v>0.2) kadfp_v = 0.2;
                        cout<<"[controller] KADFP:"<<kadfp_dir<<" "<<kadfp_v<<endl;
                        if(input_mode == 0) UserControl(fly_command[kadfp_dir], kadfp_v, 0);
                    }else{
                        cout<<"[controller] ORB:  "<<orbv_curr[0]<<" "<<orbv_curr[1]<<" "<<orbv_curr[2]<<" "<<orba_curr<<endl;
                        if(input_mode == 0) controller->sendCommand(orbv_curr[0],orbv_curr[1],orbv_curr[2],orba_curr,true);
                    }
                    // //設置動態等待時間，確保無人機命令執行完成再進入下一步
                    // int dynamic_wait = ((int)fabs(orbv_curr[0]+orbv_curr[1]+orbv_curr[2]))*20000;
                    // usleep(dynamic_wait > 20000 ? 20000 : dynamic_wait);
                    if(input_mode == 0){
                        int delay_count = 0;
                        while(1){
                            delay_count++;
                            cv::Mat tmp_im = controller->getImage();
                            float vx = controller->getvx();
                            float vy = controller->getvy();
                            float vz = controller->getvz();
                            // cout<<"[controller] vx vy vz: "<<vx<<" "<<vy<<" "<<vz<<endl;
                            if((vx<0.05 && vy<0.05 && vz<0.05) ||delay_count>10) break;
                            else usleep(200);
                        }
                    }
                }
                // 系統无人机控制end
                
                if(end_flag == true){
                    cout << "[system] 影片结束，任務完成。 " << std::endl;
                    cout<<"[system] 按住[ space ]降落,按住[ esc ]退出任務。"<<endl;
                    auto stop = chrono::system_clock::now();
                    auto stoptime = chrono::duration_cast<chrono::milliseconds>(stop - start);
                    double worktime = double(stoptime.count())/1000.0;
                    cout << "[system] total cost: " << worktime << "s" << endl;
                    while(1){
                        int key = -1;
                        key = SLAM.GetKey();
                        if(key ==' ') {
                            if(input_mode == 0) controller->land();
                            usleep(100000);
                        }
                        if(key == 27) break;//char 27 = 'esc' 
                    }
                    break;
                }
            }

        }catch(const std::exception& e){
            std::cout << "Caught exception: " << e.what() << std::endl;
            if(input_mode == 0) controller->land();
            break;
        }
    }
    SLAM.SaveTrajectoryTUM("./logs/trajs/"+ex_name+".txt");
    SLAM.SaveKeyFrameTrajectoryTUM("./logs/trajs/"+ex_name+"_kf.txt");

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
    // uint32_t idx = ntohl(idx_net);
    // 接收深度图数据
    std::vector<uchar> image_data(image_size);
    if (!recvImgData(sockfd, image_data, image_size)){
        cerr << "[socket] 接收深度图数据失败" << endl;
    }
    else{
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
    uint32_t  string_size_net;
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
    cout << "[socket] KADFP: " << string_data << endl;
    receivedString=string_data;
    delete[] string_data;
}

void get_and_set_params(string parameterFile, string map_name,int inspection_mode){
    std::ifstream file(parameterFile);
    std::string lastLine;
    std::string line;
    std::string save_str = "System.SaveAtlasToFile";
    std::string load_str = "System.LoadAtlasFromFile";
    // 读取文件内容
    std::ifstream inFile(parameterFile);
    std::stringstream buffer;
    buffer << inFile.rdbuf();
    std::string fileContent = buffer.str();
    inFile.close();

    // 查找并删除包含指定字符串的行
    size_t pos;
    while ((pos = fileContent.find(load_str)) != std::string::npos || (pos = fileContent.find(save_str)) != std::string::npos) {
        size_t line_start = fileContent.rfind('\n', pos);
        size_t line_end = fileContent.find('\n', pos);
        if (line_start == std::string::npos) line_start = 0;
        else line_start += 1;
        if (line_end == std::string::npos) line_end = fileContent.length();
        fileContent.erase(line_start, line_end - line_start + 1);
    }

    // 写入新内容
    std::ofstream outFile(parameterFile);
    if(inspection_mode == 1){
        fileContent += load_str+ ": \"./logs/maps/" + map_name + "\"";
    }else{
        fileContent += save_str+ ": \"./logs/maps/" + map_name + "\"";
    }
    outFile << fileContent;
    outFile.close();
}

//计算orb-slam位姿误差,判断位移方向
std::vector<float> getDirection(Sophus::SE3f Twc_curr, Sophus::SE3f Twc_video) {
    Eigen::Vector3f t_c = Twc_curr.translation();
    Eigen::Vector3f t_v = Twc_video.translation();
    // cout<<"[system] twc_curr: "<<t_c.z()<<" "<<t_c.x()<<" "<<t_c.y()<<endl;
    // cout<<"[system] twc_vide: "<<t_v.z()<<" "<<t_v.x()<<" "<<t_v.y()<<endl;

    float alpha =1.5;
    std::vector<float> delta = {t_v.x() - t_c.x(), -(t_v.y() - t_c.y()), t_v.z() - t_c.z()};
    // cout<<"[system] delta_origin: "<<delta[2]<<" "<<delta[0]<<" "<<delta[1]<<endl;
    float dis = sqrt(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2])/alpha;
    int axis = std::distance(delta.begin(), std::max_element(delta.begin(), delta.end(), [](float a, float b) { return std::abs(a) < std::abs(b); }));
    
    std::vector<std::string> directions = {"left", "right", "down", "up", "backward", "forward"};
    string bestdir = delta[axis] < 0 ? directions[axis * 2] : directions[axis * 2 + 1];
    cout<<"[system] ORB方向: "<<bestdir<< ", dis: "<<dis<<endl;

    for(int i=0;i<3;i++){
        if(i!=axis) delta[i]=0.0;
        else delta[i] = delta[i] < 0 ? (-dis) : dis;
    }

    std::vector<float> move = {delta[2],delta[0],delta[1]};
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

// float getRotation(const Sophus::SE3f& Twc_curr, const Sophus::SE3f& Twc_video) {
//     Eigen::Quaternionf R_curr = Twc_curr.unit_quaternion();
//     Eigen::Quaternionf R_video = Twc_video.unit_quaternion();
//     float tx = 3;
    
//     Eigen::Quaternionf ang_rel = R_curr.conjugate() * R_video;
//     // new method
//     Eigen::AngleAxisf angleAxis(ang_rel);
//     Eigen::Vector3f rotation_axis = angleAxis.axis();
//     float R_theta = angleAxis.angle();
//     float yaw = rotation_axis.z();
//     float angle = R_theta * 180.0 / M_PI;// 转换为度数
//     if(yaw>0) angle = -angle/tx;
//     else angle = angle/tx;
//     //new method end

//     cout<<"[system] angle_origin: "<<angle<<endl;
//     if(fabs(angle) < 1.0) angle = 0.0;//旋轉角度閾值，低於該值則認為無需旋轉
//     if(angle > 5.0 && angle<180.0) angle = 5.0; //控制旋轉角度在5度以內
//     else if(angle < -5.0 && angle>-180.0) angle = -5.0;
//     else if(fabs(angle)>180.0) angle = 0.0;// 若超過180度則認定爲異常值，無效
    
//     // if(angle>0.0) cout<<"[system] rotation: turnright "<< angle <<endl;
//     // else if(angle<0.0) cout<<"[system] rotation: turnleft "<< angle<<endl;
//     // else cout<<"[system] rotation: original "<< angle<<endl;
    
//     return angle;
// }

// 計算水平角度差（以相機a的朝向為正前方）
// float getRotation(const Sophus::SE3f& Twc_curr, const Sophus::SE3f& Twc_video) {
//     // 提取位姿的平移部分
//     Eigen::Vector3f t_a = Twc_curr.translation();
//     Eigen::Vector3f t_b = Twc_video.translation();

//     // 計算相對位置向量，並投影到 xy 平面
//     Eigen::Vector3f t_ab = t_b - t_a;
//     Eigen::Vector3f t_ab_xy(t_ab.x(), t_ab.y(), 0);

//     // 確保平移向量不為零
//     if (t_ab_xy.norm() < 1e-4) {
//         std::cerr << "Error: Translation vector too small!" << std::endl;
//         return 0.0;
//     }

//     // 定義相機 a 的正前方方向（z 軸方向）
//     Eigen::Vector3f forward_a = Twc_curr.rotationMatrix() * Eigen::Vector3f(0, 0, 1);
//     Eigen::Vector3f forward_a_xy(forward_a.x(), forward_a.y(), 0);

//     // 歸一化向量
//     t_ab_xy.normalize();
//     forward_a_xy.normalize();

//     // 計算角度（點積公式）
//     float cos_theta = forward_a_xy.dot(t_ab_xy);
//     cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
//     float theta = std::acos(cos_theta);
//     float angle = theta * 2;
//     // 計算叉積的 z 分量來判斷方向
//     float cross_z = forward_a_xy.x() * t_ab_xy.y() - forward_a_xy.y() * t_ab_xy.x();
//     if (cross_z < 0) {
//         cout<<"[system] rotation: turnright "<< angle <<endl;
//     }else{
//         angle = -angle;
//         cout<<"[system] rotation: turnleft "<< angle <<endl;
//     }

//     return angle; // 返回弧度制角度
// }
float getRotation(const Sophus::SE3f& Twc_curr, const Sophus::SE3f& Twc_video) {
    // 提取旋轉矩陣
    Eigen::Matrix3f R_curr = Twc_curr.rotationMatrix();
    Eigen::Matrix3f R_video = Twc_video.rotationMatrix();

    // 計算相對旋轉矩陣 R_rel = R_video * R_curr.transpose()
    Eigen::Matrix3f R_rel = R_video * R_curr.transpose();

    // 提取水平旋轉角度 (繞Y軸的旋轉)
    float theta = std::atan2(-R_rel(2, 0), R_rel(0, 0));

    // 將角度從弧度轉換為度
    float angle = theta * 180.0f / M_PI / 3;
    
    if(angle > 5.0 && angle<90.0) angle = 5.0; //控制旋轉角度在5度以內
    else if(angle < -5.0 && angle>-90.0) angle = -5.0;
    else if(fabs(angle)>90.0) angle = 0.0;// 若超過180度則認定爲異常值，無效
    if(angle>0) cout<<"[system] ORB旋轉: turnright "<< angle <<endl;
    else if(angle<0) cout<<"[system] ORB旋轉: turnleft "<< angle<<endl;
    // else cout<<"[system] rotation: original "<< angle<<endl;
    return angle;
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