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
void recv_command(int sockfd, string& receivedString);
void get_and_set_params(string parameterFile,string tar_name,int inspection_mode,cv::Mat *K,cv::Mat *distCoeffs);
std::vector<float> getDirection(Sophus::SE3f Twc_curr,Sophus::SE3f Twc_video);
float getRotation(const Sophus::SE3f& Twc_curr, const Sophus::SE3f& Twc_video);

// Add function to initialize and check controller connection
bool initializeController() {
    int retry_count = 0;
    const int max_retries = 5;
    
    while (retry_count < max_retries) {
        cout << "[Controller] Attempting to initialize controller connection (attempt " << retry_count + 1 << "/" << max_retries << ")" << endl;
        
        if (controller->initializeConnection()) {
            cout << "[Controller] Successfully connected to Android app" << endl;
            return true;
        }
        
        retry_count++;
        if (retry_count < max_retries) {
            cout << "[Controller] Connection failed, retrying in 2 seconds..." << endl;
            sleep(2);
        }
    }
    
    cerr << "[Error] Failed to establish connection with Android app after " << max_retries << " attempts" << endl;
    return false;
}

int main(int argc, char **argv) {
    if(argc < 7){
        std::cout << "[Usage]: ./client ./parameter/File ./target/map/name ./experiment/name inspectionmode(關閉: 0 / 開啓: 1) onlykadfp(关闭:0 / 开启:1) inputmode(無人機畫面: 0 / 視頻模擬輸入: 1 / RTMP輸入: 2) [./模擬影片path] [RTMP URL] [phone_ip]" << endl;
        return 1;
    }
    parameter_file = string(argv[1]);
    target_name = string(argv[2]);
    ex_name = string(argv[3]);
    inspection_mode = atoi(argv[4]);
    only_kadfp = atoi(argv[5]);
    input_mode = atoi(argv[6]);
    if(input_mode == 1 && argc >= 8) videoFile = string(argv[7]);
    string rtmpUrl = "";
    string phone_ip = "192.168.0.158"; // Default phone IP

    // Handle RTMP URL and phone IP arguments
    if(input_mode == 2) {
        if(argc < 8) {
            std::cerr << "[Error] RTMP mode (mode 2) requires RTMP URL as argument" << std::endl;
            return 1;
        }
        rtmpUrl = string(argv[7]);
        if(rtmpUrl.empty() || rtmpUrl.substr(0, 7) != "rtmp://") {
            std::cerr << "[Error] Invalid RTMP URL format. Must start with 'rtmp://'" << std::endl;
            return 1;
        }
        // Ensure URL has the correct path
        if(rtmpUrl.find("/live/stream") == string::npos) {
            rtmpUrl += "/live/stream";
            std::cout << "[Info] Added default path to RTMP URL: " << rtmpUrl << std::endl;
        }
    }

    // Check for phone IP argument
    if(argc >= 9 && (input_mode == 0 || input_mode == 2)) {
        phone_ip = string(argv[8]);
        std::cout << "[Info] Using custom phone IP: " << phone_ip << std::endl;
    }

    // Initialize controller connection if using drone or RTMP mode
    if (input_mode == 0 || input_mode == 2) {
        controller = new Controller(phone_ip);
        if (!initializeController()) {
            cerr << "[Error] Failed to initialize controller. Please check if the Android app is running and accessible." << endl;
            return 1;
        }
    }

    //設置相機參數檔中，ORBSLAM3地圖存取模式：如果inspection_mode=1，則地圖模式為load，否則地圖模式為save
    cv::Mat K = cv::Mat::zeros(3, 3, CV_32F);
    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_32F);
    get_and_set_params(parameter_file,target_name, inspection_mode, &K, &distCoeffs);

    // // 創建相機內參數矩陣
    // cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // // 創建畸變係數向量
    // cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);

    // 误差容忍度，如果室內可以再調小一點
    float trans_threshold = 0.008;
    float rot_threshold = 1.0;
    float kadfp_threshold = 15.0;
    int compare_times_threshold = 10;
    
    ORB_SLAM3::System SLAM(vocFile, parameter_file, ORB_SLAM3::System::RGBD, true);// 声明 ORB-SLAM3 系统
    // float imageScale = SLAM.GetImageScale();
    
    cv::VideoCapture video(videoFile);// 获取影片幀
    if (input_mode ==1 && !video.isOpened()) {
        std::cerr << "Failed to open the video file." << std::endl;
        return 1;
    }
  
    std::map<std::string, CommandType> fly_command = {
        {"original", CommandType::CUSTOM},
        {"left", CommandType::LEFT},
        {"right", CommandType::RIGHT},
        {"forward", CommandType::FORWARD},
        {"backward", CommandType::BACKWARD},
        {"up", CommandType::UP},
        {"down", CommandType::DOWN},
        {"turnleft", CommandType::TURN_LEFT},
        {"turnright", CommandType::TURN_RIGHT}
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
    
    int kf_pose_idx =0;
    int last_idx = 0;
    auto start = chrono::system_clock::now();
    std::vector<ORB_SLAM3::KeyFrame*> vpKFs={};
    int traj_len = 0;
    int delay_count = 0;

    if(inspection_mode == 1){
        cout << "[DEBUG] Inspection mode: Loading keyframes from atlas..." << endl;
        try {
            vpKFs = SLAM.GetAllKeyFramesFromAtlas();
            if(vpKFs.empty()) {
                cerr << "[Error] No keyframes loaded from atlas" << endl;
                return 1;
            }
            cout << "[DEBUG] Successfully loaded " << vpKFs.size() << " keyframes from atlas" << endl;
            
            sort(vpKFs.begin(),vpKFs.end(),ORB_SLAM3::KeyFrame::lId);
            traj_len = vpKFs.size();
            cout << "[DEBUG] Sorted keyframes, trajectory length: " << traj_len << endl;
        } catch (const std::exception& e) {
            cerr << "[Error] Failed to load keyframes: " << e.what() << endl;
            return 1;
        }
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
            float vx=0.1,vy=0.1,vz=0.1;
            if(input_mode == 0){ // 由終端參數決定輸入來源爲無人機 0 或使用影片模擬輸入 1
                // 载入一张无人机畫面
                cout<<"[Tips] Step1-1:從 無人機 接收一幀畫面。"<<endl;
                im = controller->getImage();
                vx = controller->getvx();
                vy = controller->getvy();
                vz = controller->getvz();
                cout<<"[controller] vx vy vz: "<<vx<<" "<<vy<<" "<<vz<<endl;
                if(im.empty()){
                    std::cout <<"[system] loop["<< counter_loop << "] ====> frame empty." << endl;
                    usleep(50000);
                    continue;
                }
                // 去畸變inspectionmode
                cv::undistort(im, imRGB, K, distCoeffs);
            }
            else if(input_mode==2){ // rtmp input
                try {
                    if(SLAM.GetKey() == 27) { // ESC key
                        std::cout << "[system] Stopping RTMP stream..." << std::endl;
                        controller->stop();  // This will stop the RTMP stream
                        break;
                    }

                    // Only set RTMP URL once at the start
                    static bool rtmpInitialized = false;
                    if (!rtmpInitialized) {
                        std::cout << "[DEBUG] Initializing RTMP with URL: " << rtmpUrl << std::endl;
                        controller->setRTMPUrl(rtmpUrl);
                        rtmpInitialized = true;
                    }

                    // Get frame with minimal delay
                    std::cout << "[DEBUG] Attempting to get frame from RTMP stream..." << std::endl;
                    im = controller->getImage();
                    if(im.empty()) {
                        std::cerr << "[Error] Failed to get image from RTMP stream - retrying..." << std::endl;
                        usleep(100000); // Wait 100ms before retry
                        continue;
                    }
                    
                    std::cout << "[DEBUG] Successfully got RTMP frame - Size: " << im.size() << " Channels: " << im.channels() << std::endl;
                    
                    // Convert frame to BGR if it's not already
                    if(im.channels() != 3) {
                        std::cout << "[DEBUG] Converting frame from " << im.channels() << " channels to BGR" << std::endl;
                        cv::cvtColor(im, im, cv::COLOR_YUV2BGR_I420);
                    }
                    
                    // Resize to expected resolution if needed (ORB-SLAM typically works with 640x480)
                    if(im.size() != cv::Size(640, 480)) {
                        cv::resize(im, im, cv::Size(640, 480));
                    }

                    // Undistort with pre-allocated output
                    std::cout << "[DEBUG] Undistorting frame..." << std::endl;
                    cv::undistort(im, imRGB, K, distCoeffs);
                    if(imRGB.empty()) {
                        std::cerr << "[Error] Undistorted frame is empty - skipping frame" << std::endl;
                        continue;
                    }
                    std::cout << "[DEBUG] Frame undistorted successfully - Size: " << imRGB.size() << std::endl;

                    vx = controller->getvx();
                    vy = controller->getvy();
                    vz = controller->getvz();
                    
                } catch (const std::exception& e) {
                    std::cerr << "[Error] RTMP error: " << e.what() << std::endl;
                    usleep(100000); // Wait 100ms before retry
                    continue;
                }
            }
            else{
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
            curr_idx++;
            auto now = chrono::system_clock::now();
            auto timestamp = chrono::duration_cast<chrono::milliseconds>(now - start);
            double ts = double(timestamp.count())/1000.0;

            // Inside the main loop, before sending frame to server:
            auto serverProcessStart = std::chrono::steady_clock::now();
            
            if(input_mode == 2) {
                // Set server as busy before sending frame
                controller->setRTMPBusy(true);
            }

            // 传输到server端进行单目深度估计
            cout<<"[Tips] Step2-1:發送畫面到 server。"<<endl;
            sendFrame2server(sockfd, curr_idx, imRGB);

            // 接收处理好的深度图
            cv::Mat imD;
            Sophus::SE3f Twc_curr;
            cout<<"[Tips] Step2-2:從 server 接收深度圖。"<<endl;
            recv_depth(sockfd, curr_idx, imD);

            // Calculate and update server processing time
            if(input_mode == 2) {
                auto serverProcessEnd = std::chrono::steady_clock::now();
                double processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                    serverProcessEnd - serverProcessStart).count();
                
                controller->updateRTMPProcessingTime(processingTime);
                controller->setRTMPBusy(false);
                
                if(processingTime > 1000) { // Log if processing takes more than 1 second
                    std::cout << "[Warning] High server processing time: " << processingTime << "ms" << std::endl;
                }
            }

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

                // 如果發生地圖融合，更新地图中的所有关键帧
                if(SLAM.getMergeState()){
                    cout << "[DEBUG] Map merge detected, updating keyframes..." << endl;
                    try {
                        vpKFs = SLAM.GetAllKeyFramesFromMergedMap();
                        if(vpKFs.empty()) {
                            cerr << "[Error] No keyframes in merged map" << endl;
                            continue;
                        }
                        sort(vpKFs.begin(),vpKFs.end(),ORB_SLAM3::KeyFrame::lId);
                        cout << "[DEBUG] Successfully updated keyframes after merge, count: " << vpKFs.size() << endl;
                    } catch (const std::exception& e) {
                        cerr << "[Error] Failed to update keyframes after merge: " << e.what() << endl;
                        continue;
                    }
                }

                cout << "[DEBUG] Accessing keyframe at index: " << kf_pose_idx << " (total: " << vpKFs.size() << ")" << endl;
                if(kf_pose_idx >= vpKFs.size()) {
                    cerr << "[Error] Keyframe index out of bounds" << endl;
                    break;
                }
                
                Sophus::SE3f Twc_video;
                try {
                    Twc_video = vpKFs[kf_pose_idx]->GetPoseInverse(); //获取当前关键帧的位姿
                    int v_idx = vpKFs[kf_pose_idx]->mnFrameId; //获取当前关键帧在所有幀中的id，用於從參考視頻中找到對應的幀
                    cout << "[DEBUG] Successfully got keyframe pose and frame ID: " << v_idx << endl;
                    cout<< "[system] kf_pose_idx(v_idx) / traj_len: " << kf_pose_idx << "(" << v_idx <<") / "<<traj_len<<endl;
                    
                    if((fabs(vx)>0 || fabs(vy)>0 || fabs(vz)>0) && delay_count<5){//如果無人機在運動中，則不進行指令計算
                        delay_count++;
                        cout<<"[system] delay_count: "<<delay_count<<endl;
                        cout<<"[Tips] Step3-1:發送無需計算kadfp信號到 server。"<<endl;
                        uint32_t v_idx_data = htonl(-1);
                        if (!sendall(sockfd, (char *)&v_idx_data, 4)) cerr << "[socket] 发送idx失败" << endl;
                    }else{
                        delay_count = 0;
                        cout<<"[Tips] Step3-1:發送關鍵幀的idx到 server。"<<endl;
                        uint32_t v_idx_data = htonl(v_idx);
                        if (!sendall(sockfd, (char *)&v_idx_data, 4)) cerr << "[socket] 发送idx失败" << endl;
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

                        if(fabs(orbv_curr[0]+orbv_curr[1]+orbv_curr[2]) < trans_threshold && orba_curr < rot_threshold)
                            orb_align = true;
                    
                        if(only_kadfp==1){
                            if(kadfp_error<kadfp_threshold) kadfp_align = true;
                            compare_times = compare_times_threshold * 3;
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
                        cout<<"[system] 比較次數 / 上限:  "<<compare_counter<<"/"<<compare_times<<endl;
                        if(kadfp_align || compare_counter>=compare_times){
                            if(kf_pose_idx<traj_len-1){
                                kf_pose_idx+=1;
                                cout<<"[system] 進入下一個關鍵幀("<< vpKFs[kf_pose_idx]->mnFrameId <<")"<<endl;
                                compare_counter = 0;
                                align_state = true;
                            }else{
                                end_flag = true;
                            }
                        }

                        // 系統无人机控制
                        if(align_state == false && end_flag == false){
                            cout<<"[Tips] Step1-2: 發送控制指令到 無人機。"<<endl;
                            if(orb_align || only_kadfp==1){
                                float kadfp_v = kadfp_error/200;
                                if(kadfp_v>0.3) kadfp_v = 0.3;
                                cout<<"[controller] KADFP command - Direction: "<<kadfp_dir<<" Velocity: "<<kadfp_v<<endl;
                                if(input_mode == 0 || input_mode == 2) {
                                    if (!controller->isConnected()) {
                                        cout << "[Controller] Connection lost, attempting to reconnect..." << endl;
                                        if (!initializeController()) {
                                            cerr << "[Error] Failed to reconnect to Android app" << endl;
                                            continue;
                                        }
                                    }
                                    cout<<"[DEBUG] Sending KADFP command to drone - Direction: "<<kadfp_dir<<" Command type: "<<static_cast<int>(fly_command[kadfp_dir])<<" Velocity: "<<kadfp_v<<endl;
                                    
                                    // Use the new unified command interface
                                    controller->executeCommand(DroneCommand(fly_command[kadfp_dir], kadfp_v));
                                    
                                    if(orba_curr>rot_threshold){
                                        cout<<"[DEBUG] Sending rotation command to drone - Angle: "<<orba_curr<<endl;
                                        usleep(5000);
                                        if(input_mode == 0 || input_mode == 2) {
                                            if (controller->isConnected()) {
                                                // Use the new unified command interface for rotation
                                                if(orba_curr > 0) {
                                                    controller->executeCommand(DroneCommand(CommandType::TURN_RIGHT, orba_curr));
                                                } else {
                                                    controller->executeCommand(DroneCommand(CommandType::TURN_LEFT, -orba_curr));
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    cout<<"[controller] ORB command - X: "<<orbv_curr[0]<<" Y: "<<orbv_curr[1]<<" Z: "<<orbv_curr[2]<<" Rotation: "<<orba_curr<<endl;
                                    if(input_mode == 0 || input_mode == 2) {
                                        if (!controller->isConnected()) {
                                            cout << "[Controller] Connection lost, attempting to reconnect..." << endl;
                                            if (!initializeController()) {
                                                cerr << "[Error] Failed to reconnect to Android app" << endl;
                                                continue;
                                            }
                                        }
                                        cout<<"[DEBUG] Sending ORB command to drone - X: "<<orbv_curr[0]<<" Y: "<<orbv_curr[1]<<" Z: "<<orbv_curr[2]<<" Rotation: "<<orba_curr<<endl;
                                        
                                        // Use the new unified command interface for direct velocity control
                                        controller->executeCommand(DroneCommand(orbv_curr[0], orbv_curr[1], orbv_curr[2], orba_curr, true));
                                    }
                                }
                                usleep(100);
                            }
                        }
                        // 系統无人机控制end
                    }
                } catch (const std::exception& e) {
                    cerr << "[Error] Failed to get keyframe pose: " << e.what() << endl;
                    continue;
                }
                
                if(end_flag == true){
                    cout << "[system] 影片结束，任務完成。 " << std::endl;
                    
                    // Save trajectories before potential exit
                    try {
                        if(inspection_mode == 0){
                            cout << "[system] Saving trajectories to target directory..." << endl;
                            SLAM.SaveTrajectoryTUM("../../../data/target/"+target_name+"/trajs_tar.txt");
                            SLAM.SaveKeyFrameTrajectoryTUM("../../../data/target/"+target_name+"/trajs_tar_kf.txt");
                        }else{
                            cout << "[system] Saving trajectories to outputs directory..." << endl;
                            SLAM.SaveTrajectoryTUM("../../../data/outputs/"+ex_name+"/trajs_res.txt");
                            SLAM.SaveKeyFrameTrajectoryTUM("../../../data/outputs/"+ex_name+"/trajs_res_kf.txt");
                        }
                        cout << "[system] Trajectories saved successfully." << endl;
                    } catch (const std::exception& e) {
                        std::cerr << "[Error] Failed to save trajectories: " << e.what() << std::endl;
                    }

                    cout<<"[system] 按住[ space ]降落,按住[ esc ]退出任務。"<<endl;
                    auto stop = chrono::system_clock::now();
                    auto stoptime = chrono::duration_cast<chrono::milliseconds>(stop - start);
                    double worktime = double(stoptime.count())/1000.0;
                    cout << "[system] total cost: " << worktime << "s" << endl;
                    while(1){
                        int key = -1;
                        key = SLAM.GetKey();
                        if(key ==' ') {
                            if(input_mode == 0 || input_mode == 2) controller->land();
                        }
                        if(key == 27) break;//char 27 = 'esc' 
                        usleep(200000);
                    }
                    break;
                }
            }
        }catch(const std::exception& e){
            std::cout << "Caught exception: " << e.what() << std::endl;
            
            // Save trajectories even if exception occurs
            try {
                if(inspection_mode == 0){
                    cout << "[system] Saving trajectories to target directory..." << endl;
                    SLAM.SaveTrajectoryTUM("../../../data/target/"+target_name+"/trajs_tar.txt");
                    SLAM.SaveKeyFrameTrajectoryTUM("../../../data/target/"+target_name+"/trajs_tar_kf.txt");
                }else{
                    cout << "[system] Saving trajectories to outputs directory..." << endl;
                    SLAM.SaveTrajectoryTUM("../../../data/outputs/"+ex_name+"/trajs_res.txt");
                    SLAM.SaveKeyFrameTrajectoryTUM("../../../data/outputs/"+ex_name+"/trajs_res_kf.txt");
                }
                cout << "[system] Trajectories saved successfully." << endl;
            } catch (const std::exception& e) {
                std::cerr << "[Error] Failed to save trajectories: " << e.what() << std::endl;
            }

            if(input_mode == 0 || input_mode == 2) controller->land();
            break;
        }
    }

    // Remove duplicate trajectory saving here since we now save before any exit points
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

void sendFrame2server(int sockfd, int frame_idx, cv::Mat frame_img) {
    if(frame_img.empty()) {
        std::cerr << "[Error] Cannot send empty frame to server" << std::endl;
        return;
    }

    try {
        // Ensure the image is in BGR format
        cv::Mat bgr_img;
        if(frame_img.channels() == 3) {
            bgr_img = frame_img;
        } else if(frame_img.channels() == 1) {
            cv::cvtColor(frame_img, bgr_img, cv::COLOR_GRAY2BGR);
        } else {
            std::cerr << "[Error] Unsupported image format with " << frame_img.channels() << " channels" << std::endl;
            return;
        }

        // Encode image
        std::vector<uchar> buf;
        if(!cv::imencode(".png", bgr_img, buf)) {
            std::cerr << "[Error] Failed to encode image" << std::endl;
            return;
        }

        string image_data(buf.begin(), buf.end());
        uint32_t image_size = htonl(image_data.size());
        uint32_t idx = htonl(frame_idx);

        // Send image size
        if (!sendall(sockfd, (char *)&image_size, 4)) {
            std::cerr << "[socket] Failed to send image size" << std::endl;
            return;
        }

        // Send image idx
        if (!sendall(sockfd, (char *)&idx, 4)) {
            std::cerr << "[socket] Failed to send image idx" << std::endl;
            return;
        }

        // Send image data
        if (!sendall(sockfd, image_data.c_str(), image_data.size())) {
            std::cerr << "[socket] Failed to send image data" << std::endl;
            return;
        }

        std::cout << "[socket] Successfully sent frame " << frame_idx << " (size: " << image_data.size() << " bytes)" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Error] Exception in sendFrame2server: " << e.what() << std::endl;
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

void get_and_set_params(string parameterFile, string tar_name,int inspection_mode, cv::Mat *K, cv::Mat *distCoeffs){
    string str = "\"../../../data/target/" + tar_name + "/map\"";
    cv::FileStorage fs(parameterFile, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        cerr << "Failed to open parameter file: " << parameterFile << endl;
        exit(EXIT_FAILURE);
    }
    try{
        // 讀取相機內參矩陣
        fs["Camera1.fx"] >> (*K).at<float>(0, 0);
        fs["Camera1.fy"] >> (*K).at<float>(1, 1);
        fs["Camera1.cx"] >> (*K).at<float>(0, 2);
        fs["Camera1.cy"] >> (*K).at<float>(1, 2);
        (*K).at<float>(2, 2) = 1.0;

        // 讀取畸變參數
        fs["Camera1.k1"] >> (*distCoeffs).at<float>(0, 0);
        fs["Camera1.k2"] >> (*distCoeffs).at<float>(0, 1);
        fs["Camera1.p1"] >> (*distCoeffs).at<float>(0, 2);
        fs["Camera1.p2"] >> (*distCoeffs).at<float>(0, 3);
        fs["Camera1.k3"] >> (*distCoeffs).at<float>(0, 4);

        // cout<<"K: "<<*K<<endl;
        // cout<<"distCoeffs: "<<*distCoeffs<<endl;
        fs.release();
    }catch(const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    //根據inspection_mode設置地圖存取模式
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
        fileContent += load_str+ ": \"../../../data/target/" + tar_name + "/map\"";
    }else{
        fileContent += save_str+ ": \"../../../data/target/" + tar_name + "/map\"";
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

    float alpha =3;
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