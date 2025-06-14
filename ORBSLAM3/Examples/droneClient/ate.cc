#include<pangolin/pangolin.h>
#include <iostream>
#include <sophus/se3.hpp>
#include <fstream>
#include <unistd.h>
#include <Eigen/Core>
#include <Eigen/Dense>
 
using namespace std;
using namespace Sophus;
using namespace Eigen;
 
string groundtruth_file = "/home/liao/workspace/slambook2/ch4/example/groundtruth.txt";
string estimated_file = "/home/liao/workspace/slambook2/ch4/example/estimated.txt";
 
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
TrajectoryType ReadTrajectory(const string& file_name);
void DrawTrajectory(const TrajectoryType& gt, const TrajectoryType& esti);
vector<double> calculateATE(const TrajectoryType& gt, const TrajectoryType& esti);
vector<double> calculateRPE(const TrajectoryType& gt, const TrajectoryType& esti);
 
int main(int argc, char **argv)
{
    //读取高博slam14讲中的groundtruth.txt和estimated.txt的数据
    auto gi = ReadTrajectory(groundtruth_file);
    cout << "gi read total " << gi.size() << endl;
    auto esti = ReadTrajectory(estimated_file);
    cout << "esti read total " << esti.size() << endl;
 
    vector<double> ATE, RPE;
    ATE = calculateATE(gi, esti); //计算ATE
    RPE = calculateRPE(gi, esti); //计算RPE
//    DrawTrajectory(gi, esti);
    return 0;
}
vector<double> calculateATE(const TrajectoryType& gt, const TrajectoryType& esti)
{
    double rmse_all, rmse_trans;
    for(size_t i = 0; i < gt.size(); i++)
    {
        //ATE旋转+平移
        double error_all = (gt[i].inverse()*esti[i]).log().norm();
        rmse_all += error_all * error_all;
        //ATE平移
        double error_trans = (gt[i].inverse()*esti[i]).translation().norm();
        rmse_trans += error_trans * error_trans;
    }
    rmse_all = sqrt(rmse_all / double(gt.size()));
    rmse_trans = sqrt(rmse_trans / double(gt.size()));
    vector<double> ATE;
    ATE.push_back(rmse_all);
    ATE.push_back(rmse_trans);
    cout << "ATE_all = " << rmse_all << " ATE_trans = " << rmse_trans << endl;
}
vector<double> calculateRPE(const TrajectoryType& gt, const TrajectoryType& esti)
{
    double rmse_all, rmse_trans;
    double delta = 1;               //delta = 2, 间隔两帧
    for(size_t i = 0; i < gt.size() - delta; i++)
    {
        //RPE旋转+平移
        double error_all = ((gt[i].inverse()*gt[i + delta]).inverse()*(esti[i].inverse()*esti[i + delta])).log().norm();
        rmse_all += error_all * error_all;
        //RPE平移
        double error_trans = ((gt[i].inverse()*gt[i + delta]).inverse()*(esti[i].inverse()*esti[i + delta])).translation().norm();
        rmse_trans += error_trans * error_trans;
    }
    rmse_all = sqrt(rmse_all / double(gt.size()));
    rmse_trans = sqrt(rmse_trans / double(gt.size()));
    vector<double> RPE;
    RPE.push_back(rmse_all);
    RPE.push_back(rmse_trans);
    cout << "RPE_all = " << rmse_all << " RPE_trans = " << rmse_trans << endl;
}
TrajectoryType ReadTrajectory(const string& file_name)
{
    ifstream fin(file_name);
    TrajectoryType trajectory;
    if(!fin)
    {
        cerr << "trajectory " << file_name << "not found" << endl;
        return trajectory;
    }
    while(!fin.eof())
    {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
        trajectory.push_back(p1);
    }
    return trajectory;
}
