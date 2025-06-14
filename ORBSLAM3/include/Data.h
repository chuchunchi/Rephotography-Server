#ifndef DATA_H
#define	DATA_H

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <math.h>
using namespace std;

class imgInfo{ 
public:
    int index;
    cv::Mat origin;
    cv::Mat pos;
    cv::Mat relativeRT;
    int needNewKeyFrame;
    double time;
    bool posFixed;
};
class frameInfo{
public:
    int frameIdx;
    int inlierNum;
};
class mapInfo{ // SFM model information
public:
    int index;
    cv::Mat pos;
};
class mapFunc{
public:
	void LoadMap(const string filename, vector<mapInfo> *Map){
		int mapPointNum;
	    cout << "map loading..." << endl;
	    cout << "filename: " << filename << endl;
	    cv::Mat mapPos(3, 1, CV_32F);
	    mapInfo tempMap;
	    ifstream mapFile;
	    mapFile.open(filename);
	    mapFile >> mapPointNum;

	    for(int i=1 ; i<=mapPointNum ; i++){
	        mapFile >> mapPos.at<float>(0,0)
	                >> mapPos.at<float>(1,0)
	                >> mapPos.at<float>(2,0);
	        tempMap.pos = mapPos.clone();
	        tempMap.index = i;
	        Map->push_back(tempMap);
	    }
	    mapFile.close();
	}
	void StoreCameraInfo(const string filename, vector<imgInfo> trackingRecord){
		ofstream f;
	    f.open(filename);
	    cout << "size: " << trackingRecord.size() << endl;
	    for(int i=0 ; i<trackingRecord.size() ; i++){
	        //f << "Tracking record " << trackingRecord[i].index << ": " << endl << trackingRecord[i].pos << endl;
	        if(trackingRecord[i].pos.empty()){
	            f << trackingRecord[i].index << endl;
	            continue;
	        }
	        f << trackingRecord[i].index << " " << trackingRecord[i].needNewKeyFrame << " " << trackingRecord[i].time << " " << 
	        
	        trackingRecord[i].relativeRT.at<float>(0,0) << " " << trackingRecord[i].relativeRT.at<float>(0,1) << " " << trackingRecord[i].relativeRT.at<float>(0,2) << " " << 
	        trackingRecord[i].relativeRT.at<float>(1,0) << " " << trackingRecord[i].relativeRT.at<float>(1,1) << " " << trackingRecord[i].relativeRT.at<float>(1,2) << " " <<
	        trackingRecord[i].relativeRT.at<float>(2,0) << " " << trackingRecord[i].relativeRT.at<float>(2,1) << " " << trackingRecord[i].relativeRT.at<float>(2,2) << " " << 
	        trackingRecord[i].relativeRT.at<float>(0,3) << " " << trackingRecord[i].relativeRT.at<float>(1,3) << " " << trackingRecord[i].relativeRT.at<float>(2,3) << " " <<
	        
	        trackingRecord[i].pos.at<float>(0,0) << " " << trackingRecord[i].pos.at<float>(0,1) << " " << trackingRecord[i].pos.at<float>(0,2) << " " << 
	        trackingRecord[i].pos.at<float>(1,0) << " " << trackingRecord[i].pos.at<float>(1,1) << " " << trackingRecord[i].pos.at<float>(1,2) << " " <<
	        trackingRecord[i].pos.at<float>(2,0) << " " << trackingRecord[i].pos.at<float>(2,1) << " " << trackingRecord[i].pos.at<float>(2,2) << " " << 
	        trackingRecord[i].pos.at<float>(0,3) << " " << trackingRecord[i].pos.at<float>(1,3) << " " << trackingRecord[i].pos.at<float>(2,3) << " " <<
	        
	        endl;
	        
	    } 
	    f.close();
	}
	
};

enum{
    Global,
    Predict
};

enum{
    ClientFixing,
    ServerFixing
};
#endif