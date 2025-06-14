#ifndef IMAGESTREAM_HPP
#define IMAGESTREAM_HPP

#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <strings.h>
#include <cstdlib>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace zmq;

#define PORT 8890
#define BUF_MAXSIZE 1000000

class ImageStreamSender {

public:
	ImageStreamSender(char strDstIP[], context_t& cxt, socket_t& sock); 

	int sendImg(Mat&);
	
	string recvInfo();

private:
	context_t& context;
	socket_t& socket;
	vector<uchar> imgBuf;
};


class ImageStreamReceiver {
public:
	ImageStreamReceiver(context_t& cxt, socket_t& sock);

	int recvImg(Mat&);

	void sendInfo(string);

private:
	context_t& context;
	socket_t& socket;
	vector<uchar> imgBuf;

};

#endif
