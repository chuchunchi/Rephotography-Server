#include "ImageStream.hpp"

ImageStreamSender::ImageStreamSender(char strDstIP[], context_t& cxt, socket_t& sock) 
	: socket(sock), context(cxt) {

	char dst[128];
	sprintf(dst, "tcp://%s:%d", strDstIP, PORT);

	socket.connect(dst);

}

int ImageStreamSender::sendImg(Mat& img) {

	imgBuf.clear();
	imencode(".jpg", img, imgBuf);

	message_t msgBuf(imgBuf.size());
	memcpy(msgBuf.data(), &imgBuf[0], imgBuf.size());

	socket.send(msgBuf);
	message_t msg;
	socket.recv(&msg);
}

string ImageStreamSender::recvInfo() {

	message_t sendMsg(8);
	memcpy(sendMsg.data(), "recvInfo", 8);

	socket.send(sendMsg);
	message_t msg;
	socket.recv(&msg);
	string reply = string(static_cast<char*>(msg.data()), msg.size());

	return reply;
}

ImageStreamReceiver::ImageStreamReceiver(context_t& cxt, socket_t& sock)
	: socket(sock), context(cxt) {

	char dst[128];
	sprintf(dst, "tcp://*:%d", PORT);

	socket.bind(dst);
}

int ImageStreamReceiver::recvImg(Mat& img) {

	message_t content;
	socket.recv(&content);

	string strContent = string(static_cast<char*>(content.data()), content.size());
	message_t ack(3);
	memcpy(ack.data(), "ack", 3);
	socket.send(ack);

	imgBuf.clear();
	for(int i = 0; i < strContent.size(); i++) {
		imgBuf.push_back(strContent[i]);
	}

	img = imdecode(imgBuf, 1);
	return 0;
}

void ImageStreamReceiver::sendInfo(string info) {
	message_t msg;	
	
	socket.recv(&msg);
	message_t sendMsg(info.size());
	memcpy(sendMsg.data(), info.c_str(), info.size());
	socket.send(sendMsg);
}
