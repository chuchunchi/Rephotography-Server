KADFP的訓練方法參考以前學長的交接檔：https://hackmd.io/@raymond09/BJ-o7u34T

使用方法：
1、啓動server
	a.開啓終端： cd /home/covis-lab-00/xr/DAV2/metric_depth/
	b.執行： python server.py --params_file [*] --datasets [*] --encoder [*] --max_depth [*] --target [*] --ex_name [*] --ins
	c.參數說明： params_file是相機參數檔。--datasets和encoder是dav2的必要參數，datasets從['hypersim', 'vkitti']中選擇，分別是室內用和室外用，--encoder默認使用vitb。--max_depth需要根據場景的大致深度進行調整，一般室內20，室外60。--target是存放錄製的參考影片的資料夾名稱，--ex_name是存放當次實驗的資料夾名稱（錄製模式可不填，無需後綴）,--ins標誌表示開啓巡檢模式。
	d.範例： python server.py  --params_file ../../data/M2Pro.yaml --datasets vkitti --max_depth 60 --target 0113_1 --ex_name 0113_1 --ins
	e.client停止後按一次ctrl+c關閉server，錄製模式下會保存參考影片到./data/target/你的資料夾/ 中，巡檢模式下會保存結果影片到./data/result/你的資料夾/ 中。
	
2、啓動client
	a.新開一個終端： cd /home/covis-lab-00/xr/ORBSLAM3/
	b.如果有寫新的client程式，要修改CMakeLists.txt，再最下面添加(修改方括號內容)：
add_executable([你的檔名] Examples/droneClient/[你的檔名].cc Examples/droneClient/VideoSocketUDP_rn.cpp Examples/droneClient/Controller.cpp Examples/droneClient/pid.cpp)
target_link_libraries([你的檔名] ${PROJECT_NAME})
	c.執行： ./build.sh   編譯成執行檔
	d.進入執行檔所在資料夾： cd ./Examples/droneClient/
	e.執行： ./client [相機參數檔] [參考影片資料夾名稱] [當次實驗資料夾名稱] [巡檢模式標誌] [僅KADFP標誌] [輸入模式] [模擬影片路徑]
	f.參數說明： [相機參數檔]跟server那邊是同一個檔案。[參考影片資料夾名稱]和[當次實驗資料夾名稱]跟server一樣是資料夾名稱。[巡檢模式標誌]0表示錄製模式，1表示巡檢模式。[僅KADFP標誌]0表示同時使用ORB-SLAM3和KADFP控制無人機移動，1表示只使用KADFP結果控制無人機。[輸入模式]0表示使用無人機畫面，1表示使用模擬影片測試。[模擬影片路徑]不用模擬影片時可不填。
	g.結束後根據終端提示關閉client（鍵盤輸入一定要先選中ORB-SLAM3的影像窗口，否則無效）。
	
3、評估
	a.開啓終端： cd /home/covis-lab-00/xr/DAV2/metric_depth/
	b.執行： python DFAE.py -t [*] -r [*] -m [*]
	c.參數說明： -t是參考影片的資料夾名稱，-r是結果影片的資料夾名稱，-m是兩個影片在ORB-SLAM3中map merge時的幀id，不填則默認0,0，會完整對比兩個影片。
	d.範例：python DFAE.py -t 0113_1 -r 0113_1 -m 0,0
        e.結果影片會存到./data/result/你的資料夾/DFAE/ 中。
        
