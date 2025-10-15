#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>              
#include <sensor_msgs/msg/image.hpp>          
#include <std_msgs/msg/header.hpp> 
#include <std_msgs/msg/float32.hpp>
#include "MvCameraControl.h"
#include <thread>
#include <atomic>

#include <torch/script.h>
#include <torch/torch.h>
#include <queue>
#include <mutex>
#include <condition_variable>


using namespace std;

static cv::Mat extractRotatedROI(const cv::Mat& src, const cv::RotatedRect& rect) {
    cv::Mat M = (rect.angle < 45) ? cv::getRotationMatrix2D(rect.center, rect.angle, 1.0) : cv::getRotationMatrix2D(rect.center, rect.angle - 90, 1.0);
    cv::Mat rotated;
    cv::warpAffine(src, rotated, M, src.size(), cv::INTER_CUBIC);
    cv::Mat roi;
    cv::getRectSubPix(rotated, rect.size, rect.center, roi);
    return roi;
}

static cv::Mat to28x28GrayFloat(const cv::Mat& src) {
    cv::Mat gray, resized, f32;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src;
    cv::resize(gray, resized, cv::Size(28, 28));
    resized.convertTo(f32, CV_32F, 1.0/255.0);
    return f32; // H×W, 单通道，float32
}

class BOARDdetection : public rclcpp::Node
{
public:
    BOARDdetection() : Node("board_dection_node"), connected_(false)
    {
        RCLCPP_INFO(this->get_logger(), "Camera 启动中...");

        // 创建 ROS2 发布器
        
        this->declare_parameter<double>("exposure_time", 4000.0);
        this->declare_parameter<double>("frame_rate", 30.0);
        this->declare_parameter<double>("gain", 10.0);
        this->declare_parameter<std::string>("pixel_format", "BGR8");

        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&BOARDdetection::onParameterChange, this, std::placeholders::_1)
        );

        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", 10);
        fps_pub_   = this->create_publisher<std_msgs::msg::Float32>("camera/frame_rate", 10); 
        // 初始化相机
        if (!initCamera())
        {
            RCLCPP_ERROR(this->get_logger(), "相机初始化失败，节点启动中止。");
            return;
        }

        // 启动取流定时器

        grab_thread_ = std::thread([this]() {
            rclcpp::Time last_time = this->now();

            while (rclcpp::ok())
            {
                if (connected_) {
                    grabAndShow();
                    auto now = this->now();
                    double elapsed = (now - last_time).seconds();
                    if (elapsed >= 1.0) {
                        MVCC_FLOATVALUE fps_value = {0};
                        int nRet = MV_CC_GetFloatValue(handle_, "ResultingFrameRate", &fps_value);
                        if (nRet == MV_OK) {
                            float fps = static_cast<float>(fps_value.fCurValue);
                            std_msgs::msg::Float32 fps_msg;
                            fps_msg.data = fps;
                            fps_pub_->publish(fps_msg);
                            RCLCPP_INFO(this->get_logger(), "硬件实际帧率: %.2f FPS", fps);
                        } 
                        last_time = now;
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 等待重连，避免占用 CPU
                }
            }
        });

        
        // 启动重连检测线程
        reconnect_thread_ = std::thread([this]() {
            rclcpp::Rate rate(1.0);  // 每秒检测一次
            while (rclcpp::ok())
            {
                while(!connected_ && rclcpp::ok())
                {
                    if (tryReconnect())
                    {
                        RCLCPP_INFO(this->get_logger(), "相机重连成功！");
                        break;
                    }
                }
                rate.sleep();
            }
        });
            // ====== 数字识别：加载模型 + 启动识别线程（异步） ======
        try {
            torch_device_ = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
            // 模型路径：按需调整
            const std::string model_path = "model/model.pt";
            digit_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path, torch_device_));
            digit_module_->eval();
            RCLCPP_INFO(this->get_logger(), "Digit model loaded on %s", (torch_device_.is_cuda() ? "CUDA" : "CPU"));
        } catch (const c10::Error& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load TorchScript model: %s", e.msg().c_str());
        }
        
        // 后台识别线程：从队列取 ROI → 预处理 → 推理
        digit_thread_ = std::thread([this] {
            torch::NoGradGuard nograd;
            while (rclcpp::ok() && !digit_stop_) {
                cv::Mat roi;
                {
                    std::unique_lock<std::mutex> lk(roi_mtx_);
                    roi_cv_.wait(lk, [&]{ return !roi_queue_.empty() || digit_stop_; });
                    if (digit_stop_) break;
                    roi = roi_queue_.front();
                    roi_queue_.pop();
                }
                if (roi.empty()) continue;
        
                // 预处理：H×W float32 单通道
                cv::Mat f32 = to28x28GrayFloat(roi);
                f32 = (f32 - 0.5f) / 0.5f; 
        
                // NCHW: 1×1×28×28
                auto input = torch::from_blob(
                    f32.data, {1, 1, 28, 28},
                    torch::TensorOptions().dtype(torch::kFloat32)
                ).clone().to(torch_device_); // clone 防止与 OpenCV 同一内存的生命周期问题
        
                try {
                    auto out = digit_module_->forward({input}).toTensor();
                    int pred = out.argmax(1).item<int>() ; // 标签是 1~5
                    RCLCPP_INFO(this->get_logger(), "[digit] predict = %d", pred);
                } catch (const std::exception& e) {
                    RCLCPP_WARN(this->get_logger(), "digit inference failed: %s", e.what());
                }
            }
        });
    }



    ~BOARDdetection()
    {
        RCLCPP_INFO(this->get_logger(), "节点关闭，释放资源...");
        
        stopAndCloseCamera();

        if (grab_thread_.joinable())
        grab_thread_.join();

        
        if (reconnect_thread_.joinable())
        reconnect_thread_.join();
        
        MV_CC_Finalize();

        digit_stop_ = true;
        roi_cv_.notify_all();
        if (digit_thread_.joinable()) digit_thread_.join();

    }
    
    private:
    // ---------------- 相机初始化 ----------------
    bool initCamera()
    {
        int nRet = MV_CC_Initialize();
        if (nRet != MV_OK)
        {
            RCLCPP_ERROR(this->get_logger(), "初始化 MVS SDK 失败! nRet [0x%x]", nRet);
            return false;
        }
        
        MV_CC_DEVICE_INFO_LIST devList;
        memset(&devList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &devList);
        if (nRet != MV_OK || devList.nDeviceNum == 0)
        {
            RCLCPP_ERROR(this->get_logger(), "未发现任何相机!");
            return false;
        }
        
        nRet = MV_CC_CreateHandle(&handle_, devList.pDeviceInfo[0]);
        if (nRet != MV_OK)
        {
            RCLCPP_ERROR(this->get_logger(), "创建相机句柄失败! nRet [0x%x]", nRet);
            return false;
        }
        
        nRet = MV_CC_OpenDevice(handle_);
        if (nRet != MV_OK)
        {
            RCLCPP_ERROR(this->get_logger(), "打开相机失败! nRet [0x%x]", nRet);
            return false;
        }
        
        double exposure = this->get_parameter("exposure_time").as_double();
        double fps = this->get_parameter("frame_rate").as_double();
        double gain = this->get_parameter("gain").as_double();

        MV_CC_SetEnumValue(handle_, "TriggerMode", 0);//设置相机的采集模式为连续采集
        MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);//启用相机的帧率设置

        MV_CC_SetFloatValue(handle_, "ExposureTime", exposure);
        MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", fps);
        MV_CC_SetFloatValue(handle_, "Gain", gain);
        MV_CC_SetEnumValue(handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed);//彩色视频流

        nRet = MV_CC_StartGrabbing(handle_);
        if (nRet != MV_OK)
        {
            RCLCPP_ERROR(this->get_logger(), "启动取流失败! nRet [0x%x]", nRet);
            return false;
        }
        
        connected_ = true;
        RCLCPP_INFO(this->get_logger(), "相机连接成功并开始取流。");
        MV_CC_RegisterExceptionCallBack(handle_, ExceptionCallback, this);
        return true;
    }
    
    // ---------------- 异常回调（断线检测） ----------------
    static void __stdcall ExceptionCallback(unsigned int nMsgType, void* pUser)
    {
        if (nMsgType == MV_EXCEPTION_DEV_DISCONNECT)
        {
            auto* node = static_cast<BOARDdetection*>(pUser);
            node->connected_ = false;
            RCLCPP_WARN(node->get_logger(), "相机已断开连接，重连中...");
        }
    }
    
    // ---------------- 自动重连 ----------------
    bool tryReconnect()
    {
        stopAndCloseCamera();
        this_thread::sleep_for(std::chrono::milliseconds(500));

        RCLCPP_INFO(this->get_logger(), "正在尝试重新初始化相机...");
        return initCamera();
    }

    // ---------------- 释放资源 ----------------
    void stopAndCloseCamera()
    {
        if (handle_)
        {
            MV_CC_StopGrabbing(handle_);
            MV_CC_CloseDevice(handle_);
            MV_CC_DestroyHandle(handle_);
            handle_ = nullptr;
            connected_ = false;
        }
    }

    float angle_change(float &angle){
        if(angle > 90){
            angle = angle -180.0;
        }
        return angle;
    }

    //------------------装甲板识别--------------------
    void detection(cv::Mat &frame)
    {
            cv::Mat high_ori, high, gray, high1, last_frame, last_high1,high_ada;
            cv::Mat num_roi = frame.clone();//为了数字识别做准备
            cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
            cv::threshold(gray,high_ori,150,255,cv::THRESH_BINARY);
            cv::adaptiveThreshold(gray,high_ada,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,51,0);
            //cv::threshold(gray,high,0,255,cv::THRESH_BINARY | cv::THRESH_OTSU);
            //cv::imshow("high_ori",high_ori);
            //cv::imshow("high_ada",high_ada);

            cv::bitwise_and(high_ori, high_ada, high);
            cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            
            cv::medianBlur(high,high,3);
            cv::erode(high, high1, kernel1);
            cv::dilate(high1, high1, kernel1); 
            //high1 = high.clone();
            cv::imshow("gray",high1);

            cv::Mat channels[3];
            cv::split(frame, channels);         // 三通道分离

            cv::Mat blue_sub_red = channels[0] - channels[2];    // 红蓝通道相减
            cv::Mat normal_mat,high_blue;
            cv::normalize(blue_sub_red, normal_mat, 0., 255., cv::NORM_MINMAX);  
            cv::threshold(normal_mat, high_blue, 150 , 255, cv::THRESH_BINARY);
            //cv::medianBlur(high_blue,high_blue,7);
            cv::dilate(high_blue, high_blue, kernel2);             
            cv::erode(high_blue, high_blue, kernel2);
            
            //cv::imshow("BLUE",high_blue);

            // cv::Mat hsv,high_blue;
            // cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
            // cv::inRange(hsv,cv::Scalar(100,43,46),cv::Scalar(124,255,255),high_blue);
            cv::imshow("BLUE",high_blue);
            cv::bitwise_and(high1, high_blue, high1);


            cv::Mat edges = high1.clone();
            vector<vector<cv::Point>> contours;
            vector<cv::Vec4i> hierachy;
            cv::findContours(edges, contours, hierachy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            
            //寻找旋转矩形并绘制，可以通过旋转矩形的长宽比来筛选
            vector<cv::RotatedRect> rbbox;
            for(int i = 0;i<contours.size();i++){
                cv::Point2f ver[4];
                rbbox.push_back(cv::minAreaRect(contours[i]));
                rbbox[i].points(ver);
                for(int j = 0;j<4;j++){
                    //cv::line(frame,ver[j],ver[(j+1)%4],cv::Scalar(0,0,255),2);
                }
            }

            //cout << rbbox.size() << endl;
            vector<cv::RotatedRect> rbbox_f;
            for(int i = 0;i<rbbox.size();i++){
                float rbbox_height = max(rbbox[i].size.width, rbbox[i].size.height);
                float rbbox_width = min(rbbox[i].size.width, rbbox[i].size.height);
                float hw_ratio = rbbox_height/rbbox_width;
                if(hw_ratio > 2.5){
                    rbbox_f.push_back(rbbox[i]);
                }
            }

            vector<cv::Rect> lightbox;
            vector<cv::Rect> board;
            int board_num = 0;
            for(int i = 0;i<rbbox_f.size();i++){
                //cout << i << "的角度" << rbbox_f[i].angle << endl;
                lightbox.push_back(rbbox_f[i].boundingRect());
                //rectangle(frame, lightbox[i], cv::Scalar(255,0,0), 2);
            }


            //打算加一段灯条配对的逻辑，也就是两个旋转矩形的旋转角度差在一定范围内
            for(int i = 1;i<lightbox.size();i++){
                board.push_back(lightbox[i] | lightbox[i-1]);
                //rectangle(frame, board[i-1], cv::Scalar(0,0,255), 2);
            }

            cv::Point2f center;
            for(int i = 0;i<board.size();i++){
                cv::circle(frame,center,2,cv::Scalar(0,0,255),-1);
                center = cv::Point(board[i].x + board[i].width/2.0,board[i].y + board[i].height/2.0);
            }

            double box1_angle, box2_angle;
            
            if(rbbox_f.size()==2){
                cv::RotatedRect box1,box2;
                box1 = rbbox_f[0];
                box2 = rbbox_f[1];
                // box1_angle = angle_change(box1.angle);
                // box2_angle = angle_change(box2.angle);
                double angle;
                if(abs(box1.angle - box2.angle) > 45){
                    angle = abs((box1.angle + box2.angle)/2.0 - abs(box1.angle - box2.angle)/2.0);
                }
                else{
                    angle = (box1.angle + box2.angle)/2.0;
                }
                //angle = (box1_angle + box2_angle)/2.0;
                cv::Point2f p1 = rbbox_f[0].center;
                cv::Point2f p2 = rbbox_f[1].center;
                cv::Vec2f diff(p1.x - p2.x, p1.y - p2.y);
                double d = cv::norm(diff);
                cv::RotatedRect boarddetected(center, cv::Size(d*0.85,d*0.85),angle);
                cv::Point2f ver1[4];                
                boarddetected.points(ver1);
                for(int j = 0;j<4;j++){
                    cv::line(frame,ver1[j],ver1[(j+1)%4],cv::Scalar(0,0,255),3);
                }  
                
                double height = (max(box1.size.width, box1.size.height)+max(box2.size.width, box2.size.height))/2.0;
                cv::RotatedRect Rotated_light_box = (angle > 45 ) ? cv::RotatedRect(center, cv::Size(height,d),angle) : cv::RotatedRect(center, cv::Size(d,height),angle);
                cv::Point2f ver2[4];
                Rotated_light_box.points(ver2);
                for(int j = 0;j<4;j++){
                    cv::line(frame,ver2[j],ver2[(j+1)%4],cv::Scalar(0,255,0),3);
                }
                
                // === 把 ROI 送进识别线程（非阻塞） ===
                try {
                    cv::Mat roi = extractRotatedROI(frame, Rotated_light_box);
                    if (!roi.empty() && roi.cols >= 12 && roi.rows >= 12) {
                        std::lock_guard<std::mutex> lk(roi_mtx_);
                        // 防止队列无限增长：限制一下排队量
                        if (roi_queue_.size() < 5) {
                            roi_queue_.push(roi.clone());
                            roi_cv_.notify_one();
                        }
                    }
                } catch (const std::exception& e) {
                    RCLCPP_WARN(this->get_logger(), "extractRotatedROI failed: %s", e.what());
                }


                cv::Point2f jpoint[4];
                Rotated_light_box.points(jpoint); //从左上角开始，顺时针提取
                // for(int j = 0;j<4;j++){
                    //     cout << jpoint[j] << endl;
                    // }
                    // cout << "//" << endl;
                    //cout << angle << endl;
                    double w = 135;
                    double h = 56;
                    
                    vector<cv::Point3f> armor_3D = {
                        {-w/2,  h/2, 0},   // 左上
                        { w/2,  h/2, 0},  // 右上
                        { w/2, -h/2, 0},  // 右下
                        {-w/2, -h/2, 0},  // 左下
                    };
                    
                    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) <<
                    4621, 0, 771,
                    0, 4622, 636,
                    0, 0, 1);
                    cv::Mat distCoeffs = (cv::Mat_<double>(1,5) << -0.065, 0.7247, 0, 0, 0);
                    
                    std::vector<cv::Point2f> imagePoints(jpoint, jpoint + 4);
                    
                    cv::Mat rvec, tvec;
                    cv::solvePnP(
                        armor_3D,
                        imagePoints,
                        cameraMatrix,
                        distCoeffs,
                        rvec,
                        tvec,
                        false,
                        cv::SOLVEPNP_IPPE_SQUARE   // 针对平面矩形目标最优
                    );
                    
                    //cout << "tvec = \n" << tvec << endl;
                    
                    imshow("ROI",to28x28GrayFloat(extractRotatedROI(num_roi,boarddetected)));

                
            }
     

            // vector<cv::Rect> board_f;
            // for(int i = 0;i<board.size();i++){
            //     if(board[i].width / board[i].height > 1){
            //         board_f.push_back(board[i]);
            //         board_num ++;
            //         //cv::rectangle(frame, board[i], cv::Scalar(0,0,255),2);
            //     }
            // }

            //cout << board_num << endl;
            
            frame.copyTo(last_frame);
            //high1.copyTo(last_high1);

        //}

        cv::imshow("origin",last_frame);
        //cv::imshow("gray",last_high1);
    }


    // ---------------- 获取并发布图像 ----------------
    void grabAndShow()
    {
        if (!connected_ || !handle_) return;

        MV_FRAME_OUT frame = {0};
        int nRet = MV_CC_GetImageBuffer(handle_, &frame, 1000);
        if (nRet == MV_OK)
        {
            std::vector<unsigned char> bgrBuf(frame.stFrameInfo.nWidth * frame.stFrameInfo.nHeight * 3);

            MV_CC_PIXEL_CONVERT_PARAM cvt{};
            cvt.nWidth = frame.stFrameInfo.nWidth;
            cvt.nHeight = frame.stFrameInfo.nHeight;
            cvt.enSrcPixelType = frame.stFrameInfo.enPixelType;
            cvt.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
            cvt.pSrcData = frame.pBufAddr;
            cvt.nSrcDataLen = frame.stFrameInfo.nFrameLen;
            cvt.pDstBuffer = bgrBuf.data();
            cvt.nDstBufferSize = bgrBuf.size();
            MV_CC_ConvertPixelType(handle_, &cvt);

            cv::Mat img(frame.stFrameInfo.nHeight, frame.stFrameInfo.nWidth, CV_8UC3, bgrBuf.data());
            cv::Mat imgcopy = img.clone();
            detection(imgcopy);
            //cv::imshow("board_detection", img);

            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", imgcopy).toImageMsg();
            msg->header.stamp = this->now();
            image_pub_->publish(*msg);


            cv::waitKey(1);
            MV_CC_FreeImageBuffer(handle_, &frame);
        }
    }

    // ---------------- 参数动态调整回调 ----------------
    rcl_interfaces::msg::SetParametersResult onParameterChange(const std::vector<rclcpp::Parameter> &params)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        for (const auto &param : params)
        {
            if (param.get_name() == "exposure_time")
            {
                double value = param.as_double();
                if (handle_ && connected_)
                    MV_CC_SetFloatValue(handle_, "ExposureTime", value);
            }
            else if (param.get_name() == "frame_rate")
            {
                double value = param.as_double();
                if (handle_ && connected_)
                    MV_CC_SetFloatValue(handle_, "AcquisitionFrameRate", value);
            }
            else if (param.get_name() == "gain")
            {
                double value = param.as_double();
                if (handle_ && connected_)
                    MV_CC_SetFloatValue(handle_, "Gain", value);
            }
            else if (param.get_name() == "pixel_format")
            {
                std::string fmt = param.as_string();
                if (handle_ && connected_) {
                    if (fmt == "Mono8")
                        MV_CC_SetEnumValue(handle_, "PixelFormat", PixelType_Gvsp_Mono8);
                    else if (fmt == "BayerRG8")
                        MV_CC_SetEnumValue(handle_, "PixelFormat", PixelType_Gvsp_BayerRG8);
                    else if (fmt == "BGR8")
                        MV_CC_SetEnumValue(handle_, "PixelFormat", PixelType_Gvsp_BGR8_Packed);
                    else
                        RCLCPP_WARN(this->get_logger(), "不支持的 PixelFormat: %s", fmt.c_str());
                        result.successful = false;
                }
            }
            
        }
        return result;
    }


    // --- 成员变量 ---
    void* handle_ = nullptr;
    std::atomic<bool> connected_;
    std::thread grab_thread_;
    std::thread reconnect_thread_;
    //rclcpp::TimerBase::SharedPtr fps_timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr fps_pub_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
    
    // --- 数字识别（异步线程） ---
    std::unique_ptr<torch::jit::script::Module> digit_module_;
    std::thread digit_thread_;
    std::queue<cv::Mat> roi_queue_;
    std::mutex roi_mtx_;
    std::condition_variable roi_cv_;
    std::atomic<bool> digit_stop_{false};
    torch::Device torch_device_ = torch::kCPU;

};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BOARDdetection>());
    rclcpp::shutdown();
    return 0;
}
