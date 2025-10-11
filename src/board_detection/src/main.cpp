#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>              
#include <sensor_msgs/msg/image.hpp>          
#include <std_msgs/msg/header.hpp> 
#include <std_msgs/msg/float32.hpp>
#include "MvCameraControl.h"
#include <thread>
#include <atomic>

using namespace std;

class BOARDdetection : public rclcpp::Node
{
public:
    BOARDdetection() : Node("board_dection_node"), connected_(false)
    {
        RCLCPP_INFO(this->get_logger(), "Camera 启动中...");

        // 创建 ROS2 发布器
        
        this->declare_parameter<double>("exposure_time", 1000.0);
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

    //------------------装甲板识别--------------------
    void detection(cv::Mat &frame)
    {
            cv::Mat high_ori, high, gray, high1, last_frame, last_high1,high_ada;
            cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
            cv::threshold(gray,high_ori,150,255,cv::THRESH_BINARY);
            cv::adaptiveThreshold(gray,high_ada,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,51,0);
            //cv::threshold(gray,high,0,255,cv::THRESH_BINARY | cv::THRESH_OTSU);
            //cv::imshow("high_ori",high_ori);
            //cv::imshow("high_ada",high_ada);

            cv::bitwise_and(high_ori, high_ada, high);
            cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
            cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            
            cv::medianBlur(high,high,7);
            cv::erode(high, high1, kernel1);
            cv::dilate(high1, high1, kernel1); 
            
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
            // cv::imshow("BLUE",high_blue);
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
                if(hw_ratio > 2.5 && hw_ratio < 7){
                    rbbox_f.push_back(rbbox[i]);
                }
            }

            vector<cv::Rect> lightbox;
            vector<cv::Rect> board;
            int board_num = 0;
            for(int i = 0;i<rbbox_f.size();i++){
                lightbox.push_back(rbbox_f[i].boundingRect());
                //rectangle(frame, lightbox[i], cv::Scalar(255,0,0), 2);
            }

            for(int i = 1;i<lightbox.size();i++){
                board.push_back(lightbox[i] | lightbox[i-1]);
                rectangle(frame, board[i-1], cv::Scalar(0,0,255), 2);
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
            
            //frame.copyTo(last_frame);
            //high1.copyTo(last_high1);

        //}

        //cv::imshow("origin",last_frame);
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
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BOARDdetection>());
    rclcpp::shutdown();
    return 0;
}
