# cmake_minimum_required(VERSION 3.8)
# project(board_detection)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# # find dependencies
# find_package(ament_cmake REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(std_msgs REQUIRED)
# find_package(sensor_msgs REQUIRED)
# find_package(cv_bridge REQUIRED)
# find_package(OpenCV REQUIRED)

# # Include directories
# set(MVS_SDK_PATH ./../../)
# include_directories(${MVS_SDK_PATH}/include)
# link_directories(${MVS_SDK_PATH}/lib)

# # Create executable
# add_executable(detection_node src/main.cpp)
# target_link_libraries(detection_node
#   MvCameraControl
#   MediaProcess
#   ${OpenCV_LIBS}
# )
# ament_target_dependencies(detection_node
#   rclcpp
#   std_msgs
#   sensor_msgs
#   cv_bridge
# )

# # Install
# install(TARGETS
#   detection_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# install(DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}
# )

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   set(ament_cmake_copyright_FOUND TRUE)
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

# ament_package() 

cmake_minimum_required(VERSION 3.8)
project(board_detection)

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# C++17 for libtorch
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------ 查找依赖 ------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# ------------------ libtorch ------------------
set(Torch_DIR "/home/zngyim/libs/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
message(STATUS "Found libtorch: ${TORCH_LIBRARIES}")

# ------------------ 海康 MVS SDK ------------------
set(MVS_SDK_PATH "/home/zngyim/Desktop/XJTU-RMV-Task06")  # 按你项目路径改
include_directories(${MVS_SDK_PATH}/include)
link_directories(${MVS_SDK_PATH}/lib)

# ------------------ 编译目标 ------------------
add_executable(detection_node src/main.cpp)

set_target_properties(detection_node PROPERTIES
    BUILD_RPATH "/home/zngyim/libs/libtorch/lib"
    INSTALL_RPATH "/home/zngyim/libs/libtorch/lib"
)

# 链接库
target_link_libraries(detection_node
  MvCameraControl
  MediaProcess
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
)

# ament 依赖
ament_target_dependencies(detection_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
)

# ------------------ Torch 设置 ------------------
# 防止 Libtorch 多线程冲突
set_property(TARGET detection_node PROPERTY CXX_STANDARD 17)
set_property(TARGET detection_node PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(detection_node PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

# ------------------ 安装 ------------------
install(TARGETS
  detection_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
