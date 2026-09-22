cmake_minimum_required(VERSION 3.18)
project(ORB_SLAM3_EXAMPLES LANGUAGES CXX)

# DTO-AR keeps the RGB-D entry point used by the structured-light pipeline.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/RGB-D)

add_executable(rgbd_tum
        RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum PRIVATE ${TARGET_ORB_SLAM3})
