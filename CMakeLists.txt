cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()
project(ORB_SLAM3 LANGUAGES CXX)

# Build type (default to Release)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Debug postfix
set(CMAKE_DEBUG_POSTFIX "_d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "_rd")

# Check C++11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DTO_AR_CUDA_ARCHITECTURE 86 CACHE STRING "CUDA architecture used by DTO-AR dense optimization")
cmake_policy(SET CMP0057 NEW) # enable IN_LIST operator
if(cxx_std_11 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  add_definitions(-DCOMPILEDWITHC11)
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# Compiler specific options for gcc
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
  
  # suppress warnings about deprecations (many in Eigen)
  add_compile_options("-Wno-deprecated-declarations")
  
  # @soeroesg: Disable this flag because it causes trouble with g2o
  #set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
  #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()
# Compiler specific options for MSVC
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /bigobj")
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
if ("${CMAKE_GENERATOR}" MATCHES "Visual Studio")
  # we don't support the Debug and Release subdirs
  set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Available build configurations" FORCE)
  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${config}" config_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  endforeach ()
endif ()



# Target definition
set(TARGET_ORB_SLAM3 ORB_SLAM3)
set(ORB_SLAM3_LIB_TYPE SHARED)
IF(WIN32)
  # uncomment only one of the lines below:
  # because there is no DLL API in the code,
  # we either have to build STATIC libraries,
  set(ORB_SLAM3_LIB_TYPE STATIC)
  # or we have to export all symbols from all SHARED libraries
  #set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  # Note that the ideal solution would be to add an export API to the relevant classes.
ENDIF(WIN32)


set(ORB_SLAM3_SOURCES
  src/System.cc
  src/Tracking.cc
  src/LocalMapping.cc
  src/LoopClosing.cc
  src/ORBextractor.cc
  src/ORBmatcher.cc
  src/FrameDrawer.cc
  src/Converter.cc
  src/MapPoint.cc
  src/KeyFrame.cc
  src/Atlas.cc
  src/Map.cc
  src/MapDrawer.cc
  src/Optimizer.cc
  src/Frame.cc
  src/KeyFrameDatabase.cc
  src/Sim3Solver.cc
  src/Viewer.cc
  src/ImuTypes.cc
  src/G2oTypes.cc
  src/OptimizableTypes.cpp
  src/MLPnPsolver.cpp
  src/GeometricTools.cc
  src/TwoViewReconstruction.cc
  src/Config.cc
  src/Settings.cc
  src/Scanner.cpp
  src/CameraModels/Pinhole.cpp
  src/CameraModels/KannalaBrandt8.cpp
)

set(ORB_SLAM3_HEADERS
  include/System.h
  include/Tracking.h
  include/LocalMapping.h
  include/LoopClosing.h
  include/ORBextractor.h
  include/ORBmatcher.h
  include/FrameDrawer.h
  include/Converter.h
  include/MapPoint.h
  include/KeyFrame.h
  include/Atlas.h
  include/Map.h
  include/MapDrawer.h
  include/Optimizer.h
  include/Frame.h
  include/KeyFrameDatabase.h
  include/Sim3Solver.h
  include/Viewer.h
  include/ImuTypes.h
  include/G2oTypes.h
  include/OptimizableTypes.h
  include/MLPnPsolver.h
  include/GeometricTools.h
  include/TwoViewReconstruction.h
  include/SerializationUtils.h
  include/Config.h
  include/Settings.h
  include/CameraModels/GeometricCamera.h
  include/CameraModels/Pinhole.h
  include/CameraModels/KannalaBrandt8.h
  include/Load_metadata.h
  include/IncrementalMeshing.h
  include/ICP_svd.hpp
  include/Scanner.h
  include/DenseOptimizer.cuh
)

add_library(${TARGET_ORB_SLAM3} ${ORB_SLAM3_LIB_TYPE}
  ${ORB_SLAM3_HEADERS} ${ORB_SLAM3_SOURCES}
)

target_include_directories(${TARGET_ORB_SLAM3} PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/CameraModels>
  $<INSTALL_INTERFACE:include/${TARGET_ORB_SLAM3}>
  $<INSTALL_INTERFACE:include/${TARGET_ORB_SLAM3}/CameraModels>
)

option(ORB_SLAM3_USE_OWN_DBoW2 "Use DBoW2 from Thirdparty folder" TRUE)
option(ORB_SLAM3_USE_OWN_g2o "Use g2o from Thirdparty folder" TRUE)
option(ORB_SLAM3_USE_OWN_hashlibpp "Use hashlibpp from Thirdparty folder" TRUE)
option(ORB_SLAM3_USE_OWN_Sophus "Use Sophus from Thirdparty folder" TRUE)
option(ORB_SLAM3_BUILD_EXAMPLES "Build examples" TRUE)
option(ORB_SLAM3_BUILD_EXAMPLES_OLD "Build old examples" TRUE)

# Dependencies

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)

if(WIN32)
  if(NOT DEFINED OpenCV_DIR AND EXISTS "D:/opencv/build/OpenCVConfig.cmake")
    set(OpenCV_DIR "D:/opencv/build" CACHE PATH "OpenCV config directory")
  endif()
  if(NOT DEFINED Eigen3_DIR AND EXISTS "D:/eigen-3.4.0/build/Eigen3Config.cmake")
    set(Eigen3_DIR "D:/eigen-3.4.0/build" CACHE PATH "Eigen3 config directory")
  endif()
  if(NOT DEFINED DTO_AR_EIGEN3_INCLUDE_DIR AND EXISTS "D:/eigen-3.4.0/Eigen/Core")
    set(DTO_AR_EIGEN3_INCLUDE_DIR "D:/eigen-3.4.0" CACHE PATH "DTO-AR Eigen include directory")
  endif()
  if(NOT DEFINED Pangolin_DIR AND EXISTS "D:/ins_att/ORB_WIN/Thirdparty/Pangolin/build/src/PangolinConfig.cmake")
    set(Pangolin_DIR "D:/ins_att/ORB_WIN/Thirdparty/Pangolin/build/src" CACHE PATH "Pangolin config directory")
  endif()
  if(NOT DEFINED DTO_AR_PANGOLIN_ROOT AND EXISTS "D:/ins_att/ORB_WIN/Thirdparty/Pangolin/include/pangolin/pangolin.h")
    set(DTO_AR_PANGOLIN_ROOT "D:/ins_att/ORB_WIN/Thirdparty/Pangolin" CACHE PATH "DTO-AR Pangolin installation root")
  endif()
  if(NOT DEFINED Boost_DIR AND EXISTS "D:/boost_1_83_0/stage/lib/cmake/Boost-1.83.0/BoostConfig.cmake")
    set(Boost_DIR "D:/boost_1_83_0/stage/lib/cmake/Boost-1.83.0" CACHE PATH "Boost config directory")
  endif()
  if(NOT DEFINED DTO_AR_BOOST_INCLUDE_DIR AND EXISTS "D:/boost_1_83_0/boost")
    set(DTO_AR_BOOST_INCLUDE_DIR "D:/boost_1_83_0" CACHE PATH "DTO-AR Boost include directory")
  endif()
  if(NOT DEFINED PhoXiControlPath)
    if(DEFINED ENV{PHOXI_CONTROL_PATH})
      set(PhoXiControlPath "$ENV{PHOXI_CONTROL_PATH}" CACHE PATH "PhoXi Control installation directory")
    elseif(EXISTS "D:/photoneo/PhoXiControl-1.13.4/PhoXiConfig.cmake")
      set(PhoXiControlPath "D:/photoneo/PhoXiControl-1.13.4" CACHE PATH "PhoXi Control installation directory")
    endif()
  endif()
  if(NOT DEFINED OpenSceneGraph_ROOT AND EXISTS "D:/OSG/OpenSceneGraph-3.6.5-VC2022-64-Release-2023-01/include")
    set(OpenSceneGraph_ROOT "D:/OSG/OpenSceneGraph-3.6.5-VC2022-64-Release-2023-01" CACHE PATH "OpenSceneGraph root directory")
  endif()
endif()

# DBoW2
if(ORB_SLAM3_USE_OWN_DBoW2)
  add_subdirectory(Thirdparty/DBoW2)
  target_include_directories(${TARGET_ORB_SLAM3} PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2>
      $<INSTALL_INTERFACE:include/DBoW2>
  )
else()
  find_package(DBoW2 REQUIRED)
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC DBoW2)

# G2O
if(ORB_SLAM3_USE_OWN_g2o)
  add_subdirectory(Thirdparty/g2o)
  target_include_directories(${TARGET_ORB_SLAM3} PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Thirdparty/g2o>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Thirdparty/g2o/g2o>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Thirdparty/g2o/g2o/core>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Thirdparty/g2o/g2o/solvers>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Thirdparty/g2o/g2o/stuff>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Thirdparty/g2o/g2o/types>
      $<INSTALL_INTERFACE:include/g2o>
  )
else()
  find_package(g2o REQUIRED)
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC g2o)

# Sophus
if(ORB_SLAM3_USE_OWN_Sophus)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(BUILD_TESTS OFF)
  set(BUILD_EXAMPLES OFF)
  add_subdirectory(Thirdparty/Sophus)
  # (include public because used in OS3 headers)
  target_include_directories(${TARGET_ORB_SLAM3} PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Thirdparty/Sophus>
      $<INSTALL_INTERFACE:>
  )
else()
  find_package(Sophus REQUIRED)
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC sophus)

# hashlibpp
if(ORB_SLAM3_USE_OWN_hashlibpp)
  add_subdirectory(Thirdparty/hashlibpp)
else()
  find_package(hashlibpp REQUIRED)
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PRIVATE hashlibpp)
target_compile_definitions(${TARGET_ORB_SLAM3} PRIVATE USE_HASHLIBPP)
target_compile_definitions(${TARGET_ORB_SLAM3} PUBLIC DTO_AR_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

# OpenSSL is used only in System.cc for MD5 hash calculation
# but because OpenSSL is not trivial to compile on windows, we can use hashlibpp instead.
#set(OPENSSL_USE_STATIC_LIBS TRUE)
#find_package(OpenSSL) # for crypto library
#if(OpenSSL_FOUND)
#  target_include_directories(${TARGET_ORB_SLAM3} ${OPENSSL_INCLUDE_DIR})
#  target_link_libraries(${TARGET_ORB_SLAM3} ${OPENSSL_CRYPTO_LIBRARY})
#endif()

# OpenCV
find_package(OpenCV 4.0 QUIET)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3.0 QUIET)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 3.0 not found.")
  endif()
endif()
message(STATUS "OpenCV VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC ${OpenCV_LIBS})

# Eigen
find_package(Eigen3 3.1.0 REQUIRED)
#target_include_directories(${TARGET_ORB_SLAM3} PRIVATE ${EIGEN3_INCLUDE_DIR})
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3_VERSION: ${Eigen3_VERSION}")
if(DTO_AR_EIGEN3_INCLUDE_DIR)
  target_include_directories(${TARGET_ORB_SLAM3} BEFORE PUBLIC ${DTO_AR_EIGEN3_INCLUDE_DIR})
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC Eigen3::Eigen)

# Pangolin
find_package(Pangolin REQUIRED)
message(STATUS "Pangolin_LIBRARIES: ${Pangolin_LIBRARIES}")
message(STATUS "Pangolin_INCLUDE_DIRS: ${Pangolin_INCLUDE_DIRS}")
message(STATUS "Pangolin_VERSION: ${Pangolin_VERSION}")
if(DTO_AR_PANGOLIN_ROOT)
  set(DTO_AR_PANGOLIN_INCLUDE_DIRS
    "${DTO_AR_PANGOLIN_ROOT}/include"
    "${DTO_AR_PANGOLIN_ROOT}/build/src/include"
    "${DTO_AR_PANGOLIN_ROOT}/build/external/glew/include"
    "${DTO_AR_PANGOLIN_ROOT}/build/external/libjpeg"
    "${DTO_AR_PANGOLIN_ROOT}/build/external/libpng"
    "${DTO_AR_PANGOLIN_ROOT}/build/external/zlib"
  )
  target_include_directories(${TARGET_ORB_SLAM3} SYSTEM PUBLIC ${DTO_AR_PANGOLIN_INCLUDE_DIRS})

  find_library(DTO_AR_GLEW_LIBRARY glew PATHS "${DTO_AR_PANGOLIN_ROOT}/build/external/glew/lib" NO_DEFAULT_PATH)
  find_library(DTO_AR_LIBPNG_LIBRARY libpng16_static PATHS "${DTO_AR_PANGOLIN_ROOT}/build/external/libpng/lib" NO_DEFAULT_PATH)
  find_library(DTO_AR_ZLIB_LIBRARY zlibstatic PATHS "${DTO_AR_PANGOLIN_ROOT}/build/external/zlib/lib" NO_DEFAULT_PATH)
  find_library(DTO_AR_JPEG_LIBRARY jpeg PATHS "${DTO_AR_PANGOLIN_ROOT}/build/external/libjpeg/lib" NO_DEFAULT_PATH)
  target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC
    opengl32
    glu32
    ${DTO_AR_GLEW_LIBRARY}
    ${DTO_AR_LIBPNG_LIBRARY}
    ${DTO_AR_ZLIB_LIBRARY}
    ${DTO_AR_JPEG_LIBRARY}
  )
endif()
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC ${Pangolin_LIBRARIES})

# CUDA dense refinement
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC
  CUDA::cudart
  CUDA::cublas
  CUDA::cusolver
  CUDA::cusparse
  CUDA::curand
)

# Photoneo PhoXi SDK
find_package(PhoXi REQUIRED CONFIG PATHS "${PhoXiControlPath}" NO_DEFAULT_PATH)
message(STATUS "PHOXI_INCLUDE_DIRS: ${PHOXI_INCLUDE_DIRS}")
message(STATUS "PHOXI_LIBRARY: ${PHOXI_LIBRARY}")
target_include_directories(${TARGET_ORB_SLAM3} SYSTEM PUBLIC ${PHOXI_INCLUDE_DIRS})
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC ${PHOXI_LIBRARY})
if(PHOXI_LIBRARY_CXX_ABI)
  target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC ${PHOXI_LIBRARY_CXX_ABI})
endif()
if(PHOXI_LIBRARY_SYSTEM_LIBS)
  target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC ${PHOXI_LIBRARY_SYSTEM_LIBS})
endif()

# OpenSceneGraph used by Scanner/point-cloud viewer.
find_path(OPENSCENEGRAPH_INCLUDE_DIR osg/Group
  PATHS "${OpenSceneGraph_ROOT}/include"
  NO_DEFAULT_PATH
)
foreach(osg_lib OpenThreads osg osgDB osgGA osgText osgUtil osgViewer)
  find_library(OPENSCENEGRAPH_${osg_lib}_LIBRARY
    NAMES ${osg_lib}
    PATHS "${OpenSceneGraph_ROOT}/lib"
    NO_DEFAULT_PATH
  )
  if(NOT OPENSCENEGRAPH_${osg_lib}_LIBRARY)
    message(FATAL_ERROR "OpenSceneGraph library not found: ${osg_lib}")
  endif()
  list(APPEND OPENSCENEGRAPH_LIBRARIES ${OPENSCENEGRAPH_${osg_lib}_LIBRARY})
endforeach()
if(NOT OPENSCENEGRAPH_INCLUDE_DIR)
  message(FATAL_ERROR "OpenSceneGraph include directory not found. Set OpenSceneGraph_ROOT.")
endif()
message(STATUS "OPENSCENEGRAPH_INCLUDE_DIR: ${OPENSCENEGRAPH_INCLUDE_DIR}")
target_include_directories(${TARGET_ORB_SLAM3} SYSTEM PUBLIC ${OPENSCENEGRAPH_INCLUDE_DIR})
target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC ${OPENSCENEGRAPH_LIBRARIES})

# realsense2
find_package(realsense2 QUIET)
if(realsense2_FOUND)
  message(STATUS "realsense2_VERSION: ${realsense2_VERSION}")
  target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC ${realsense2_LIBRARY}) # TODO: move these to the examples only
  set(ORB_SLAM3_DEPENDENCY_realsense2 "find_dependency(realsense2 REQUIRED)")
endif()

# Boost
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED COMPONENTS serialization)
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
target_link_libraries(${TARGET_ORB_SLAM3} PRIVATE ${Boost_LIBRARIES})

# CMake's Visual Studio CUDA language integration requires the CUDA MSBuild
# toolset to match the installed VS version. Compile the dense solver through
# nvcc explicitly so this project can be regenerated from source on this host.
set(DTO_AR_DENSE_CUDA_SOURCE
  "${PROJECT_SOURCE_DIR}/src/PCGsolver_full_optimized_joint_resmon_macro_directed_confidence.cu"
)
set(DTO_AR_DENSE_CUDA_OBJECT
  "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/dto_ar_dense/$<CONFIG>/PCGsolver_full_optimized_joint_resmon_macro_directed_confidence.cu.obj"
)
set(DTO_AR_CUDA_INCLUDE_DIRS
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/include/CameraModels"
  "${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2"
  "${PROJECT_SOURCE_DIR}/Thirdparty/g2o"
  "${PROJECT_SOURCE_DIR}/Thirdparty/g2o/g2o"
  "${PROJECT_SOURCE_DIR}/Thirdparty/g2o/g2o/core"
  "${PROJECT_SOURCE_DIR}/Thirdparty/g2o/g2o/solvers"
  "${PROJECT_SOURCE_DIR}/Thirdparty/g2o/g2o/stuff"
  "${PROJECT_SOURCE_DIR}/Thirdparty/g2o/g2o/types"
  "${PROJECT_SOURCE_DIR}/Thirdparty/Sophus"
  "${PROJECT_SOURCE_DIR}/Thirdparty/hashlibpp"
  ${DTO_AR_EIGEN3_INCLUDE_DIR}
  ${DTO_AR_BOOST_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${PHOXI_INCLUDE_DIRS}
  ${OPENSCENEGRAPH_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  "${CUDAToolkit_INCLUDE_DIRS}/crt"
)
foreach(dto_ar_include_dir IN LISTS DTO_AR_CUDA_INCLUDE_DIRS)
  if(dto_ar_include_dir)
    list(APPEND DTO_AR_CUDA_INCLUDE_FLAGS "-I${dto_ar_include_dir}")
  endif()
endforeach()
get_filename_component(DTO_AR_CXX_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
set(DTO_AR_NVCC_HOST_COMPILER_ARGS)
if(MSVC AND DTO_AR_CXX_COMPILER_DIR)
  list(APPEND DTO_AR_NVCC_HOST_COMPILER_ARGS -ccbin "${DTO_AR_CXX_COMPILER_DIR}")
endif()
add_custom_command(
  OUTPUT "${DTO_AR_DENSE_CUDA_OBJECT}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/dto_ar_dense/$<CONFIG>"
  COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
    -allow-unsupported-compiler
    --expt-relaxed-constexpr
    -std=c++11
    --machine 64
    "-gencode=arch=compute_${DTO_AR_CUDA_ARCHITECTURE},code=sm_${DTO_AR_CUDA_ARCHITECTURE}"
    "-gencode=arch=compute_${DTO_AR_CUDA_ARCHITECTURE},code=compute_${DTO_AR_CUDA_ARCHITECTURE}"
    ${DTO_AR_NVCC_HOST_COMPILER_ARGS}
    ${DTO_AR_CUDA_INCLUDE_FLAGS}
    -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
    -DEIGEN_HAS_VARIADIC_TEMPLATES=1
    -Xcompiler=/EHsc
    -Xcompiler=/bigobj
    "$<$<CONFIG:Debug>:-Xcompiler=/MTd>"
    "$<$<NOT:$<CONFIG:Debug>>:-Xcompiler=/MT>"
    -c "${DTO_AR_DENSE_CUDA_SOURCE}"
    -o "${DTO_AR_DENSE_CUDA_OBJECT}"
  DEPENDS "${DTO_AR_DENSE_CUDA_SOURCE}" "${PROJECT_SOURCE_DIR}/include/DenseOptimizer.cuh"
  COMMAND_EXPAND_LISTS
  VERBATIM
  COMMENT "Compiling DTO-AR dense CUDA solver"
)
set_source_files_properties("${DTO_AR_DENSE_CUDA_OBJECT}" PROPERTIES
  GENERATED TRUE
  EXTERNAL_OBJECT TRUE
)
target_sources(${TARGET_ORB_SLAM3} PRIVATE "${DTO_AR_DENSE_CUDA_OBJECT}")

# Backward-cpp
if(CMAKE_BUILD_TYPE MATCHES Debug)
  find_package(Backward REQUIRED)
  message(STATUS "BACKWARD_LIBRARIES: ${BACKWARD_LIBRARIES}")
  target_link_libraries(${TARGET_ORB_SLAM3} PUBLIC Backward::Backward)
endif()


# Build examples
if(ORB_SLAM3_BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif()

#Old examples
#if(ORB_SLAM3_BUILD_EXAMPLES_OLD)
#  add_subdirectory(Examples_old)
#endif()


# Install

# Set standard installation directories
set(RUNTIME_DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
set(LIBRARY_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(ARCHIVE_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(INCLUDES_DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
set(INCLUDES_INSTALL_DIR ${INCLUDES_DESTINATION}/${TARGET_ORB_SLAM3})
set(CMAKECONFIG_INSTALL_DIR ${LIBRARY_DESTINATION}/cmake/${TARGET_ORB_SLAM3})

# this relative path allows installed files to be relocatable.
file(RELATIVE_PATH REL_INCLUDE_DIR
    "${CMAKECONFIG_INSTALL_DIR}"
    "${INCLUDES_INSTALL_DIR}"
)
set(EXPORT_LIB_INC_DIRS "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}")
set(EXPORT_LIB_INC_DIRS ${EXPORT_LIB_INC_DIRS} "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}/CameraModels")

# install subprojects
if(ORB_SLAM3_USE_OWN_DBoW2)
    #install(FILES Thirdparty/DBoW2/lib/libDBoW2.so DESTINATION ${LIBRARY_DESTINATION})
    install(TARGETS DBoW2 DESTINATION ${LIBRARY_DESTINATION})
    # HACK: copy the DBoW2 headers because the own version has no package install
    install(FILES
        Thirdparty/DBoW2/DBoW2/BowVector.h
        Thirdparty/DBoW2/DBoW2/FClass.h
        Thirdparty/DBoW2/DBoW2/FeatureVector.h
        Thirdparty/DBoW2/DBoW2/FORB.h
        Thirdparty/DBoW2/DBoW2/ScoringObject.h
        Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h
        DESTINATION ${INCLUDES_DESTINATION}/DBoW2/DBoW2)
    install(FILES
        FILES Thirdparty/DBoW2/DUtils/Random.h
        FILES Thirdparty/DBoW2/DUtils/Timestamp.h
        DESTINATION ${INCLUDES_DESTINATION}/DBoW2/DUtils
    )
    install(TARGETS DBoW2 EXPORT ${TARGET_ORB_SLAM3}Targets)
    set(EXPORT_LIB_INC_DIRS ${EXPORT_LIB_INC_DIRS} "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}/../DBoW2")
endif()
if(ORB_SLAM3_USE_OWN_g2o)
    #install(FILES Thirdparty/gREQUIRED2o/lib/libg2o.so DESTINATION ${LIBRARY_DESTINATION})
    install(TARGETS g2o DESTINATION ${LIBRARY_DESTINATION})
    # HACK: copy the DBoW2 headers because the own version has no package install
    install(FILES "Thirdparty/g2o/config.h" DESTINATION ${INCLUDES_DESTINATION}/g2o)
    file(GLOB g2o_HEADERS_H "Thirdparty/g2o/g2o/core/*.h")
    file(GLOB g2o_HEADERS_HPP "Thirdparty/g2o/g2o/core/*.hpp")
    install(FILES ${g2o_HEADERS_H} ${g2o_HEADERS_HPP} DESTINATION ${INCLUDES_DESTINATION}/g2o/g2o/core)
    file(GLOB g2o_HEADERS_H "Thirdparty/g2o/g2o/solvers/*.h")
    file(GLOB g2o_HEADERS_HPP "Thirdparty/g2o/g2o/solvers/*.hpp")
    install(FILES ${g2o_HEADERS_H} ${g2o_HEADERS_HPP} DESTINATION ${INCLUDES_DESTINATION}/g2o/g2o/solvers)
    file(GLOB g2o_HEADERS_H "Thirdparty/g2o/g2o/stuff/*.h")
    file(GLOB g2o_HEADERS_HPP "Thirdparty/g2o/g2o/stuff/*.hpp")
    install(FILES ${g2o_HEADERS_H} ${g2o_HEADERS_HPP} DESTINATION ${INCLUDES_DESTINATION}/g2o/g2o/stuff)
    file(GLOB g2o_HEADERS_H "Thirdparty/g2o/g2o/types/*.h")
    file(GLOB g2o_HEADERS_HPP "Thirdparty/g2o/g2o/types/*.hpp")
    install(FILES ${g2o_HEADERS_H} ${g2o_HEADERS_HPP} DESTINATION ${INCLUDES_DESTINATION}/g2o/g2o/types)
    install(TARGETS g2o EXPORT ${TARGET_ORB_SLAM3}Targets)
    set(EXPORT_LIB_INC_DIRS ${EXPORT_LIB_INC_DIRS} "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}/../g2o")
endif()
if(ORB_SLAM3_USE_OWN_Sophus)
    install(TARGETS sophus DESTINATION ${LIBRARY_DESTINATION})
    set(EXPORT_LIB_INC_DIRS ${EXPORT_LIB_INC_DIRS} "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}/..")
endif()
if(ORB_SLAM3_USE_OWN_hashlibpp)
    install(TARGETS hashlibpp
        EXPORT ${TARGET_ORB_SLAM3}Targets
        DESTINATION ${LIBRARY_DESTINATION})
endif()

# install headers
file(GLOB ORB_SLAM3_HEADERS "include/*.h")
install(FILES ${ORB_SLAM3_HEADERS} DESTINATION ${INCLUDES_INSTALL_DIR})
file(GLOB ORB_SLAM3_HEADERS "include/CameraModels/*.h")
install(FILES ${ORB_SLAM3_HEADERS} DESTINATION ${INCLUDES_INSTALL_DIR}/CameraModels)

# install libraries
install(TARGETS ${TARGET_ORB_SLAM3}
    EXPORT ${TARGET_ORB_SLAM3}Targets
    RUNTIME DESTINATION ${RUNTIME_DESTINATION}
    LIBRARY DESTINATION ${LIBRARY_DESTINATION}
    ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
    INCLUDES DESTINATION ${INCLUDES_DESTINATION}
)

# write package config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${TARGET_ORB_SLAM3}ConfigVersion.cmake"
    VERSION 1.0
    COMPATIBILITY AnyNewerVersion
)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/ORB_SLAM3Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${TARGET_ORB_SLAM3}Config.cmake"
    INSTALL_DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)
install(EXPORT ${TARGET_ORB_SLAM3}Targets
    DESTINATION ${CMAKECONFIG_INSTALL_DIR})
install(FILES "${PROJECT_BINARY_DIR}/${TARGET_ORB_SLAM3}ConfigVersion.cmake"
    "${PROJECT_BINARY_DIR}/${TARGET_ORB_SLAM3}Config.cmake"
    DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)


# TODO: extract and copy the vocabulary into the bin folder
