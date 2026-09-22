cmake_minimum_required(VERSION 3.9)
set(HASHLIBPP_TARGET "hashlibpp")
project(${HASHLIBPP_TARGET})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB SRCS *.cpp)
file(GLOB HDRS *.h)

add_library(${HASHLIBPP_TARGET} STATIC ${HDRS} ${SRCS})
target_include_directories(${HASHLIBPP_TARGET} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  $<INSTALL_INTERFACE:include/hashlibpp>
)
