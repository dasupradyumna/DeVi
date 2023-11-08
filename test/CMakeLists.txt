# project setup
cmake_minimum_required(VERSION 3.24)
project("DeVi:Tests")
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      ${CMAKE_SOURCE_DIR}/..
      CACHE PATH "Installation path prefix" FORCE)
endif()

# 1) devi::core::shape
add_executable(test_shape core/shape.cc)
target_include_directories(test_shape PRIVATE ../include)
target_compile_options(test_shape PRIVATE -Wall)
install(TARGETS test_shape RUNTIME DESTINATION bin)

# compile commands
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  install(FILES ${CMAKE_BINARY_DIR}/compile_commands.json DESTINATION ../.nvim)
endif()
