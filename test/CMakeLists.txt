# project setup
cmake_minimum_required(VERSION 3.24)
project("DeVi:Tests")
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      ${CMAKE_SOURCE_DIR}/..
      CACHE PATH "Installation path prefix" FORCE)
endif()

# adds and configures tests as specified
function(build_test test_name src_file)
  add_executable(${test_name} ${src_file})
  target_include_directories(${test_name} PRIVATE ../include)
  target_compile_options(${test_name} PRIVATE -Wall)

  install(TARGETS ${test_name} RUNTIME DESTINATION bin)
endfunction()

# 1) devi::core::shape
build_test(test_shape core/shape.cc)
# 2) devi::core::array
build_test(test_array core/array.cc)
# 3) devi::core::index
build_test(test_index core/index.cc)

# compile commands
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  install(FILES ${CMAKE_BINARY_DIR}/compile_commands.json DESTINATION ../.nvim)
endif()
