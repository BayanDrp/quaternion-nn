cmake_minimum_required(VERSION 3.14)

project(quaternion-nn VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE QNN_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(qnn STATIC ${QNN_SOURCES})

target_include_directories(qnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(qnn PRIVATE -Wall -Wextra -Werror)

find_package(ZLIB REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(qnn PUBLIC OpenMP::OpenMP_CXX)

enable_testing()

function(qnn_add_executable target source)
    file(SIZE "${CMAKE_CURRENT_SOURCE_DIR}/${source}" _size)
    if(_size GREATER 0)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE qnn)
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()
endfunction()

file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS examples/*.cpp)
foreach(_src ${EXAMPLE_SOURCES})
    get_filename_component(_name ${_src} NAME_WE)
    qnn_add_executable(example_${_name} examples/${_name}.cpp)
endforeach()

file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS benchmarks/*.cpp)
foreach(_src ${BENCH_SOURCES})
    get_filename_component(_name ${_src} NAME_WE)
    qnn_add_executable(bench_${_name} benchmarks/${_name}.cpp)
endforeach()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*/*.cpp)
foreach(_src ${TEST_SOURCES})
    get_filename_component(_name ${_src} NAME_WE)
    get_filename_component(_dir ${_src} DIRECTORY)
    get_filename_component(_group ${_dir} NAME)
    qnn_add_executable(test_${_group}_${_name} tests/${_group}/${_name}.cpp)
    if(TARGET test_${_group}_${_name})
        add_test(NAME test_${_group}_${_name} COMMAND test_${_group}_${_name}
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
endforeach()

if(TARGET example_mnist)
    target_link_libraries(example_mnist PRIVATE ZLIB::ZLIB)
endif()
if(TARGET example_mnist_with_not_real_target)
    target_link_libraries(example_mnist_with_not_real_target PRIVATE ZLIB::ZLIB)
endif()
if(TARGET test_data_test_mnist)
    target_link_libraries(test_data_test_mnist PRIVATE ZLIB::ZLIB)
endif()
