cmake_minimum_required(VERSION 3.10)
project(MothershipCapturer)

set(CMAKE_CXX_STANDARD 17)

# Include source files
set(SOURCES
    src/main.cpp
    src/DXGICapturer.cpp
    src/SharedMemory.cpp
)

add_executable(MothershipCapturer ${SOURCES})

# Link Windows libraries
target_link_libraries(MothershipCapturer d3d11 dxgi)

# Set output directory to bin/
set_target_properties(MothershipCapturer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
