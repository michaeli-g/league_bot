#include <iostream>
#include <chrono>
#include <thread>
#include "DXGICapturer.h"
#include "SharedMemory.h"

int main() {
    std::cout << "--- Mothership Frame Capturer ---" << std::endl;

    DXGICapturer capturer;
    if (!capturer.init()) {
        std::cerr << "Failed to initialize DXGI capturer." << std::endl;
        return -1;
    }

    // Allocate 64MB for shared memory (3840x2160x4 is ~33MB)
    SharedMemory shm("Global\\LoLBotFrame", 1024 * 1024 * 64);
    if (!shm.create()) {
        std::cerr << "Failed to create shared memory." << std::endl;
        return -1;
    }

    std::cout << "Capturer started. Writing to shared memory 'Global\\LoLBotFrame'..." << std::endl;

    int width = 0, height = 0;
    auto last_time = std::chrono::high_resolution_clock::now();
    int frames = 0;

    while (true) {
        if (capturer.capture_frame(shm.get_ptr(), width, height)) {
            frames++;
            
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = now - last_time;

            if (elapsed.count() >= 1.0) {
                std::cout << "FPS: " << frames << " (" << width << "x" << height << ")" << std::endl;
                frames = 0;
                last_time = now;
            }
        }

        // Small sleep to prevent 100% CPU usage if capture is too fast
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    return 0;
}
