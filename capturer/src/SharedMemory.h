#pragma once
#include <windows.h>
#include <string>
#include <iostream>

class SharedMemory {
public:
    SharedMemory(const std::string& name, size_t size);
    ~SharedMemory();

    bool create();
    void* get_ptr() const { return m_ptr; }
    size_t get_size() const { return m_size; }

private:
    std::string m_name;
    size_t m_size;
    HANDLE m_hMapFile;
    void* m_ptr;
};
