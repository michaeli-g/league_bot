#include "SharedMemory.h"

SharedMemory::SharedMemory(const std::string& name, size_t size)
    : m_name(name), m_size(size), m_hMapFile(NULL), m_ptr(NULL) {}

SharedMemory::~SharedMemory() {
    if (m_ptr) UnmapViewOfFile(m_ptr);
    if (m_hMapFile) CloseHandle(m_hMapFile);
}

bool SharedMemory::create() {
    m_hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        (DWORD)m_size,
        m_name.c_str()
    );

    if (m_hMapFile == NULL) {
        std::cerr << "Could not create file mapping object (" << GetLastError() << ")." << std::endl;
        return false;
    }

    m_ptr = MapViewOfFile(
        m_hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        m_size
    );

    if (m_ptr == NULL) {
        std::cerr << "Could not map view of file (" << GetLastError() << ")." << std::endl;
        CloseHandle(m_hMapFile);
        return false;
    }

    return true;
}
