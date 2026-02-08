#pragma once
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <vector>

class DXGICapturer {
public:
    DXGICapturer();
    ~DXGICapturer();

    bool init();
    bool capture_frame(void* dest_buffer, int& width, int& height);

private:
    ID3D11Device* m_device;
    ID3D11DeviceContext* m_context;
    IDXGIOutputDuplication* m_desk_dupl;
    DXGI_OUTPUT_DESC m_output_desc;

    bool setup_duplication();
};
