#include "DXGICapturer.h"
#include <iostream>

DXGICapturer::DXGICapturer() 
    : m_device(NULL), m_context(NULL), m_desk_dupl(NULL) {}

DXGICapturer::~DXGICapturer() {
    if (m_desk_dupl) m_desk_dupl->Release();
    if (m_context) m_context->Release();
    if (m_device) m_device->Release();
}

bool DXGICapturer::init() {
    D3D_FEATURE_LEVEL feature_level;
    HRESULT hr = D3D11CreateDevice(
        NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 0, NULL, 0,
        D3D11_SDK_VERSION, &m_device, &feature_level, &m_context
    );

    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device: " << std::hex << hr << std::endl;
        return false;
    }

    return setup_duplication();
}

bool DXGICapturer::setup_duplication() {
    IDXGIDevice* dxgi_device = NULL;
    m_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device);

    IDXGIAdapter* dxgi_adapter = NULL;
    dxgi_device->GetParent(__uuidof(IDXGIAdapter), (void**)&dxgi_adapter);
    dxgi_device->Release();

    IDXGIOutput* dxgi_output = NULL;
    dxgi_adapter->GetEnumOutputs(0, &dxgi_output);
    dxgi_adapter->Release();

    IDXGIOutput1* dxgi_output1 = NULL;
    dxgi_output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&dxgi_output1);
    dxgi_output->GetDesc(&m_output_desc);
    dxgi_output->Release();

    HRESULT hr = dxgi_output1->DuplicateOutput(m_device, &m_desk_dupl);
    dxgi_output1->Release();

    if (FAILED(hr)) {
        std::cerr << "Failed to duplicate output: " << std::hex << hr << std::endl;
        return false;
    }

    return true;
}

bool DXGICapturer::capture_frame(void* dest_buffer, int& width, int& height) {
    IDXGIResource* desktop_res = NULL;
    DXGI_OUTDUPL_FRAME_INFO frame_info;
    
    HRESULT hr = m_desk_dupl->AcquireNextFrame(100, &frame_info, &desktop_res);
    
    if (hr == DXGI_ERROR_ACCESS_LOST) {
        m_desk_dupl->Release();
        m_desk_dupl = NULL;
        return setup_duplication();
    }
    
    if (FAILED(hr)) return false;

    ID3D11Texture2D* gpu_tex = NULL;
    desktop_res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&gpu_tex);
    desktop_res->Release();

    D3D11_TEXTURE2D_DESC desc;
    gpu_tex->GetDesc(&desc);
    width = desc.Width;
    height = desc.Height;

    // Create a staging texture to copy GPU data to CPU
    D3D11_TEXTURE2D_DESC staging_desc = desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;

    ID3D11Texture2D* staging_tex = NULL;
    m_device->CreateTexture2D(&staging_desc, NULL, &staging_tex);
    m_context->CopyResource(staging_tex, gpu_tex);

    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = m_context->Map(staging_tex, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        // Copy the data to our destination buffer (shared memory)
        memcpy(dest_buffer, mapped.pData, width * height * 4); // RGBA
        m_context->Unmap(staging_tex, 0);
    }

    staging_tex->Release();
    gpu_tex->Release();
    m_desk_dupl->ReleaseFrame();

    return SUCCEEDED(hr);
}
