#define GLFW_INCLUDE_NONE
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

static constexpr uint32_t WIDTH = 800;
static constexpr uint32_t HEIGHT = 600;
static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

static const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
static const std::vector<const char*> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
static constexpr bool kEnableValidation = false;
#else
static constexpr bool kEnableValidation = true;
#endif

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Mat4 {
    std::array<float, 16> values{};
};

struct PushConstants {
    Mat4 modelViewProj{};
};

struct Vertex {
    Vec3 position{};
    Vec3 color{};
};

static const std::array<Vertex, 36> kCubeVertices = { {
    { { -1.0f, -1.0f,  1.0f }, { 0.95f, 0.32f, 0.25f } }, { {  1.0f, -1.0f,  1.0f }, { 0.95f, 0.32f, 0.25f } }, { {  1.0f,  1.0f,  1.0f }, { 0.95f, 0.32f, 0.25f } },
    { { -1.0f, -1.0f,  1.0f }, { 0.95f, 0.32f, 0.25f } }, { {  1.0f,  1.0f,  1.0f }, { 0.95f, 0.32f, 0.25f } }, { { -1.0f,  1.0f,  1.0f }, { 0.95f, 0.32f, 0.25f } },
    { {  1.0f, -1.0f, -1.0f }, { 0.22f, 0.58f, 0.95f } }, { { -1.0f, -1.0f, -1.0f }, { 0.22f, 0.58f, 0.95f } }, { { -1.0f,  1.0f, -1.0f }, { 0.22f, 0.58f, 0.95f } },
    { {  1.0f, -1.0f, -1.0f }, { 0.22f, 0.58f, 0.95f } }, { { -1.0f,  1.0f, -1.0f }, { 0.22f, 0.58f, 0.95f } }, { {  1.0f,  1.0f, -1.0f }, { 0.22f, 0.58f, 0.95f } },
    { { -1.0f, -1.0f, -1.0f }, { 0.20f, 0.78f, 0.42f } }, { { -1.0f, -1.0f,  1.0f }, { 0.20f, 0.78f, 0.42f } }, { { -1.0f,  1.0f,  1.0f }, { 0.20f, 0.78f, 0.42f } },
    { { -1.0f, -1.0f, -1.0f }, { 0.20f, 0.78f, 0.42f } }, { { -1.0f,  1.0f,  1.0f }, { 0.20f, 0.78f, 0.42f } }, { { -1.0f,  1.0f, -1.0f }, { 0.20f, 0.78f, 0.42f } },
    { {  1.0f, -1.0f,  1.0f }, { 0.96f, 0.74f, 0.18f } }, { {  1.0f, -1.0f, -1.0f }, { 0.96f, 0.74f, 0.18f } }, { {  1.0f,  1.0f, -1.0f }, { 0.96f, 0.74f, 0.18f } },
    { {  1.0f, -1.0f,  1.0f }, { 0.96f, 0.74f, 0.18f } }, { {  1.0f,  1.0f, -1.0f }, { 0.96f, 0.74f, 0.18f } }, { {  1.0f,  1.0f,  1.0f }, { 0.96f, 0.74f, 0.18f } },
    { { -1.0f,  1.0f,  1.0f }, { 0.71f, 0.37f, 0.96f } }, { {  1.0f,  1.0f,  1.0f }, { 0.71f, 0.37f, 0.96f } }, { {  1.0f,  1.0f, -1.0f }, { 0.71f, 0.37f, 0.96f } },
    { { -1.0f,  1.0f,  1.0f }, { 0.71f, 0.37f, 0.96f } }, { {  1.0f,  1.0f, -1.0f }, { 0.71f, 0.37f, 0.96f } }, { { -1.0f,  1.0f, -1.0f }, { 0.71f, 0.37f, 0.96f } },
    { { -1.0f, -1.0f, -1.0f }, { 0.14f, 0.82f, 0.85f } }, { {  1.0f, -1.0f, -1.0f }, { 0.14f, 0.82f, 0.85f } }, { {  1.0f, -1.0f,  1.0f }, { 0.14f, 0.82f, 0.85f } },
    { { -1.0f, -1.0f, -1.0f }, { 0.14f, 0.82f, 0.85f } }, { {  1.0f, -1.0f,  1.0f }, { 0.14f, 0.82f, 0.85f } }, { { -1.0f, -1.0f,  1.0f }, { 0.14f, 0.82f, 0.85f } }
} };

static Vec3 operator+(const Vec3& lhs, const Vec3& rhs)
{
    return { lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
}

static Vec3 operator-(const Vec3& lhs, const Vec3& rhs)
{
    return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
}

static Vec3 operator*(const Vec3& value, float scalar)
{
    return { value.x * scalar, value.y * scalar, value.z * scalar };
}

static Vec3 operator/(const Vec3& value, float scalar)
{
    return { value.x / scalar, value.y / scalar, value.z / scalar };
}

static float dot(const Vec3& lhs, const Vec3& rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

static Vec3 cross(const Vec3& lhs, const Vec3& rhs)
{
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x
    };
}

static float length(const Vec3& value)
{
    return std::sqrt(dot(value, value));
}

static Vec3 normalize(const Vec3& value)
{
    const float len = length(value);
    if (len <= 1e-5f) {
        return {};
    }
    return value * (1.0f / len);
}

static Mat4 identityMatrix()
{
    Mat4 result{};
    result.values[0] = 1.0f;
    result.values[5] = 1.0f;
    result.values[10] = 1.0f;
    result.values[15] = 1.0f;
    return result;
}

static Mat4 multiply(const Mat4& lhs, const Mat4& rhs)
{
    Mat4 result{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += lhs.values[k * 4 + row] * rhs.values[column * 4 + k];
            }
            result.values[column * 4 + row] = sum;
        }
    }
    return result;
}

static Mat4 perspectiveMatrix(float fovRadians, float aspect, float nearPlane, float farPlane)
{
    const float tanHalfFov = std::tan(fovRadians * 0.5f);

    Mat4 result{};
    result.values[0] = 1.0f / (aspect * tanHalfFov);
    result.values[5] = -1.0f / tanHalfFov;
    result.values[10] = farPlane / (nearPlane - farPlane);
    result.values[11] = -1.0f;
    result.values[14] = -(farPlane * nearPlane) / (farPlane - nearPlane);
    return result;
}

static Mat4 lookAtMatrix(const Vec3& eye, const Vec3& center, const Vec3& up)
{
    const Vec3 forward = normalize(center - eye);
    const Vec3 right = normalize(cross(forward, up));
    const Vec3 upVector = cross(right, forward);

    Mat4 result = identityMatrix();
    result.values[0] = right.x;
    result.values[1] = right.y;
    result.values[2] = right.z;
    result.values[4] = upVector.x;
    result.values[5] = upVector.y;
    result.values[6] = upVector.z;
    result.values[8] = -forward.x;
    result.values[9] = -forward.y;
    result.values[10] = -forward.z;
    result.values[12] = -dot(right, eye);
    result.values[13] = -dot(upVector, eye);
    result.values[14] = dot(forward, eye);
    result.values[15] = 1.0f;
    return result;
}

static std::vector<char> readFile(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan] " << data->pMessage << '\n';
    }
    return VK_FALSE;
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;

    bool isComplete() const
    {
        return graphics.has_value() && present.has_value();
    }
};

struct SwapChainSupport {
    vk::SurfaceCapabilitiesKHR capabilities{};
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

class VulkanApp
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* m_window = nullptr;
    bool m_framebufferResized = false;
    Vec3 m_cameraPosition{ 3.0f, 1.8f, 3.0f };
    float m_cameraYaw = -135.0f;
    float m_cameraPitch = -20.0f;
    bool m_isLeftMouseDragging = false;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
    std::chrono::steady_clock::time_point m_lastFrameTime{};

    vk::raii::Context m_context{};
    vk::raii::Instance m_instance{ nullptr };
    vk::raii::DebugUtilsMessengerEXT m_debugMessenger{ nullptr };
    vk::raii::SurfaceKHR m_surface{ nullptr };
    vk::raii::PhysicalDevice m_physDevice{ nullptr };
    vk::raii::Device m_device{ nullptr };
    vk::Queue m_graphicsQueue{};
    vk::Queue m_presentQueue{};

    vk::raii::SwapchainKHR m_swapChain{ nullptr };
    vk::Format m_scFormat{};
    vk::Extent2D m_scExtent{};
    std::vector<vk::Image> m_scImages;
    std::vector<vk::raii::ImageView> m_scImageViews;

    vk::raii::RenderPass m_renderPass{ nullptr };
    vk::raii::PipelineLayout m_pipelineLayout{ nullptr };
    vk::raii::Pipeline m_graphicsPipeline{ nullptr };
    vk::raii::Buffer m_vertexBuffer{ nullptr };
    vk::raii::DeviceMemory m_vertexBufferMemory{ nullptr };

    std::vector<vk::raii::Framebuffer> m_framebuffers;
    vk::raii::CommandPool m_commandPool{ nullptr };
    vk::raii::CommandBuffers m_commandBuffers{ nullptr };

    std::vector<vk::raii::Semaphore> m_imageAvailableSems;
    std::vector<vk::raii::Semaphore> m_renderFinishedSems;
    std::vector<vk::raii::Fence> m_inFlightFences;
    std::vector<vk::Fence> m_imagesInFlight;
    uint32_t m_currentFrame = 0;

    void initWindow()
    {
        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        m_window = glfwCreateWindow(WIDTH, HEIGHT, "VulkanRenderer - Cube Camera (WASD)", nullptr, nullptr);
        if (!m_window) {
            throw std::runtime_error("Failed to create GLFW window");
        }

        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* window, int, int) {
            reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(window))->m_framebufferResized = true;
        });
        glfwSetMouseButtonCallback(m_window, [](GLFWwindow* window, int button, int action, int) {
            auto* app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(window));
            if (button != GLFW_MOUSE_BUTTON_LEFT) {
                return;
            }

            if (action == GLFW_PRESS) {
                app->m_isLeftMouseDragging = true;
                glfwGetCursorPos(window, &app->m_lastMouseX, &app->m_lastMouseY);
            } else if (action == GLFW_RELEASE) {
                app->m_isLeftMouseDragging = false;
            }
        });
        glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, double xpos, double ypos) {
            auto* app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(window));
            if (!app->m_isLeftMouseDragging) {
                app->m_lastMouseX = xpos;
                app->m_lastMouseY = ypos;
                return;
            }

            const double deltaX = xpos - app->m_lastMouseX;
            const double deltaY = ypos - app->m_lastMouseY;
            app->m_lastMouseX = xpos;
            app->m_lastMouseY = ypos;

            const float sensitivity = 0.18f;
            app->m_cameraYaw += static_cast<float>(deltaX) * sensitivity;
            app->m_cameraPitch -= static_cast<float>(deltaY) * sensitivity;
            app->m_cameraPitch = std::clamp(app->m_cameraPitch, -89.0f, 89.0f);
        });
    }

    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createGraphicsPipeline();
        createCommandBuffers();
        createSyncObjects();
    }

    void createInstance()
    {
        if (kEnableValidation && !checkValidationLayers()) {
            throw std::runtime_error("Validation layers not available");
        }

        vk::ApplicationInfo appInfo{};
        appInfo.pApplicationName = "VulkanRenderer";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "None";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        auto extensions = requiredExtensions();

        vk::InstanceCreateInfo createInfo{};
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (kEnableValidation) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
            createInfo.ppEnabledLayerNames = kValidationLayers.data();
            fillDebugMessengerCI(debugCreateInfo);
            createInfo.pNext = &debugCreateInfo;
        }

        m_instance = vk::raii::Instance(m_context, createInfo);
    }

    bool checkValidationLayers()
    {
        auto available = m_context.enumerateInstanceLayerProperties();
        for (const char* name : kValidationLayers) {
            bool found = false;
            for (const auto& prop : available) {
                if (std::strcmp(name, prop.layerName) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }

    std::vector<const char*> requiredExtensions()
    {
        uint32_t glfwCount = 0;
        const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
        std::vector<const char*> extensions(glfwExts, glfwExts + glfwCount);
        if (kEnableValidation) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    void fillDebugMessengerCI(vk::DebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo.messageSeverity =
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        createInfo.messageType =
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
        createInfo.pfnUserCallback = reinterpret_cast<vk::PFN_DebugUtilsMessengerCallbackEXT>(debugCallback);
    }

    void setupDebugMessenger()
    {
        if (!kEnableValidation) {
            return;
        }

        vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
        fillDebugMessengerCI(createInfo);
        m_debugMessenger = vk::raii::DebugUtilsMessengerEXT(m_instance, createInfo);
    }

    void createSurface()
    {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (glfwCreateWindowSurface(static_cast<VkInstance>(*m_instance), m_window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        }

        m_surface = vk::raii::SurfaceKHR(m_instance, surface);
    }

    void pickPhysicalDevice()
    {
        auto devices = m_instance.enumeratePhysicalDevices();
        if (devices.empty()) {
            throw std::runtime_error("No Vulkan-capable GPU found");
        }

        bool found = false;
        for (auto& device : devices) {
            if (isSuitable(device)) {
                m_physDevice = std::move(device);
                found = true;
                break;
            }
        }

        if (!found) {
            throw std::runtime_error("No suitable GPU found");
        }

        auto props = m_physDevice.getProperties();
        std::cout << "GPU: " << props.deviceName << '\n';
    }

    bool isSuitable(const vk::raii::PhysicalDevice& device)
    {
        if (!findQueueFamilies(device).isComplete() || !checkDeviceExtensions(device)) {
            return false;
        }

        auto swapChainSupport = querySwapChainSupport(device);
        return !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    bool checkDeviceExtensions(const vk::raii::PhysicalDevice& device)
    {
        auto available = device.enumerateDeviceExtensionProperties();
        std::set<std::string> required(kDeviceExtensions.begin(), kDeviceExtensions.end());
        for (const auto& extension : available) {
            required.erase(extension.extensionName);
        }
        return required.empty();
    }

    QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& device)
    {
        QueueFamilyIndices indices;
        auto families = device.getQueueFamilyProperties();

        for (uint32_t i = 0; i < static_cast<uint32_t>(families.size()); ++i) {
            if (families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphics = i;
            }
            if (device.getSurfaceSupportKHR(i, *m_surface)) {
                indices.present = i;
            }
            if (indices.isComplete()) {
                break;
            }
        }

        return indices;
    }

    SwapChainSupport querySwapChainSupport(const vk::raii::PhysicalDevice& device)
    {
        SwapChainSupport support;
        support.capabilities = device.getSurfaceCapabilitiesKHR(*m_surface);
        support.formats = device.getSurfaceFormatsKHR(*m_surface);
        support.presentModes = device.getSurfacePresentModesKHR(*m_surface);
        return support;
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(m_physDevice);
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphics.value(), indices.present.value() };

        float priority = 1.0f;
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &priority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        auto availableFeatures = m_physDevice.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features>();
        if (!availableFeatures.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters) {
            throw std::runtime_error("GPU does not support Vulkan 1.1 shaderDrawParameters");
        }

        vk::PhysicalDeviceFeatures2 features{};
        vk::PhysicalDeviceVulkan11Features vulkan11Features{};
        vulkan11Features.shaderDrawParameters = VK_TRUE;
        features.pNext = &vulkan11Features;

        vk::DeviceCreateInfo createInfo{};
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pNext = &features;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
        createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();
        if (kEnableValidation) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
            createInfo.ppEnabledLayerNames = kValidationLayers.data();
        }

        m_device = vk::raii::Device(m_physDevice, createInfo);
        m_graphicsQueue = m_device.getQueue(indices.graphics.value(), 0);
        m_presentQueue = m_device.getQueue(indices.present.value(), 0);
    }

    vk::SurfaceFormatKHR chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats)
    {
        for (const auto& format : formats) {
            if (format.format == vk::Format::eB8G8R8A8Srgb &&
                format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return format;
            }
        }
        return formats[0];
    }

    vk::PresentModeKHR choosePresentMode(const std::vector<vk::PresentModeKHR>& modes)
    {
        for (auto mode : modes) {
            if (mode == vk::PresentModeKHR::eMailbox) {
                return mode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR& caps)
    {
        if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return caps.currentExtent;
        }

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);

        vk::Extent2D extent{
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };
        extent.width = std::clamp(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
        return extent;
    }

    void createSwapChain()
    {
        SwapChainSupport support = querySwapChainSupport(m_physDevice);
        vk::SurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(support.formats);
        vk::PresentModeKHR presentMode = choosePresentMode(support.presentModes);
        vk::Extent2D extent = chooseExtent(support.capabilities);

        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0) {
            imageCount = std::min(imageCount, support.capabilities.maxImageCount);
        }

        vk::SwapchainCreateInfoKHR createInfo{};
        createInfo.surface = *m_surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

        QueueFamilyIndices indices = findQueueFamilies(m_physDevice);
        uint32_t queueFamilies[] = { indices.graphics.value(), indices.present.value() };
        if (indices.graphics != indices.present) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilies;
        } else {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        }

        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        if (m_swapChain != nullptr) {
            createInfo.oldSwapchain = *m_swapChain;
        }

        m_swapChain = vk::raii::SwapchainKHR(m_device, createInfo);
        m_scImages = m_swapChain.getImages();
        m_scFormat = surfaceFormat.format;
        m_scExtent = extent;
    }

    void createImageViews()
    {
        m_scImageViews.clear();
        m_scImageViews.reserve(m_scImages.size());

        for (auto image : m_scImages) {
            vk::ImageViewCreateInfo createInfo{};
            createInfo.image = image;
            createInfo.viewType = vk::ImageViewType::e2D;
            createInfo.format = m_scFormat;
            createInfo.components = {
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity
            };
            createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            m_scImageViews.emplace_back(m_device, createInfo);
        }
    }

    void createRenderPass()
    {
        vk::AttachmentDescription colorAttachment{};
        colorAttachment.format = m_scFormat;
        colorAttachment.samples = vk::SampleCountFlagBits::e1;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

        vk::AttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::SubpassDescription subpass{};
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorRef;

        vk::SubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

        vk::RenderPassCreateInfo createInfo{};
        createInfo.attachmentCount = 1;
        createInfo.pAttachments = &colorAttachment;
        createInfo.subpassCount = 1;
        createInfo.pSubpasses = &subpass;
        createInfo.dependencyCount = 1;
        createInfo.pDependencies = &dependency;

        m_renderPass = vk::raii::RenderPass(m_device, createInfo);
    }

    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code)
    {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        return vk::raii::ShaderModule(m_device, createInfo);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const
    {
        const vk::PhysicalDeviceMemoryProperties memoryProperties = m_physDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            const bool supported = (typeFilter & (1u << i)) != 0;
            const bool matches = (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties;
            if (supported && matches) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }

    void createVertexBuffer()
    {
        const vk::DeviceSize bufferSize = sizeof(kCubeVertices);

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = bufferSize;
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer;
        bufferInfo.sharingMode = vk::SharingMode::eExclusive;
        m_vertexBuffer = vk::raii::Buffer(m_device, bufferInfo);

        const vk::MemoryRequirements memRequirements = m_vertexBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(
            memRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        m_vertexBufferMemory = vk::raii::DeviceMemory(m_device, allocInfo);
        m_vertexBuffer.bindMemory(*m_vertexBufferMemory, 0);

        void* mappedMemory = m_vertexBufferMemory.mapMemory(0, bufferSize);
        std::memcpy(mappedMemory, kCubeVertices.data(), static_cast<size_t>(bufferSize));
        m_vertexBufferMemory.unmapMemory();
    }

    void createGraphicsPipeline()
    {
        auto vertCode = readFile("shaders/triangle.vert.spv");
        auto fragCode = readFile("shaders/triangle.frag.spv");

        vk::raii::ShaderModule vertModule = createShaderModule(vertCode);
        vk::raii::ShaderModule fragModule = createShaderModule(fragCode);

        vk::PipelineShaderStageCreateInfo vertStage{};
        vertStage.stage = vk::ShaderStageFlagBits::eVertex;
        vertStage.module = *vertModule;
        vertStage.pName = "main";

        vk::PipelineShaderStageCreateInfo fragStage{};
        fragStage.stage = vk::ShaderStageFlagBits::eFragment;
        fragStage.module = *fragModule;
        fragStage.pName = "main";

        vk::PipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;

        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = static_cast<uint32_t>(offsetof(Vertex, position));
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = static_cast<uint32_t>(offsetof(Vertex, color));

        vk::PipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &bindingDescription;
        vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInput.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

        vk::PipelineViewportStateCreateInfo viewportState{};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
        rasterizer.lineWidth = 1.0f;

        vk::PipelineMultisampleStateCreateInfo multisampling{};
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask =
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA;

        vk::PipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments = &blendAttachment;

        vk::DynamicState dynamicStates[] = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(std::size(dynamicStates));
        dynamicState.pDynamicStates = dynamicStates;

        vk::PushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eVertex;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstants);

        vk::PipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        m_pipelineLayout = vk::raii::PipelineLayout(m_device, layoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.stageCount = static_cast<uint32_t>(std::size(stages));
        pipelineInfo.pStages = stages;
        pipelineInfo.pVertexInputState = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlend;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = *m_pipelineLayout;
        pipelineInfo.renderPass = *m_renderPass;
        pipelineInfo.subpass = 0;

        m_graphicsPipeline = vk::raii::Pipeline(m_device, nullptr, pipelineInfo);
    }

    PushConstants buildCameraData() const
    {
        constexpr float pi = 3.1415926535f;
        const float aspect = static_cast<float>(m_scExtent.width) / static_cast<float>(m_scExtent.height);

        PushConstants pushConstants{};
        const Mat4 model = identityMatrix();
        const Mat4 view = lookAtMatrix(m_cameraPosition, m_cameraPosition + cameraForward(), Vec3{ 0.0f, 1.0f, 0.0f });
        const Mat4 projection = perspectiveMatrix(45.0f * pi / 180.0f, aspect, 0.1f, 100.0f);
        pushConstants.modelViewProj = multiply(projection, multiply(view, model));
        return pushConstants;
    }

    Vec3 cameraForward() const
    {
        constexpr float pi = 3.1415926535f;
        const float yawRadians = m_cameraYaw * pi / 180.0f;
        const float pitchRadians = m_cameraPitch * pi / 180.0f;

        return normalize(Vec3{
            std::cos(yawRadians) * std::cos(pitchRadians),
            std::sin(pitchRadians),
            std::sin(yawRadians) * std::cos(pitchRadians)
        });
    }

    void createFramebuffers()
    {
        m_framebuffers.clear();
        m_framebuffers.reserve(m_scImageViews.size());

        for (auto& imageView : m_scImageViews) {
            vk::ImageView attachment = *imageView;

            vk::FramebufferCreateInfo createInfo{};
            createInfo.renderPass = *m_renderPass;
            createInfo.attachmentCount = 1;
            createInfo.pAttachments = &attachment;
            createInfo.width = m_scExtent.width;
            createInfo.height = m_scExtent.height;
            createInfo.layers = 1;

            m_framebuffers.emplace_back(m_device, createInfo);
        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices indices = findQueueFamilies(m_physDevice);
        vk::CommandPoolCreateInfo createInfo{};
        createInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        createInfo.queueFamilyIndex = indices.graphics.value();
        m_commandPool = vk::raii::CommandPool(m_device, createInfo);
    }

    void createCommandBuffers()
    {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = *m_commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        m_commandBuffers = vk::raii::CommandBuffers(m_device, allocInfo);
    }

    void recordCommandBuffer(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
    {
        vk::CommandBufferBeginInfo beginInfo{};
        commandBuffer.begin(beginInfo);

        vk::ClearValue clearColor = vk::ClearColorValue(std::array<float, 4>{ 0.05f, 0.05f, 0.05f, 1.0f });

        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.renderPass = *m_renderPass;
        renderPassInfo.framebuffer = *m_framebuffers[imageIndex];
        renderPassInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
        renderPassInfo.renderArea.extent = m_scExtent;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_graphicsPipeline);

        vk::Viewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(m_scExtent.width);
        viewport.height = static_cast<float>(m_scExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        commandBuffer.setViewport(0, viewport);

        vk::Rect2D scissor{ { 0, 0 }, m_scExtent };
        commandBuffer.setScissor(0, scissor);

        const vk::Buffer vertexBuffers[] = { *m_vertexBuffer };
        const vk::DeviceSize offsets[] = { 0 };
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);

        const PushConstants pushConstants = buildCameraData();
        commandBuffer.pushConstants<PushConstants>(*m_pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pushConstants);
        commandBuffer.draw(static_cast<uint32_t>(kCubeVertices.size()), 1, 0, 0);
        commandBuffer.endRenderPass();
        commandBuffer.end();
    }

    void createSyncObjects()
    {
        m_imageAvailableSems.clear();
        m_inFlightFences.clear();
        m_imageAvailableSems.reserve(MAX_FRAMES_IN_FLIGHT);
        m_inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);

        vk::SemaphoreCreateInfo semInfo{};
        vk::FenceCreateInfo fenceInfo{};
        fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            m_imageAvailableSems.emplace_back(m_device, semInfo);
            m_inFlightFences.emplace_back(m_device, fenceInfo);
        }

        createPresentSyncObjects();
    }

    void createPresentSyncObjects()
    {
        m_renderFinishedSems.clear();
        m_imagesInFlight.clear();
        m_renderFinishedSems.reserve(m_scImages.size());
        m_imagesInFlight.resize(m_scImages.size(), VK_NULL_HANDLE);

        vk::SemaphoreCreateInfo semInfo{};
        for (size_t i = 0; i < m_scImages.size(); ++i) {
            m_renderFinishedSems.emplace_back(m_device, semInfo);
        }
    }

    void cleanupSwapChain()
    {
        m_framebuffers.clear();
        m_scImageViews.clear();
        m_scImages.clear();
        m_renderFinishedSems.clear();
        m_imagesInFlight.clear();
        m_swapChain = nullptr;
    }

    void recreateSwapChain()
    {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(m_window, &width, &height);
            glfwWaitEvents();
        }

        m_device.waitIdle();
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createFramebuffers();
        createPresentSyncObjects();
    }

    void updateCamera(float deltaSeconds)
    {
        const Vec3 forward = cameraForward();
        Vec3 flatForward = normalize(Vec3{ forward.x, 0.0f, forward.z });
        if (length(flatForward) <= 1e-5f) {
            flatForward = { 0.0f, 0.0f, -1.0f };
        }

        const Vec3 flatRight = normalize(cross(flatForward, Vec3{ 0.0f, 1.0f, 0.0f }));
        const float speed = 2.5f * deltaSeconds;

        Vec3 movement{};
        if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS) {
            movement = movement + flatForward * speed;
        }
        if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS) {
            movement = movement - flatForward * speed;
        }
        if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS) {
            movement = movement - flatRight * speed;
        }
        if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS) {
            movement = movement + flatRight * speed;
        }

        m_cameraPosition = m_cameraPosition + movement;
    }

    void mainLoop()
    {
        m_lastFrameTime = std::chrono::steady_clock::now();

        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
            const auto currentTime = std::chrono::steady_clock::now();
            const float deltaSeconds = std::chrono::duration<float>(currentTime - m_lastFrameTime).count();
            m_lastFrameTime = currentTime;
            updateCamera(deltaSeconds);
            drawFrame();
        }
        m_device.waitIdle();
    }

    void cleanup()
    {
        if (m_device != nullptr) {
            m_device.waitIdle();
        }

        if (m_window) {
            glfwDestroyWindow(m_window);
            m_window = nullptr;
        }
        glfwTerminate();
    }

    void drawFrame()
    {
        vk::Fence inFlightFence = *m_inFlightFences[m_currentFrame];
        static_cast<void>(m_device.waitForFences(inFlightFence, VK_TRUE, UINT64_MAX));

        vk::Semaphore imageAvailable = *m_imageAvailableSems[m_currentFrame];
        auto acquireResult = m_swapChain.acquireNextImage(UINT64_MAX, imageAvailable, {});

        if (acquireResult.result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        if (acquireResult.result != vk::Result::eSuccess &&
            acquireResult.result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to acquire swap chain image");
        }

        const uint32_t imageIndex = acquireResult.value;
        if (m_imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            static_cast<void>(m_device.waitForFences(m_imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX));
        }
        m_imagesInFlight[imageIndex] = inFlightFence;

        m_device.resetFences(inFlightFence);

        auto& commandBuffer = m_commandBuffers[m_currentFrame];
        commandBuffer.reset();
        recordCommandBuffer(commandBuffer, imageIndex);

        vk::CommandBuffer rawCommandBuffer = *commandBuffer;
        vk::Semaphore renderFinished = *m_renderFinishedSems[imageIndex];
        vk::Semaphore signalSems[] = { renderFinished };
        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

        vk::SubmitInfo submitInfo{};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailable;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &rawCommandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSems;

        m_graphicsQueue.submit(submitInfo, inFlightFence);

        vk::SwapchainKHR swapChain = *m_swapChain;
        vk::PresentInfoKHR presentInfo{};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSems;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapChain;
        presentInfo.pImageIndices = &imageIndex;

        vk::Result result = m_presentQueue.presentKHR(presentInfo);
        if (result == vk::Result::eErrorOutOfDateKHR ||
            result == vk::Result::eSuboptimalKHR ||
            m_framebufferResized) {
            m_framebufferResized = false;
            recreateSwapChain();
        } else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to present swap chain image");
        }

        m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
};

int main()
{
    VulkanApp app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
