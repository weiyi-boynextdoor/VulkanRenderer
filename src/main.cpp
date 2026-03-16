#define GLFW_INCLUDE_NONE
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
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

static std::vector<char> readFile(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    size_t size = static_cast<size_t>(file.tellg());
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

static vk::DebugUtilsMessengerEXT createDebugUtilsMessenger(
    vk::Instance instance,
    const vk::DebugUtilsMessengerCreateInfoEXT& createInfo)
{
    auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(static_cast<VkInstance>(instance), "vkCreateDebugUtilsMessengerEXT"));
    if (!fn) {
        throw std::runtime_error("Debug utils extension is not available");
    }

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkResult result = fn(
        static_cast<VkInstance>(instance),
        reinterpret_cast<const VkDebugUtilsMessengerCreateInfoEXT*>(&createInfo),
        nullptr,
        &messenger);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create debug messenger");
    }

    return messenger;
}

static void destroyDebugUtilsMessenger(vk::Instance instance, vk::DebugUtilsMessengerEXT messenger)
{
    auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(static_cast<VkInstance>(instance), "vkDestroyDebugUtilsMessengerEXT"));
    if (fn) {
        fn(static_cast<VkInstance>(instance), static_cast<VkDebugUtilsMessengerEXT>(messenger), nullptr);
    }
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

    vk::Instance m_instance{};
    vk::DebugUtilsMessengerEXT m_debugMessenger{};
    vk::SurfaceKHR m_surface{};
    vk::PhysicalDevice m_physDevice{};
    vk::Device m_device{};
    vk::Queue m_graphicsQueue{};
    vk::Queue m_presentQueue{};

    vk::SwapchainKHR m_swapChain{};
    vk::Format m_scFormat{};
    vk::Extent2D m_scExtent{};
    std::vector<vk::Image> m_scImages;
    std::vector<vk::ImageView> m_scImageViews;

    vk::RenderPass m_renderPass{};
    vk::PipelineLayout m_pipelineLayout{};
    vk::Pipeline m_graphicsPipeline{};

    std::vector<vk::Framebuffer> m_framebuffers;
    vk::CommandPool m_commandPool{};
    std::vector<vk::CommandBuffer> m_commandBuffers;

    std::vector<vk::Semaphore> m_imageAvailableSems;
    std::vector<vk::Semaphore> m_renderFinishedSems;
    std::vector<vk::Fence> m_inFlightFences;
    uint32_t m_currentFrame = 0;

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Triangle", nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* window, int, int) {
            reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(window))->m_framebufferResized = true;
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
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    void createInstance()
    {
        if (kEnableValidation && !checkValidationLayers()) {
            throw std::runtime_error("Validation layers not available");
        }

        vk::ApplicationInfo appInfo{};
        appInfo.pApplicationName = "VulkanTriangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "None";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

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

        m_instance = vk::createInstance(createInfo);
    }

    bool checkValidationLayers()
    {
        auto available = vk::enumerateInstanceLayerProperties();
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
        createInfo = vk::DebugUtilsMessengerCreateInfoEXT{};
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
        m_debugMessenger = createDebugUtilsMessenger(m_instance, createInfo);
    }

    void createSurface()
    {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        if (glfwCreateWindowSurface(static_cast<VkInstance>(m_instance), m_window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        }
        m_surface = surface;
    }

    void pickPhysicalDevice()
    {
        auto devices = m_instance.enumeratePhysicalDevices();
        if (devices.empty()) {
            throw std::runtime_error("No Vulkan-capable GPU found");
        }

        for (const auto& device : devices) {
            if (isSuitable(device)) {
                m_physDevice = device;
                break;
            }
        }

        if (!m_physDevice) {
            throw std::runtime_error("No suitable GPU found");
        }

        auto props = m_physDevice.getProperties();
        std::cout << "GPU: " << props.deviceName << '\n';
    }

    bool isSuitable(vk::PhysicalDevice device)
    {
        if (!findQueueFamilies(device).isComplete() || !checkDeviceExtensions(device)) {
            return false;
        }

        auto swapChainSupport = querySwapChainSupport(device);
        return !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    bool checkDeviceExtensions(vk::PhysicalDevice device)
    {
        auto available = device.enumerateDeviceExtensionProperties();
        std::set<std::string> required(kDeviceExtensions.begin(), kDeviceExtensions.end());
        for (const auto& extension : available) {
            required.erase(extension.extensionName);
        }
        return required.empty();
    }

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device)
    {
        QueueFamilyIndices indices;
        auto families = device.getQueueFamilyProperties();

        for (uint32_t i = 0; i < static_cast<uint32_t>(families.size()); ++i) {
            if (families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphics = i;
            }
            if (device.getSurfaceSupportKHR(i, m_surface)) {
                indices.present = i;
            }
            if (indices.isComplete()) {
                break;
            }
        }

        return indices;
    }

    SwapChainSupport querySwapChainSupport(vk::PhysicalDevice device)
    {
        SwapChainSupport support;
        support.capabilities = device.getSurfaceCapabilitiesKHR(m_surface);
        support.formats = device.getSurfaceFormatsKHR(m_surface);
        support.presentModes = device.getSurfacePresentModesKHR(m_surface);
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

        vk::PhysicalDeviceFeatures features{};
        vk::DeviceCreateInfo createInfo{};
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &features;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
        createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();
        if (kEnableValidation) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
            createInfo.ppEnabledLayerNames = kValidationLayers.data();
        }

        m_device = m_physDevice.createDevice(createInfo);
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
        createInfo.surface = m_surface;
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

        m_swapChain = m_device.createSwapchainKHR(createInfo);
        m_scImages = m_device.getSwapchainImagesKHR(m_swapChain);
        m_scFormat = surfaceFormat.format;
        m_scExtent = extent;
    }

    void createImageViews()
    {
        m_scImageViews.resize(m_scImages.size());
        for (size_t i = 0; i < m_scImages.size(); ++i) {
            vk::ImageViewCreateInfo createInfo{};
            createInfo.image = m_scImages[i];
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

            m_scImageViews[i] = m_device.createImageView(createInfo);
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

        m_renderPass = m_device.createRenderPass(createInfo);
    }

    vk::ShaderModule createShaderModule(const std::vector<char>& code)
    {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        return m_device.createShaderModule(createInfo);
    }

    void createGraphicsPipeline()
    {
        auto vertCode = readFile("shaders/triangle.vert.spv");
        auto fragCode = readFile("shaders/triangle.frag.spv");

        vk::ShaderModule vertModule = createShaderModule(vertCode);
        vk::ShaderModule fragModule = createShaderModule(fragCode);

        vk::PipelineShaderStageCreateInfo vertStage{};
        vertStage.stage = vk::ShaderStageFlagBits::eVertex;
        vertStage.module = vertModule;
        vertStage.pName = "main";

        vk::PipelineShaderStageCreateInfo fragStage{};
        fragStage.stage = vk::ShaderStageFlagBits::eFragment;
        fragStage.module = fragModule;
        fragStage.pName = "main";

        vk::PipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

        vk::PipelineVertexInputStateCreateInfo vertexInput{};

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

        vk::PipelineViewportStateCreateInfo viewportState{};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.frontFace = vk::FrontFace::eClockwise;
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

        vk::PipelineLayoutCreateInfo layoutInfo{};
        m_pipelineLayout = m_device.createPipelineLayout(layoutInfo);

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
        pipelineInfo.layout = m_pipelineLayout;
        pipelineInfo.renderPass = m_renderPass;
        pipelineInfo.subpass = 0;

        m_graphicsPipeline = m_device.createGraphicsPipeline({}, pipelineInfo).value;

        m_device.destroyShaderModule(vertModule);
        m_device.destroyShaderModule(fragModule);
    }

    void createFramebuffers()
    {
        m_framebuffers.resize(m_scImageViews.size());
        for (size_t i = 0; i < m_scImageViews.size(); ++i) {
            vk::FramebufferCreateInfo createInfo{};
            createInfo.renderPass = m_renderPass;
            createInfo.attachmentCount = 1;
            createInfo.pAttachments = &m_scImageViews[i];
            createInfo.width = m_scExtent.width;
            createInfo.height = m_scExtent.height;
            createInfo.layers = 1;

            m_framebuffers[i] = m_device.createFramebuffer(createInfo);
        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices indices = findQueueFamilies(m_physDevice);
        vk::CommandPoolCreateInfo createInfo{};
        createInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        createInfo.queueFamilyIndex = indices.graphics.value();
        m_commandPool = m_device.createCommandPool(createInfo);
    }

    void createCommandBuffers()
    {
        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = m_commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        m_commandBuffers = m_device.allocateCommandBuffers(allocInfo);
    }

    void recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t imageIndex)
    {
        vk::CommandBufferBeginInfo beginInfo{};
        commandBuffer.begin(beginInfo);

        vk::ClearValue clearColor = vk::ClearColorValue(std::array<float, 4>{ 0.05f, 0.05f, 0.05f, 1.0f });

        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.renderPass = m_renderPass;
        renderPassInfo.framebuffer = m_framebuffers[imageIndex];
        renderPassInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
        renderPassInfo.renderArea.extent = m_scExtent;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline);

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

        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRenderPass();
        commandBuffer.end();
    }

    void createSyncObjects()
    {
        m_imageAvailableSems.resize(MAX_FRAMES_IN_FLIGHT);
        m_renderFinishedSems.resize(MAX_FRAMES_IN_FLIGHT);
        m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        vk::SemaphoreCreateInfo semInfo{};
        vk::FenceCreateInfo fenceInfo{};
        fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            m_imageAvailableSems[i] = m_device.createSemaphore(semInfo);
            m_renderFinishedSems[i] = m_device.createSemaphore(semInfo);
            m_inFlightFences[i] = m_device.createFence(fenceInfo);
        }
    }

    void cleanupSwapChain()
    {
        for (auto framebuffer : m_framebuffers) {
            m_device.destroyFramebuffer(framebuffer);
        }
        m_framebuffers.clear();

        for (auto imageView : m_scImageViews) {
            m_device.destroyImageView(imageView);
        }
        m_scImageViews.clear();

        if (m_swapChain) {
            m_device.destroySwapchainKHR(m_swapChain);
            m_swapChain = nullptr;
        }
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
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
            drawFrame();
        }
        m_device.waitIdle();
    }

    void drawFrame()
    {
        static_cast<void>(m_device.waitForFences(m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX));

        uint32_t imageIndex = 0;
        vk::Result result = m_device.acquireNextImageKHR(
            m_swapChain,
            UINT64_MAX,
            m_imageAvailableSems[m_currentFrame],
            {},
            &imageIndex);

        if (result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Failed to acquire swap chain image");
        }

        m_device.resetFences(m_inFlightFences[m_currentFrame]);

        vk::CommandBuffer commandBuffer = m_commandBuffers[m_currentFrame];
        commandBuffer.reset();
        recordCommandBuffer(commandBuffer, imageIndex);

        vk::Semaphore waitSems[] = { m_imageAvailableSems[m_currentFrame] };
        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
        vk::Semaphore signalSems[] = { m_renderFinishedSems[m_currentFrame] };

        vk::SubmitInfo submitInfo{};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSems;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSems;

        m_graphicsQueue.submit(submitInfo, m_inFlightFences[m_currentFrame]);

        vk::PresentInfoKHR presentInfo{};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSems;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &m_swapChain;
        presentInfo.pImageIndices = &imageIndex;

        result = m_presentQueue.presentKHR(presentInfo);
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

    void cleanup()
    {
        cleanupSwapChain();

        if (m_graphicsPipeline) {
            m_device.destroyPipeline(m_graphicsPipeline);
        }
        if (m_pipelineLayout) {
            m_device.destroyPipelineLayout(m_pipelineLayout);
        }
        if (m_renderPass) {
            m_device.destroyRenderPass(m_renderPass);
        }

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (m_imageAvailableSems[i]) {
                m_device.destroySemaphore(m_imageAvailableSems[i]);
            }
            if (m_renderFinishedSems[i]) {
                m_device.destroySemaphore(m_renderFinishedSems[i]);
            }
            if (m_inFlightFences[i]) {
                m_device.destroyFence(m_inFlightFences[i]);
            }
        }

        if (m_commandPool) {
            m_device.destroyCommandPool(m_commandPool);
        }
        if (m_device) {
            m_device.destroy();
        }

        if (kEnableValidation && m_debugMessenger) {
            destroyDebugUtilsMessenger(m_instance, m_debugMessenger);
        }
        if (m_surface) {
            m_instance.destroySurfaceKHR(m_surface);
        }
        if (m_instance) {
            m_instance.destroy();
        }

        glfwDestroyWindow(m_window);
        glfwTerminate();
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
