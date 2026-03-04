#define GLFW_INCLUDE_VULKAN
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr uint32_t WIDTH  = 800;
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::vector<char> readFile(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + path);

    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    return buf;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*userdata*/)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[Vulkan] " << data->pMessage << '\n';
    return VK_FALSE;
}

// Proxy loaders for debug-utils extension functions
static VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pMessenger)
{
    auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    return fn ? fn(instance, pCreateInfo, pAllocator, pMessenger)
              : VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void DestroyDebugUtilsMessengerEXT(
    VkInstance instance, VkDebugUtilsMessengerEXT messenger,
    const VkAllocationCallbacks* pAllocator)
{
    auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (fn) fn(instance, messenger, pAllocator);
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------
struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool isComplete() const { return graphics.has_value() && present.has_value(); }
};

struct SwapChainSupport {
    VkSurfaceCapabilitiesKHR        capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

// ---------------------------------------------------------------------------
// Application class
// ---------------------------------------------------------------------------
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
    // ---- GLFW ----
    GLFWwindow* m_window = nullptr;
    bool        m_framebufferResized = false;

    // ---- Core Vulkan handles ----
    VkInstance               m_instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR             m_surface        = VK_NULL_HANDLE;
    VkPhysicalDevice         m_physDevice     = VK_NULL_HANDLE;
    VkDevice                 m_device         = VK_NULL_HANDLE;
    VkQueue                  m_graphicsQueue  = VK_NULL_HANDLE;
    VkQueue                  m_presentQueue   = VK_NULL_HANDLE;

    // ---- Swap chain ----
    VkSwapchainKHR           m_swapChain      = VK_NULL_HANDLE;
    VkFormat                 m_scFormat{};
    VkExtent2D               m_scExtent{};
    std::vector<VkImage>     m_scImages;
    std::vector<VkImageView> m_scImageViews;

    // ---- Render pass / pipeline ----
    VkRenderPass     m_renderPass      = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout  = VK_NULL_HANDLE;
    VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;

    // ---- Framebuffers / commands ----
    std::vector<VkFramebuffer>   m_framebuffers;
    VkCommandPool                m_commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;

    // ---- Synchronisation ----
    std::vector<VkSemaphore> m_imageAvailableSems;
    std::vector<VkSemaphore> m_renderFinishedSems;
    std::vector<VkFence>     m_inFlightFences;
    uint32_t                 m_currentFrame = 0;

    // =======================================================================
    // Window
    // =======================================================================
    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Triangle", nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, [](GLFWwindow* w, int, int){
            reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(w))
                ->m_framebufferResized = true;
        });
    }

    // =======================================================================
    // Vulkan init sequence
    // =======================================================================
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

    // -----------------------------------------------------------------------
    // Instance
    // -----------------------------------------------------------------------
    void createInstance()
    {
        if (kEnableValidation && !checkValidationLayers())
            throw std::runtime_error("Validation layers not available");

        VkApplicationInfo app{};
        app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName   = "VulkanTriangle";
        app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app.pEngineName        = "None";
        app.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
        app.apiVersion         = VK_API_VERSION_1_0;

        auto extensions = requiredExtensions();

        VkInstanceCreateInfo ci{};
        ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo        = &app;
        ci.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT dbgCi{};
        if (kEnableValidation) {
            ci.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
            ci.ppEnabledLayerNames = kValidationLayers.data();
            fillDebugMessengerCI(dbgCi);
            ci.pNext = &dbgCi;
        }

        if (vkCreateInstance(&ci, nullptr, &m_instance) != VK_SUCCESS)
            throw std::runtime_error("vkCreateInstance failed");
    }

    bool checkValidationLayers()
    {
        uint32_t count;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> available(count);
        vkEnumerateInstanceLayerProperties(&count, available.data());
        for (const char* name : kValidationLayers) {
            bool found = false;
            for (const auto& p : available)
                if (strcmp(name, p.layerName) == 0) { found = true; break; }
            if (!found) return false;
        }
        return true;
    }

    std::vector<const char*> requiredExtensions()
    {
        uint32_t     glfwCount{};
        const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
        std::vector<const char*> exts(glfwExts, glfwExts + glfwCount);
        if (kEnableValidation)
            exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        return exts;
    }

    // -----------------------------------------------------------------------
    // Debug messenger
    // -----------------------------------------------------------------------
    void fillDebugMessengerCI(VkDebugUtilsMessengerCreateInfoEXT& ci)
    {
        ci       = {};
        ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        ci.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        ci.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT    |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        ci.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger()
    {
        if (!kEnableValidation) return;
        VkDebugUtilsMessengerCreateInfoEXT ci{};
        fillDebugMessengerCI(ci);
        if (CreateDebugUtilsMessengerEXT(m_instance, &ci, nullptr, &m_debugMessenger) != VK_SUCCESS)
            throw std::runtime_error("Failed to create debug messenger");
    }

    // -----------------------------------------------------------------------
    // Surface
    // -----------------------------------------------------------------------
    void createSurface()
    {
        if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS)
            throw std::runtime_error("Failed to create window surface");
    }

    // -----------------------------------------------------------------------
    // Physical device
    // -----------------------------------------------------------------------
    void pickPhysicalDevice()
    {
        uint32_t count;
        vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
        if (count == 0) throw std::runtime_error("No Vulkan-capable GPU found");
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

        for (auto d : devices) {
            if (isSuitable(d)) { m_physDevice = d; break; }
        }
        if (m_physDevice == VK_NULL_HANDLE)
            throw std::runtime_error("No suitable GPU found");

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(m_physDevice, &props);
        std::cout << "GPU: " << props.deviceName << '\n';
    }

    bool isSuitable(VkPhysicalDevice dev)
    {
        return findQueueFamilies(dev).isComplete() &&
               checkDeviceExtensions(dev) &&
               querySwapChainSupport(dev).formats.size() > 0 &&
               querySwapChainSupport(dev).presentModes.size() > 0;
    }

    bool checkDeviceExtensions(VkPhysicalDevice dev)
    {
        uint32_t count;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> available(count);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, available.data());
        std::set<std::string> required(kDeviceExtensions.begin(), kDeviceExtensions.end());
        for (const auto& e : available) required.erase(e.extensionName);
        return required.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev)
    {
        QueueFamilyIndices idx;
        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());

        for (uint32_t i = 0; i < count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                idx.graphics = i;
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, m_surface, &presentSupport);
            if (presentSupport) idx.present = i;
            if (idx.isComplete()) break;
        }
        return idx;
    }

    SwapChainSupport querySwapChainSupport(VkPhysicalDevice dev)
    {
        SwapChainSupport sc;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, m_surface, &sc.capabilities);

        uint32_t fmtCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, m_surface, &fmtCount, nullptr);
        if (fmtCount) { sc.formats.resize(fmtCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(dev, m_surface, &fmtCount, sc.formats.data()); }

        uint32_t pmCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, m_surface, &pmCount, nullptr);
        if (pmCount) { sc.presentModes.resize(pmCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(dev, m_surface, &pmCount, sc.presentModes.data()); }

        return sc;
    }

    // -----------------------------------------------------------------------
    // Logical device
    // -----------------------------------------------------------------------
    void createLogicalDevice()
    {
        QueueFamilyIndices idx = findQueueFamilies(m_physDevice);
        std::set<uint32_t> uniqueQueueFamilies = { idx.graphics.value(), idx.present.value() };

        float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueCIs;
        for (uint32_t qf : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo ci{};
            ci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            ci.queueFamilyIndex = qf;
            ci.queueCount       = 1;
            ci.pQueuePriorities = &priority;
            queueCIs.push_back(ci);
        }

        VkPhysicalDeviceFeatures features{};

        VkDeviceCreateInfo ci{};
        ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        ci.queueCreateInfoCount    = static_cast<uint32_t>(queueCIs.size());
        ci.pQueueCreateInfos       = queueCIs.data();
        ci.pEnabledFeatures        = &features;
        ci.enabledExtensionCount   = static_cast<uint32_t>(kDeviceExtensions.size());
        ci.ppEnabledExtensionNames = kDeviceExtensions.data();
        if (kEnableValidation) {
            ci.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
            ci.ppEnabledLayerNames = kValidationLayers.data();
        }

        if (vkCreateDevice(m_physDevice, &ci, nullptr, &m_device) != VK_SUCCESS)
            throw std::runtime_error("Failed to create logical device");

        vkGetDeviceQueue(m_device, idx.graphics.value(), 0, &m_graphicsQueue);
        vkGetDeviceQueue(m_device, idx.present.value(),  0, &m_presentQueue);
    }

    // -----------------------------------------------------------------------
    // Swap chain
    // -----------------------------------------------------------------------
    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& fmts)
    {
        for (const auto& f : fmts)
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return f;
        return fmts[0];
    }

    VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes)
    {
        for (auto m : modes)
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& caps)
    {
        if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return caps.currentExtent;

        int w, h;
        glfwGetFramebufferSize(m_window, &w, &h);
        VkExtent2D ext = { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
        ext.width  = std::clamp(ext.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        ext.height = std::clamp(ext.height, caps.minImageExtent.height, caps.maxImageExtent.height);
        return ext;
    }

    void createSwapChain()
    {
        SwapChainSupport sc  = querySwapChainSupport(m_physDevice);
        VkSurfaceFormatKHR fmt = chooseSurfaceFormat(sc.formats);
        VkPresentModeKHR   pm  = choosePresentMode(sc.presentModes);
        VkExtent2D         ext = chooseExtent(sc.capabilities);

        uint32_t imageCount = sc.capabilities.minImageCount + 1;
        if (sc.capabilities.maxImageCount > 0)
            imageCount = std::min(imageCount, sc.capabilities.maxImageCount);

        VkSwapchainCreateInfoKHR ci{};
        ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        ci.surface          = m_surface;
        ci.minImageCount    = imageCount;
        ci.imageFormat      = fmt.format;
        ci.imageColorSpace  = fmt.colorSpace;
        ci.imageExtent      = ext;
        ci.imageArrayLayers = 1;
        ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices idx = findQueueFamilies(m_physDevice);
        uint32_t families[] = { idx.graphics.value(), idx.present.value() };
        if (idx.graphics.value() != idx.present.value()) {
            ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices   = families;
        } else {
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        ci.preTransform   = sc.capabilities.currentTransform;
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        ci.presentMode    = pm;
        ci.clipped        = VK_TRUE;
        ci.oldSwapchain   = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(m_device, &ci, nullptr, &m_swapChain) != VK_SUCCESS)
            throw std::runtime_error("Failed to create swap chain");

        vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, nullptr);
        m_scImages.resize(imageCount);
        vkGetSwapchainImagesKHR(m_device, m_swapChain, &imageCount, m_scImages.data());
        m_scFormat = fmt.format;
        m_scExtent = ext;
    }

    // -----------------------------------------------------------------------
    // Image views
    // -----------------------------------------------------------------------
    void createImageViews()
    {
        m_scImageViews.resize(m_scImages.size());
        for (size_t i = 0; i < m_scImages.size(); ++i) {
            VkImageViewCreateInfo ci{};
            ci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ci.image                           = m_scImages[i];
            ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            ci.format                          = m_scFormat;
            ci.components                      = { VK_COMPONENT_SWIZZLE_IDENTITY,
                                                   VK_COMPONENT_SWIZZLE_IDENTITY,
                                                   VK_COMPONENT_SWIZZLE_IDENTITY,
                                                   VK_COMPONENT_SWIZZLE_IDENTITY };
            ci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            ci.subresourceRange.baseMipLevel   = 0;
            ci.subresourceRange.levelCount     = 1;
            ci.subresourceRange.baseArrayLayer = 0;
            ci.subresourceRange.layerCount     = 1;

            if (vkCreateImageView(m_device, &ci, nullptr, &m_scImageViews[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create image view");
        }
    }

    // -----------------------------------------------------------------------
    // Render pass
    // -----------------------------------------------------------------------
    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format         = m_scFormat;
        colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments    = &colorRef;

        VkSubpassDependency dep{};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass    = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0;
        dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo ci{};
        ci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        ci.attachmentCount = 1;
        ci.pAttachments    = &colorAttachment;
        ci.subpassCount    = 1;
        ci.pSubpasses      = &subpass;
        ci.dependencyCount = 1;
        ci.pDependencies   = &dep;

        if (vkCreateRenderPass(m_device, &ci, nullptr, &m_renderPass) != VK_SUCCESS)
            throw std::runtime_error("Failed to create render pass");
    }

    // -----------------------------------------------------------------------
    // Graphics pipeline
    // -----------------------------------------------------------------------
    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo ci{};
        ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = code.size();
        ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule mod;
        if (vkCreateShaderModule(m_device, &ci, nullptr, &mod) != VK_SUCCESS)
            throw std::runtime_error("Failed to create shader module");
        return mod;
    }

    void createGraphicsPipeline()
    {
        auto vertCode = readFile("shaders/triangle.vert.spv");
        auto fragCode = readFile("shaders/triangle.frag.spv");

        VkShaderModule vertMod = createShaderModule(vertCode);
        VkShaderModule fragMod = createShaderModule(fragCode);

        // Shader stages
        VkPipelineShaderStageCreateInfo vertStage{};
        vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module = vertMod;
        vertStage.pName  = "main";

        VkPipelineShaderStageCreateInfo fragStage{};
        fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = fragMod;
        fragStage.pName  = "main";

        VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

        // Vertex input — no vertex buffer; data is hardcoded in the shader
        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        // Input assembly
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // Viewport / scissor  (dynamic state)
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount  = 1;

        // Rasterizer
        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode    = VK_CULL_MODE_BACK_BIT;
        raster.frontFace   = VK_FRONT_FACE_CLOCKWISE;
        raster.lineWidth   = 1.0f;

        // Multisampling
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Colour blending — opaque
        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments    = &blendAttachment;

        // Dynamic state — viewport and scissor set at draw time
        VkDynamicState dynamicStates[] = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates    = dynamicStates;

        // Pipeline layout (no descriptors)
        VkPipelineLayoutCreateInfo layoutCI{};
        layoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        if (vkCreatePipelineLayout(m_device, &layoutCI, nullptr, &m_pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create pipeline layout");

        VkGraphicsPipelineCreateInfo pipelineCI{};
        pipelineCI.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCI.stageCount          = 2;
        pipelineCI.pStages             = stages;
        pipelineCI.pVertexInputState   = &vertexInput;
        pipelineCI.pInputAssemblyState = &inputAssembly;
        pipelineCI.pViewportState      = &viewportState;
        pipelineCI.pRasterizationState = &raster;
        pipelineCI.pMultisampleState   = &multisampling;
        pipelineCI.pColorBlendState    = &colorBlend;
        pipelineCI.pDynamicState       = &dynamicState;
        pipelineCI.layout              = m_pipelineLayout;
        pipelineCI.renderPass          = m_renderPass;
        pipelineCI.subpass             = 0;

        if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr,
                                      &m_graphicsPipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline");

        vkDestroyShaderModule(m_device, vertMod, nullptr);
        vkDestroyShaderModule(m_device, fragMod, nullptr);
    }

    // -----------------------------------------------------------------------
    // Framebuffers
    // -----------------------------------------------------------------------
    void createFramebuffers()
    {
        m_framebuffers.resize(m_scImageViews.size());
        for (size_t i = 0; i < m_scImageViews.size(); ++i) {
            VkFramebufferCreateInfo ci{};
            ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            ci.renderPass      = m_renderPass;
            ci.attachmentCount = 1;
            ci.pAttachments    = &m_scImageViews[i];
            ci.width           = m_scExtent.width;
            ci.height          = m_scExtent.height;
            ci.layers          = 1;
            if (vkCreateFramebuffer(m_device, &ci, nullptr, &m_framebuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create framebuffer");
        }
    }

    // -----------------------------------------------------------------------
    // Command pool and buffers
    // -----------------------------------------------------------------------
    void createCommandPool()
    {
        QueueFamilyIndices idx = findQueueFamilies(m_physDevice);
        VkCommandPoolCreateInfo ci{};
        ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = idx.graphics.value();
        if (vkCreateCommandPool(m_device, &ci, nullptr, &m_commandPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create command pool");
    }

    void createCommandBuffers()
    {
        m_commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool        = m_commandPool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = static_cast<uint32_t>(m_commandBuffers.size());
        if (vkAllocateCommandBuffers(m_device, &ai, m_commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate command buffers");
    }

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("Failed to begin command buffer");

        VkClearValue clearColor = { {{ 0.05f, 0.05f, 0.05f, 1.0f }} };

        VkRenderPassBeginInfo rpInfo{};
        rpInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpInfo.renderPass        = m_renderPass;
        rpInfo.framebuffer       = m_framebuffers[imageIndex];
        rpInfo.renderArea.offset = { 0, 0 };
        rpInfo.renderArea.extent = m_scExtent;
        rpInfo.clearValueCount   = 1;
        rpInfo.pClearValues      = &clearColor;

        vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

        VkViewport viewport{};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = static_cast<float>(m_scExtent.width);
        viewport.height   = static_cast<float>(m_scExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{ {0, 0}, m_scExtent };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdDraw(cmd, 3, 1, 0, 0);    // 3 vertices, hardcoded in shader

        vkCmdEndRenderPass(cmd);
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
            throw std::runtime_error("Failed to record command buffer");
    }

    // -----------------------------------------------------------------------
    // Synchronisation objects
    // -----------------------------------------------------------------------
    void createSyncObjects()
    {
        m_imageAvailableSems.resize(MAX_FRAMES_IN_FLIGHT);
        m_renderFinishedSems.resize(MAX_FRAMES_IN_FLIGHT);
        m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semCI{};
        semCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;   // start signalled so first wait succeeds

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            if (vkCreateSemaphore(m_device, &semCI, nullptr, &m_imageAvailableSems[i]) != VK_SUCCESS ||
                vkCreateSemaphore(m_device, &semCI, nullptr, &m_renderFinishedSems[i]) != VK_SUCCESS ||
                vkCreateFence    (m_device, &fenceCI, nullptr, &m_inFlightFences[i])   != VK_SUCCESS)
                throw std::runtime_error("Failed to create sync objects");
        }
    }

    // =======================================================================
    // Swap chain recreation (resize)
    // =======================================================================
    void cleanupSwapChain()
    {
        for (auto fb : m_framebuffers)    vkDestroyFramebuffer(m_device, fb, nullptr);
        for (auto iv : m_scImageViews)    vkDestroyImageView(m_device, iv, nullptr);
        vkDestroySwapchainKHR(m_device, m_swapChain, nullptr);
    }

    void recreateSwapChain()
    {
        // Handle minimisation — wait until the window has a non-zero size
        int w = 0, h = 0;
        glfwGetFramebufferSize(m_window, &w, &h);
        while (w == 0 || h == 0) {
            glfwGetFramebufferSize(m_window, &w, &h);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(m_device);
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    // =======================================================================
    // Render loop
    // =======================================================================
    void mainLoop()
    {
        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
            drawFrame();
        }
        vkDeviceWaitIdle(m_device);
    }

    void drawFrame()
    {
        // Wait for the previous frame using this slot to finish
        vkWaitForFences(m_device, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(m_device, m_swapChain, UINT64_MAX,
                                                m_imageAvailableSems[m_currentFrame],
                                                VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swap chain image");
        }

        vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrame]);

        VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
        vkResetCommandBuffer(cmd, 0);
        recordCommandBuffer(cmd, imageIndex);

        VkSemaphore          waitSems[]   = { m_imageAvailableSems[m_currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        VkSemaphore          signalSems[] = { m_renderFinishedSems[m_currentFrame] };

        VkSubmitInfo submit{};
        submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount   = 1;
        submit.pWaitSemaphores      = waitSems;
        submit.pWaitDstStageMask    = waitStages;
        submit.commandBufferCount   = 1;
        submit.pCommandBuffers      = &cmd;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores    = signalSems;

        if (vkQueueSubmit(m_graphicsQueue, 1, &submit, m_inFlightFences[m_currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("Failed to submit draw command");

        VkPresentInfoKHR present{};
        present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores    = signalSems;
        present.swapchainCount     = 1;
        present.pSwapchains        = &m_swapChain;
        present.pImageIndices      = &imageIndex;

        result = vkQueuePresentKHR(m_presentQueue, &present);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_framebufferResized) {
            m_framebufferResized = false;
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to present swap chain image");
        }

        m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // =======================================================================
    // Cleanup
    // =======================================================================
    void cleanup()
    {
        cleanupSwapChain();

        vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        vkDestroyRenderPass(m_device, m_renderPass, nullptr);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            vkDestroySemaphore(m_device, m_imageAvailableSems[i], nullptr);
            vkDestroySemaphore(m_device, m_renderFinishedSems[i], nullptr);
            vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
        }

        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        vkDestroyDevice(m_device, nullptr);

        if (kEnableValidation)
            DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);

        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        vkDestroyInstance(m_instance, nullptr);

        glfwDestroyWindow(m_window);
        glfwTerminate();
    }
};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
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
