cmake_minimum_required(VERSION 3.20)
project(VulkanTriangle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Vulkan
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED)

# ---------------------------------------------------------------------------
# GLFW  (fetched automatically – no pre-install needed)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ---------------------------------------------------------------------------
# Shader compilation  (requires glslc from the Vulkan SDK)
# ---------------------------------------------------------------------------
find_program(GLSLC glslc
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
)
if(NOT GLSLC)
    message(FATAL_ERROR
        "glslc not found. Make sure the Vulkan SDK is installed and "
        "VULKAN_SDK env-var is set (e.g. C:/VulkanSDK/1.x.y.z).")
endif()

set(SHADER_SRC_DIR  "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_BIN_DIR  "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_BIN_DIR}")

set(SHADER_SOURCES
    "${SHADER_SRC_DIR}/triangle.vert"
    "${SHADER_SRC_DIR}/triangle.frag"
)

foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV "${SHADER_BIN_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT  ${SPV}
        COMMAND ${GLSLC} ${SHADER} -o ${SPV}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader: ${SHADER_NAME}"
        VERBATIM
    )
    list(APPEND SPV_FILES ${SPV})
endforeach()

add_custom_target(Shaders ALL DEPENDS ${SPV_FILES})

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(VulkanTriangle src/main.cpp)
add_dependencies(VulkanTriangle Shaders)

target_include_directories(VulkanTriangle PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(VulkanTriangle PRIVATE Vulkan::Vulkan glfw)

# Copy compiled shaders next to the executable so relative path "shaders/" works
add_custom_command(TARGET VulkanTriangle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADER_BIN_DIR}"
        "$<TARGET_FILE_DIR:VulkanTriangle>/shaders"
    COMMENT "Copying shaders to output directory"
)

# Set VS debugger working directory so shaders are found when F5-launched
set_target_properties(VulkanTriangle PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:VulkanTriangle>"
)
