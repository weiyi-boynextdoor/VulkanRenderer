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
# Shader compilation  (requires slangc from the Vulkan SDK)
# ---------------------------------------------------------------------------
find_program(SLANGC slangc
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
)
if(NOT SLANGC)
    message(FATAL_ERROR
        "slangc not found. Make sure the Vulkan SDK is installed and "
        "VULKAN_SDK env-var is set (e.g. C:/VulkanSDK/1.x.y.z).")
endif()

set(SHADER_SRC_DIR  "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_BIN_DIR  "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_BIN_DIR}")

set(SLANG_SHADER "${SHADER_SRC_DIR}/triangle.slang")
set(VERT_SPV "${SHADER_BIN_DIR}/triangle.vert.spv")
set(FRAG_SPV "${SHADER_BIN_DIR}/triangle.frag.spv")

add_custom_command(
    OUTPUT  ${VERT_SPV}
    COMMAND ${SLANGC}
        -target spirv
        -profile glsl_450
        -entry vertexMain
        -stage vertex
        -o ${VERT_SPV}
        ${SLANG_SHADER}
    DEPENDS ${SLANG_SHADER}
    COMMENT "Compiling Slang vertex shader: triangle.slang"
    VERBATIM
)

add_custom_command(
    OUTPUT  ${FRAG_SPV}
    COMMAND ${SLANGC}
        -target spirv
        -profile glsl_450
        -entry fragmentMain
        -stage fragment
        -o ${FRAG_SPV}
        ${SLANG_SHADER}
    DEPENDS ${SLANG_SHADER}
    COMMENT "Compiling Slang fragment shader: triangle.slang"
    VERBATIM
)

set(SPV_FILES
    ${VERT_SPV}
    ${FRAG_SPV}
)

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
