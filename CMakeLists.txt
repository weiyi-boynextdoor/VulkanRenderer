cmake_minimum_required(VERSION 3.20)
project(VulkanRenderer LANGUAGES CXX)

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

file(GLOB_RECURSE SLANG_SHADERS CONFIGURE_DEPENDS
    "${SHADER_SRC_DIR}/*.slang"
)

if(NOT SLANG_SHADERS)
    message(FATAL_ERROR "No .slang files found under ${SHADER_SRC_DIR}")
endif()

set(SPV_FILES)
foreach(SLANG_SHADER IN LISTS SLANG_SHADERS)
    file(RELATIVE_PATH SLANG_REL_PATH "${SHADER_SRC_DIR}" "${SLANG_SHADER}")
    get_filename_component(SLANG_REL_DIR "${SLANG_REL_PATH}" DIRECTORY)
    get_filename_component(SLANG_NAME_WE "${SLANG_SHADER}" NAME_WE)

    if(SLANG_REL_DIR)
        set(SHADER_OUTPUT_DIR "${SHADER_BIN_DIR}/${SLANG_REL_DIR}")
    else()
        set(SHADER_OUTPUT_DIR "${SHADER_BIN_DIR}")
    endif()

    file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

    set(SHADER_SPV "${SHADER_OUTPUT_DIR}/${SLANG_NAME_WE}.spv")

    add_custom_command(
        OUTPUT  ${SHADER_SPV}
        COMMAND ${SLANGC}
            -target spirv
            -profile spirv_1_3
            -emit-spirv-directly
            -fvk-use-entrypoint-name
            -entry vertexMain
            -entry fragmentMain
            -o ${SHADER_SPV}
            ${SLANG_SHADER}
        DEPENDS ${SLANG_SHADER}
        COMMENT "Compiling Slang shader module: ${SLANG_REL_PATH}"
        VERBATIM
    )

    list(APPEND SPV_FILES ${SHADER_SPV})
endforeach()

add_custom_target(Shaders ALL DEPENDS ${SPV_FILES})

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(VulkanRenderer src/main.cpp)
add_dependencies(VulkanRenderer Shaders)

target_include_directories(VulkanRenderer PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(VulkanRenderer PRIVATE Vulkan::Vulkan glfw)

# Copy compiled shaders next to the executable so relative path "shaders/" works
add_custom_command(TARGET VulkanRenderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADER_BIN_DIR}"
        "$<TARGET_FILE_DIR:VulkanRenderer>/shaders"
    COMMENT "Copying shaders to output directory"
)

# Set VS debugger working directory so shaders are found when F5-launched
set_target_properties(VulkanRenderer PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:VulkanRenderer>"
)
