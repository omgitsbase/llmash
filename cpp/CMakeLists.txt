cmake_minimum_required(VERSION 3.20)
project(llmash CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only, both vendored, both the ones llama.cpp's own server uses.
add_library(llmash_vendor INTERFACE)
target_include_directories(llmash_vendor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/vendor)

set(LLMASH_SOURCES
    src/config.cpp
    src/gguf.cpp
    src/registry.cpp
)

add_library(llmash_core STATIC ${LLMASH_SOURCES})
target_include_directories(llmash_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(llmash_core PUBLIC llmash_vendor)

if (WIN32)
    target_compile_definitions(llmash_core PUBLIC
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
        UNICODE
        _UNICODE)
    target_link_libraries(llmash_core PUBLIC shlwapi)
endif()

add_executable(llmash src/main.cpp)
target_link_libraries(llmash PRIVATE llmash_core)

# the same program without a console, for the tray
add_executable(llmashw src/main.cpp)
target_link_libraries(llmashw PRIVATE llmash_core)
if (MSVC)
    set_target_properties(llmashw PROPERTIES LINK_FLAGS "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
endif()

enable_testing()
add_executable(llmash_test src/test_main.cpp)
target_link_libraries(llmash_test PRIVATE llmash_core)
add_test(NAME llmash_test COMMAND llmash_test)
