cmake_minimum_required(VERSION 3.20)
project(llmash CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only, both vendored, both the ones llama.cpp's own server uses.
# cpp-httplib is vendored in its split form: the header declares, httplib.cpp
# defines. Without the .cpp every Client call is an unresolved symbol.
add_library(llmash_vendor STATIC vendor/httplib.cpp)
target_include_directories(llmash_vendor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vendor)
if (WIN32)
    target_link_libraries(llmash_vendor PUBLIC ws2_32 crypt32)
endif()

set(LLMASH_SOURCES
    src/config.cpp
    src/gguf.cpp
    src/registry.cpp
    src/manager.cpp
    src/help.cpp
    src/launch.cpp
    src/remote.cpp
    src/readline.cpp
    src/cmd_doctor.cpp
    src/cmd_models.cpp
    src/cli_console.cpp
    src/cli_format.cpp
    src/cli_http.cpp
    src/cli_process.cpp
    src/cli_run.cpp
    src/cli_util.cpp
    src/cli_win.cpp
    src/tray.cpp
    src/tray_process.cpp
    src/api_logic.cpp
    src/api.cpp
    src/chat.cpp
    src/cli_commands.cpp
    src/pull.cpp
    src/draft.cpp
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
    target_link_libraries(llmash_core PUBLIC shlwapi user32 shell32 advapi32 ws2_32 psapi)
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
foreach(t test_main cli_run_test test_cli_doctor_models tray_test
              api_logic_test api_test chat_test cli_commands_test pull_test)
    add_executable(${t} src/${t}.cpp)
    target_link_libraries(${t} PRIVATE llmash_core)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
