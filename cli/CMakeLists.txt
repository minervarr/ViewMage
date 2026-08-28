# ---------------------------------------------------------------------------
# viewmage_cli — headless DNG/JXL develop-to-PNG.  No Vulkan, no GPU, no
# Wayland.  Links only viewmage_core (which brings libjxl).
# ---------------------------------------------------------------------------

add_executable(viewmage_cli
    cli_main.cc
    diagnose.cc
)

target_include_directories(viewmage_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
    ${CMAKE_SOURCE_DIR}/thirdparty/stb
)

target_compile_options(viewmage_cli PRIVATE -Wall)

target_link_libraries(viewmage_cli PRIVATE viewmage_core)
