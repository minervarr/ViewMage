# ---------------------------------------------------------------------------
# viewmage executable. Portable app logic (viewmage_app.cc, png_export.cc)
# plus exactly one platform Host backend via app_shell.
# ---------------------------------------------------------------------------

set(VIEWMAGE_GUI_SRC
    src/gui_main.cc
    src/viewmage_app.cc
    src/png_export.cc)

if(WIN32)
    # TODO: Win32 backend when needed
    message(FATAL_ERROR "Win32 backend not yet ported")
else()
    # Linux/Wayland
    add_executable(viewmage
        ${VIEWMAGE_GUI_SRC}
        ${CMAKE_SOURCE_DIR}/framework/vk_canvas/platform/linux/wayland_platform.cc)

    target_include_directories(viewmage PRIVATE
        src
        ${CMAKE_SOURCE_DIR}/core/include
        ${CMAKE_SOURCE_DIR}/thirdparty/stb
        ${CMAKE_BINARY_DIR}/generated
        ${CMAKE_SOURCE_DIR}/framework/vk_canvas/platform/linux)

    target_compile_options(viewmage PRIVATE -Wall)

    add_dependencies(viewmage compile_font_shaders compile_canvas_shaders)

    find_package(Threads REQUIRED)
    target_link_libraries(viewmage PRIVATE
        app_shell
        app_shell_wayland
        viewmage_core
        vk_canvas_core
        vk_canvas_wayland
        Threads::Threads)
endif()

# Compiled shaders land at ${CMAKE_BINARY_DIR}/assets/shaders — on Linux the
# exe lives in gui/, so FileAssetReader needs this copy to find them.
add_custom_target(copy_shader_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/assets
        $<TARGET_FILE_DIR:viewmage>/assets
    DEPENDS compile_font_shaders compile_canvas_shaders
    COMMENT "Copying compiled shaders next to the executable")
add_dependencies(copy_shader_assets viewmage)

# Bundled fonts next to the executable
add_custom_command(TARGET viewmage POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets/fonts
        $<TARGET_FILE_DIR:viewmage>/fonts)
