# ── ViewMage ─────────────────────────────────────────────────────────────────
#
# One shared library, `viewmage`, loaded by a NativeActivity. Everything below
# is assembling other people's builds:
#
#   vk_canvas_core     draws it        (framework/, first party, ours)
#   app_shell(_android) makes it a program (framework/, first party, ours)
#   jxl_dec            turns bytes into pixels (thirdparty/, NOT ours)
#
# Reached from app/build.gradle's externalNativeBuild.
cmake_minimum_required(VERSION 3.22.1)
project(viewmage LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(REPO_ROOT     ${CMAKE_CURRENT_SOURCE_DIR})
set(APP_SHELL_DIR ${REPO_ROOT}/framework/app_shell)
set(VK_CANVAS_DIR ${REPO_ROOT}/framework/vk_canvas CACHE PATH "vk_canvas root")
set(LIBJXL_DIR    ${REPO_ROOT}/thirdparty/libjxl)

if(NOT EXISTS ${LIBJXL_DIR}/CMakeLists.txt)
    message(FATAL_ERROR
        "thirdparty/libjxl is empty. Run: git submodule update --init --recursive")
endif()

# ── libjxl (third party — configured, never modified) ────────────────────────
#
# Decoder only. ViewMage never encodes, never transcodes a JPEG, ships no
# command-line tools and runs no benchmarks, and every one of those pulls in
# dependencies (libpng, zlib, libjpeg-turbo, gtest) that would otherwise have
# to be cloned and cross-compiled for a phone.
#
# These are set as cache variables from here rather than patched into libjxl's
# own CMakeLists, which is the whole point of the thirdparty/ rule: the
# submodule stays byte-identical to upstream v0.12.0, so it can be updated by
# moving the pin and nothing else.
set(BUILD_TESTING                 OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS             OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_TOOLS           OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_VIEWERS         OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_EXAMPLES        OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_MANPAGES        OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_BENCHMARK       OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_DEVTOOLS        OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_FUZZERS         OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_JNI             OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_PLUGINS         OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_OPENEXR         OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_SJPEG           OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_TCMALLOC        OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_COVERAGE        OFF CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_SKCMS           ON  CACHE BOOL "" FORCE)
set(JPEGXL_BUNDLE_SKCMS           ON  CACHE BOOL "" FORCE)
set(JPEGXL_ENABLE_TRANSCODE_JPEG  OFF CACHE BOOL "" FORCE)
set(JPEGXL_STATIC                 OFF CACHE BOOL "" FORCE)
set(JPEGXL_WARNINGS_AS_ERRORS     OFF CACHE BOOL "" FORCE)
# Highway is libjxl's SIMD layer and builds its own tests/examples otherwise.
set(HWY_ENABLE_TESTS              OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES           OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB            OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL            OFF CACHE BOOL "" FORCE)

# libjxl is ALWAYS optimised, including in a Debug build.
#
# Gradle's debug variant configures CMake with CMAKE_BUILD_TYPE=Debug, which
# hands -O0 to everything in the tree. For a SIMD codec that is not a slower
# build, it is a different program: highway's vector code depends on the
# optimiser to become vector code at all. Measured on an S23 Ultra, a 3060x4080
# photo took 19 SECONDS to decode at -O0 — long enough to look like a hang, and
# long enough to be blamed on the decoder rather than on the flags.
#
# Debugging OUR code is why the Debug variant exists; nobody steps through
# libjxl, and it is third party so we may not annotate its targets. Setting the
# Debug flags around the add_subdirectory() is the seam that needs no consent
# from the submodule: directory properties are captured when it is added, so
# this applies to libjxl and its nested highway/brotli and to nothing else.
set(_saved_c_debug   "${CMAKE_C_FLAGS_DEBUG}")
set(_saved_cxx_debug "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2")

add_subdirectory(${LIBJXL_DIR} ${CMAKE_BINARY_DIR}/libjxl EXCLUDE_FROM_ALL)

set(CMAKE_C_FLAGS_DEBUG   "${_saved_c_debug}")
set(CMAKE_CXX_FLAGS_DEBUG "${_saved_cxx_debug}")

# ── vk_canvas ────────────────────────────────────────────────────────────────
add_subdirectory(${VK_CANVAS_DIR}/core ${CMAKE_BINARY_DIR}/vk_canvas_core)

# ── app_shell ────────────────────────────────────────────────────────────────
set(APP_SHELL_APP_NAME "viewmage" CACHE STRING "" FORCE)
set(APP_SHELL_WINDOW_TITLE "ViewMage" CACHE STRING "" FORCE)
set(APP_SHELL_APP_ID "io.nava.viewmage" CACHE STRING "" FORCE)
set(APP_SHELL_NATIVE_APP_GLUE_DIR ${ANDROID_NDK}/sources/android/native_app_glue)

# app_shell compiles ui_metrics.cc unconditionally, and its header includes a
# build-generated floor derived from THE APP'S OWN FONTS. ViewMage has none:
# it draws one line of text, in the font engine's built-in stroke fallback,
# and never constructs a UiMetrics at all. The real generator also cannot run
# under a cross-compiler (see AppShellMinTextSize.cmake), so rather than build
# a desktop tree to compute a number nothing reads, we state the situation.
set(_gen_dir ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${_gen_dir})
file(WRITE ${_gen_dir}/ui_min_text_size.gen.h
"#pragma once\n"
"// GENERATED by ViewMage's CMakeLists.txt — deliberately NOT by\n"
"// vulkan_font_engine's min_text_size tool.\n"
"//\n"
"// That tool derives a legibility floor from the outlines of the faces an app\n"
"// ships. ViewMage ships no faces: its only text is one error line drawn with\n"
"// the stroke-fallback glyphs built into the font engine, and it never\n"
"// constructs a UiMetrics. There is no font geometry here to measure, so this\n"
"// is a placeholder that keeps ui_metrics.cc compiling rather than a computed\n"
"// value. An app that starts shipping a font must switch to the real recipe.\n"
"static constexpr float kMinReadableTextSizePx = 12.0f;\n")

add_subdirectory(${APP_SHELL_DIR} ${CMAKE_BINARY_DIR}/app_shell)

# ── Shaders ──────────────────────────────────────────────────────────────────
#
# Compiled into the Gradle module's assets/ so AGP packages them. ViewMage uses
# the CURVE path (overlay + tiling/coverage/composite) for its one line of
# fallback text and the IMAGE path for the photograph. It never calls
# useShapes() and never calls initMsdf(), so shape_* and msdf_* are genuinely
# not needed and are left out rather than compiled and shipped unused.
include(${VK_CANVAS_DIR}/cmake/VceShaders.cmake)
set(SHADER_OUT ${CMAKE_CURRENT_SOURCE_DIR}/app/src/main/assets/shaders)

vce_compile_slang(compile_font_shaders ${SHADER_OUT}
    ${VK_CANVAS_DIR}/first_party/vulkan_font_engine/shaders_src
    composite_vert composite_frag tiling coverage)

vce_compile_slang(compile_canvas_shaders ${SHADER_OUT}
    ${VK_CANVAS_DIR}/shaders_src
    overlay_vert overlay_frag image_vert image_frag)

# ── The application ──────────────────────────────────────────────────────────
add_library(viewmage SHARED
    src/android_main.cc
    src/viewmage_app.cc
    src/view_transform.cc
    src/jxl_image.cc

    # vk_canvas's Android implementations of the platform.hh seams
    # (AndroidSurfaceProvider / AndroidAssetReader). They live in that repo's
    # platform/ layer, which neither vk_canvas_core nor app_shell_android
    # compiles — core may not include a platform SDK header, and app_shell only
    # USES these classes. So the consumer compiles them, which is exactly the
    # arrangement native_app_glue below already has.
    ${VK_CANVAS_DIR}/platform/android/android_platform.cc

    ${APP_SHELL_NATIVE_APP_GLUE_DIR}/android_native_app_glue.c
)

add_dependencies(viewmage compile_font_shaders compile_canvas_shaders)

target_include_directories(viewmage PRIVATE
    ${REPO_ROOT}/src
    ${APP_SHELL_NATIVE_APP_GLUE_DIR}
    ${VK_CANVAS_DIR}/platform/android      # android_platform.hh, jni_util.hh
    # libjxl's public headers. Stated here rather than inherited from
    # jxl_dec-static, which is assembled from an object library and carries no
    # usable INTERFACE include directories. The second path is the BUILD tree:
    # jxl/version.h and jxl/jxl_export.h are generated, so they are not in the
    # submodule and must never be looked for there.
    ${LIBJXL_DIR}/lib/include
    ${CMAKE_BINARY_DIR}/libjxl/lib/include
)

target_compile_options(viewmage PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)

target_link_libraries(viewmage PRIVATE
    app_shell_android
    vk_canvas_core
    # libjxl v0.12's decoder and its thread pool. Not "-static": with
    # BUILD_SHARED_LIBS OFF above, these target names ARE the static archives,
    # and the -static suffixed names belong to an older layout.
    jxl_dec
    jxl_threads
    android
    log
    vulkan
)

# native_app_glue's entry point is only ever called by the framework, so the
# linker cannot see that anything uses it.
target_link_options(viewmage PRIVATE -u ANativeActivity_onCreate)
