# ---------------------------------------------------------------------------
# viewmage_core — portable static lib: image decode (JXL + DNG develop),
# post-processing, denoise, view transform. Zero OS headers, zero Vulkan.
# ---------------------------------------------------------------------------

add_library(viewmage_core STATIC
    src/view_transform.cc
    src/jxl_image.cc
    src/dng_image.cc
    src/decoded_image.cc
    src/ai_denoise.cc
    src/enhance.cc
    src/export_math.cc
)

target_include_directories(viewmage_core PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../framework/vk_canvas/core
)

target_compile_features(viewmage_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(viewmage_core PRIVATE /W3 /utf-8)
else()
    target_compile_options(viewmage_core PRIVATE -Wall -O2)
endif()

# libjxl v0.12: decoder + thread pool + colour management. jxl_dec does NOT
# link jxl_cms (see lib/jxl.cmake:226-229 vs :219), so both must be listed.
target_link_libraries(viewmage_core PUBLIC jxl_dec jxl_threads jxl_cms)

# ONNX Runtime (optional): the neural raw denoiser (ai_denoise.cc). When
# absent the class still exists and load() fails with a readable message, so
# the develop's fallback path is exercised for free.
if(VIEWMAGE_HAS_ORT)
    target_link_libraries(viewmage_core PRIVATE onnxruntime)
    target_compile_definitions(viewmage_core PRIVATE VIEWMAGE_WITH_ORT)
endif()

# ---------------------------------------------------------------------------
# Debug-only assert-based tests, same convention as Matrix_Player.
# ---------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # view_transform_test: pure math, no deps at all.
    add_executable(view_transform_test
        tests/view_transform_test.cc
        src/view_transform.cc)
    target_include_directories(view_transform_test PRIVATE include)
    if(NOT MSVC)
        target_compile_options(view_transform_test PRIVATE -Wall)
    endif()

    # dng_image_test: builds its own DNG fixture in memory. Needs
    # decoded_image.cc and ai_denoise.cc but NO libjxl and NO Vulkan.
    add_executable(dng_image_test
        tests/dng_image_test.cc
        src/dng_image.cc
        src/decoded_image.cc
        src/ai_denoise.cc)
    target_include_directories(dng_image_test PRIVATE
        include
        ${CMAKE_CURRENT_SOURCE_DIR}/../framework/vk_canvas/core)
    if(NOT MSVC)
        target_compile_options(dng_image_test PRIVATE -Wall)
    endif()

    # jxl_image_test: needs system or submodule libjxl (uses the submodule
    # here, since it is already configured by the root CMakeLists.txt).
    add_executable(jxl_image_test
        tests/jxl_image_test.cc
        src/jxl_image.cc
        src/decoded_image.cc)
    target_include_directories(jxl_image_test PRIVATE
        include
        ${CMAKE_CURRENT_SOURCE_DIR}/../framework/vk_canvas/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/libjxl/lib/include
        ${CMAKE_BINARY_DIR}/libjxl/lib/include)
    target_link_libraries(jxl_image_test PRIVATE jxl_dec jxl_threads jxl_cms)
    if(NOT MSVC)
        target_compile_options(jxl_image_test PRIVATE -Wall)
    endif()
endif()
