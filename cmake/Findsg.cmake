include(FindPackageHandleStandardArgs)

if (DEFINED sg_PATH)
    set(sg_include_path ${sg_PATH}/include)
    set(sg_lib_path ${sg_PATH}/lib)
endif()

find_path(
    sg_INCLUDE_DIR
    NAMES bmlib_runtime.h
    HINTS
    /opt/sophon/libsophon-current/include
    ${sg_include_path})
find_library(
    bmlib_LIBRARY
    NAMES bmlib
    HINTS
    /opt/sophon/libsophon-current/lib
    ${sg_lib_path})
find_library(
    bmrt_LIBRARY
    NAMES bmrt
    HINTS
    /opt/sophon/libsophon-current/lib
    ${sg_lib_path})

find_package_handle_standard_args(
    sg
    REQUIRED_VARS sg_INCLUDE_DIR bmlib_LIBRARY bmrt_LIBRARY)

if (sg_FOUND)
    foreach (lib bmlib bmrt)
        add_library(sg::${lib} IMPORTED SHARED)
        set_target_properties(
            sg::${lib} PROPERTIES
            IMPORTED_NO_SONAME TRUE
            INTERFACE_INCLUDE_DIRECTORIES ${sg_INCLUDE_DIR}
            IMPORTED_LOCATION ${${lib}_LIBRARY})
        list(APPEND sg_LIBRARIES sg::${lib})
    endforeach()
endif()
