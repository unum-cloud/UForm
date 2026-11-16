# The install rules, the export sets, the CMake and pkg-config package files, and the preview SDK archive.
#
# The root names what installs before including this file, as each project ships its own set of libraries:
#
# - `uform_install_sdk_targets_`: the libraries of the `sdk` component, exported as `uformTargets`;
# - `uform_install_static_targets_`: the archives of the `static` component, exported as `uformStaticTargets`;
# - `uform_install_programs_`: the executables the `sdk` component ships beside its libraries, exported by neither.
#
# An empty list installs nothing for its component, and `cmake/uformConfig.cmake.in` loads the export sets the prefix
# holds. A component no list covers, like a header-only core, installs from the root after this file.

if (UFORM_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    set(uform_config_install_dir_ "${CMAKE_INSTALL_LIBDIR}/cmake/uform")
    if (uform_install_sdk_targets_)
        install(
            TARGETS ${uform_install_sdk_targets_}
            EXPORT uformTargets
            COMPONENT sdk
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
        install(
            EXPORT uformTargets
            NAMESPACE uform::
            DESTINATION ${uform_config_install_dir_}
            COMPONENT sdk
        )
    endif ()
    if (uform_install_static_targets_)
        install(
            TARGETS ${uform_install_static_targets_}
            EXPORT uformStaticTargets
            COMPONENT static
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
        install(
            EXPORT uformStaticTargets
            NAMESPACE uform::
            DESTINATION ${uform_config_install_dir_}
            COMPONENT static
        )
    endif ()
    if (uform_install_programs_)
        install(
            TARGETS ${uform_install_programs_}
            COMPONENT sdk
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif ()

    # A major version is the only break the C interface admits, so a consumer asking for one accepts its later minors.
    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/uformConfig.cmake.in" "${PROJECT_BINARY_DIR}/uformConfig.cmake"
        INSTALL_DESTINATION ${uform_config_install_dir_}
    )
    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/uformConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES "${PROJECT_BINARY_DIR}/uformConfig.cmake" "${PROJECT_BINARY_DIR}/uformConfigVersion.cmake"
        DESTINATION ${uform_config_install_dir_}
        COMPONENT sdk
    )
    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/c/uform"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT sdk
    )
    if (IS_DIRECTORY "${PROJECT_SOURCE_DIR}/cpp/uform")
        install(
            DIRECTORY "${PROJECT_SOURCE_DIR}/cpp/uform"
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            COMPONENT sdk
        )
    endif ()
    install(
        FILES "${PROJECT_SOURCE_DIR}/LICENSE"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/uform
        COMPONENT sdk
    )

    # A `.pc` naming the shared library, for the consumers CMake never reaches, wherever the `sdk` installs one.
    # Spelling the prefix against `${pcfiledir}` keeps it correct after `cmake --install --prefix` moves the tree.
    # `file(RELATIVE_PATH)` ends a path to an ancestor with a slash, which the `.` closes.
    # `GNUInstallDirs` returns a relative path unless a distribution forced an absolute one, and only the former may
    # follow a prefix.
    if (uform_shared IN_LIST uform_install_sdk_targets_)
        file(RELATIVE_PATH uform_pc_to_prefix_ "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
        set(uform_pc_prefix_ "\${pcfiledir}/${uform_pc_to_prefix_}.")
        if (IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
            set(uform_pc_libdir_ "${CMAKE_INSTALL_LIBDIR}")
        else ()
            set(uform_pc_libdir_ "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
        endif ()
        if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
            set(uform_pc_includedir_ "${CMAKE_INSTALL_INCLUDEDIR}")
        else ()
            set(uform_pc_includedir_ "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
        endif ()
        configure_file("${PROJECT_SOURCE_DIR}/cmake/uform.pc.in" "${PROJECT_BINARY_DIR}/uform.pc" @ONLY)
        install(
            FILES "${PROJECT_BINARY_DIR}/uform.pc"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
            COMPONENT sdk
        )
    endif ()
endif ()

# The preview SDK: `cpack --config <build>/CPackConfig.cmake` archives the `sdk` component alone, stripped, under one
# top-level directory, as `uform-<version>-<platform>-sdk.tar.gz`, or a `.zip` on Windows.
# A parent project packages itself.
if (UFORM_INSTALL AND PROJECT_IS_TOP_LEVEL)
    set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
    if (WIN32)
        set(CPACK_GENERATOR ZIP)
    else ()
        set(CPACK_GENERATOR TGZ)
    endif ()
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY ON)
    set(CPACK_COMPONENTS_ALL sdk)
    set(CPACK_STRIP_FILES ON)
    include(CPack)
endif ()
