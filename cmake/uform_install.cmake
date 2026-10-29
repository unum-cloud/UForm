# The install rules, the export sets, the CMake and pkg-config package files, and the preview SDK archive.

# `find_package(uform)` names the libraries `uform::static` and `uform::shared`. The `sdk` component holds the shared
# library, the C headers, both package files and the license, and the archive installs beside it in a component of its
# own, with an export set the package loads only when present.
if (UFORM_INSTALL AND UFORM_BUILD_SHARED)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    set(uform_config_install_dir_ "${CMAKE_INSTALL_LIBDIR}/cmake/uform")
    install(
        TARGETS uform_header uform_shared
        EXPORT uformTargets
        COMPONENT sdk
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(
        TARGETS uform_static
        EXPORT uformStaticTargets
        COMPONENT static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(
        EXPORT uformTargets
        NAMESPACE uform::
        DESTINATION ${uform_config_install_dir_}
        COMPONENT sdk
    )
    install(
        EXPORT uformStaticTargets
        NAMESPACE uform::
        DESTINATION ${uform_config_install_dir_}
        COMPONENT static
    )
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
        DIRECTORY c/uform
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT sdk
    )
    install(
        FILES LICENSE
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/uform
        COMPONENT sdk
    )

    # A `.pc` for the consumers CMake never reaches. Spelling the prefix against `${pcfiledir}` keeps it correct after
    # the tree moves, and only a relative `GNUInstallDirs` path may follow the prefix.
    file(RELATIVE_PATH uform_pc_relative_prefix_ "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
    set(uform_pc_prefix_ "\${pcfiledir}/${uform_pc_relative_prefix_}")
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
    configure_file(cmake/uform.pc.in "${PROJECT_BINARY_DIR}/uform.pc" @ONLY)
    install(
        FILES "${PROJECT_BINARY_DIR}/uform.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
        COMPONENT sdk
    )

    # `cpack --config <build>/CPackConfig.cmake` packs the `sdk` component alone, stripped.
    set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
    if (WIN32)
        set(CPACK_GENERATOR ZIP)
    else ()
        set(CPACK_GENERATOR TGZ)
    endif ()
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_COMPONENTS_ALL sdk)
    set(CPACK_STRIP_FILES ON)
    include(CPack)
endif ()
