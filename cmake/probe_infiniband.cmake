# Probe the InfiniBand transport — the one native dependency absent on the consumer / workstation cards
# (RTX 3090, RTX PRO Blackwell) the stack must still build on. Every other dependency is available
# everywhere and hard-required, so this one alone gets a tri-state gate via `UFORM_WITH_INFINIBAND`:
#
#   AUTO (default)  probe; enable when libibverbs + libmlx5 (direct verbs) + `infiniband/verbs.h` are all
#                   present, otherwise disable quietly.
#   ON              probe; `FATAL_ERROR` naming the missing piece if the transport cannot be built.
#   OFF             skip the probe; the wire PODs still compile and the collectives fail loudly at runtime.
#
# On success the caller receives `UFORM_INFINIBAND_ENABLED` plus the cached `UFORM_IBVERBS_INCLUDE_DIR` /
# `UFORM_IBVERBS_LIBRARY` / `UFORM_MLX5_LIBRARY` (`find_*` results are cache entries, already global).
#
# Naming: `UFORM_*` are public cache variables, `_uform_*` are internal.

function (uform_probe_infiniband)
    if (UFORM_WITH_INFINIBAND STREQUAL "OFF")
        set(UFORM_INFINIBAND_ENABLED
            OFF
            PARENT_SCOPE
        )
        message(STATUS "uform: InfiniBand transport disabled (UFORM_WITH_INFINIBAND=OFF)")
        return()
    endif ()

    find_path(UFORM_IBVERBS_INCLUDE_DIR infiniband/verbs.h)
    find_library(UFORM_IBVERBS_LIBRARY ibverbs)
    find_library(UFORM_MLX5_LIBRARY mlx5)

    if (UFORM_IBVERBS_INCLUDE_DIR
        AND UFORM_IBVERBS_LIBRARY
        AND UFORM_MLX5_LIBRARY
    )
        set(UFORM_INFINIBAND_ENABLED
            ON
            PARENT_SCOPE
        )
        message(STATUS "uform: InfiniBand transport enabled (${UFORM_IBVERBS_LIBRARY}, ${UFORM_MLX5_LIBRARY})")
    elseif (UFORM_WITH_INFINIBAND STREQUAL "ON")
        message(
            FATAL_ERROR
                "uform: UFORM_WITH_INFINIBAND=ON but the transport cannot be built — "
                "verbs.h=${UFORM_IBVERBS_INCLUDE_DIR} ibverbs=${UFORM_IBVERBS_LIBRARY} mlx5=${UFORM_MLX5_LIBRARY}. "
                "Install the MLNX OFED stack (libibverbs-dev + libmlx5), or pass -DUFORM_WITH_INFINIBAND=OFF."
        )
    else ()
        set(UFORM_INFINIBAND_ENABLED
            OFF
            PARENT_SCOPE
        )
        message(STATUS "uform: InfiniBand transport disabled — libibverbs / libmlx5 not found (AUTO)")
    endif ()
endfunction ()
