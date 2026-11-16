# Options affecting formatting.
with section("format"):
    # How wide to allow formatted cmake files
    line_width = 120

    # How many spaces to tab for indent
    tab_size = 4

    # If true, separate flow control names from their parentheses with a space
    separate_ctrl_name_with_space = True

    # If true, separate function names from parentheses with a space
    separate_fn_name_with_space = False

    # If a statement is wrapped to more than one line, than dangle the closing
    # parenthesis on its own line.
    dangle_parens = True

# The project's own functions, so a keyword stays on the line of the items it brings.
with section("parse"):
    additional_commands = {
        "uform_cpu_class": {"pargs": "2+", "kwargs": {"OVER": 1, "FORKUNION": "*"}},
        "uform_find_infiniband": {"pargs": 1},
        "uform_find_library": {"pargs": 2, "kwargs": {"EXTRA_NAMES": "*", "EXTRA_HINTS": "*"}},
        "uform_gpu_bundle": {"pargs": 4, "kwargs": {"CODES": "*"}},
        "uform_gpu_class": {"pargs": 4},
        "uform_gpu_codes_claimed": {"pargs": "1+"},
        "uform_sibling": {"pargs": 4},
        "uform_variants": {"pargs": "3+"},
    }

with section("markup"):
    # Never reflow hand-authored comment blocks: a file-level docstring can carry multi-line apt,
    # uv or cmake usage that a reflow would restitch into an unreadable run. Disabling markup keeps
    # every comment verbatim, so the docstring survives without a `cmake-format: off` guard.
    enable_markup = False
    first_comment_is_literal = True
