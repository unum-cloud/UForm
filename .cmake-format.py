with section("parse"):
    # Teach cmake-format the project's helper signatures so it groups each keyword with its value
    # instead of exploding every token onto its own line (which would otherwise force a
    # `cmake-format: off` guard — itself banned).
    additional_commands = {
        "uform_dependency": {
            "kwargs": {
                "NAME": 1,
                "MODE": 1,
                "ROOT_VAR": 1,
                "INCLUDE_VAR": 1,
                "REPO": 1,
                "TAG": 1,
            },
        },
    }

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

with section("markup"):
    # Never reflow hand-authored comment blocks: the file-level docstring carries multi-line apt / uv
    # / cmake usage that a reflow would restitch into an unreadable run. Disabling markup keeps every
    # comment verbatim, so the docstring survives without a (banned) `cmake-format: off` guard.
    enable_markup = False
    first_comment_is_literal = True
