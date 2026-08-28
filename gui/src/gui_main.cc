// gui_main.cc — ViewMage's desktop entry point.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// The Wayland bootstrap owns main(); it sets up DPI awareness, the crash
// handler, the log file, and the timer resolution, then calls
// app_shell_main() once all that is done. This is the app's half.
#include "viewmage_app.hh"

#include "host.hh"
#include "log.hh"

int app_shell_main(int argc, char** argv) {
    VCE_LOGI("ViewMage", "desktop build");

    auto host = make_host();
    if (!host) {
        fprintf(stderr, "ViewMage: failed to create a host\n");
        return 1;
    }

    ViewMageApp app(std::move(host));

    // On desktop there is no Intent — the file to open comes from argv[1].
    if (argc >= 2) {
        std::vector<uint8_t> bytes;
        FILE* f = fopen(argv[1], "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            bytes.resize((size_t)ftell(f));
            fseek(f, 0, SEEK_SET);
            if (!bytes.empty())
                fread(bytes.data(), 1, bytes.size(), f);
            fclose(f);
        }
        if (bytes.empty()) {
            fprintf(stderr, "ViewMage: could not read '%s'\n", argv[1]);
            return 1;
        }
        VCE_LOGI("ViewMage", "opened %zu bytes from %s", bytes.size(), argv[1]);
        app.setSource(std::move(bytes));
    }

    if (!app.create()) {
        fprintf(stderr, "ViewMage: create() failed\n");
        return 1;
    }
    app.run();
    return 0;
}
