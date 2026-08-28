// android_main.cc — the whole of ViewMage's platform bootstrap.
//
// Copyright (C) 2026 nava. AGPLv3 or later; see LICENSE.
//
// app_shell owns everything a program needs before it exists; what is left for
// an application to write is this. Read the bytes we were launched to show,
// build the host and the app, run.
#include <android_native_app_glue.h>

#include <memory>
#include <utility>

#include "android_host.hh"
#include "launch_intent.hh"
#include "log.hh"
#include "viewmage_app.hh"

void android_main(android_app* state) {
    // requestAllFilesAccess = false. ViewMage never walks the filesystem: it is
    // handed one document by whatever launched it, through a ContentResolver
    // read that needs no permission of ours. Asking for MANAGE_EXTERNAL_STORAGE
    // to open a single file the user explicitly picked would be a Settings
    // screen shown for nothing — and the manifest does not carry the permission,
    // so it would be a toggle that cannot even be pressed.
    auto host = std::make_unique<AndroidHost>(state,
                                              /*launchExtraKey=*/nullptr,
                                              /*fallback=*/nullptr,
                                              /*requestAllFilesAccess=*/false);

    ViewMageApp app(std::move(host));

    // Read BEFORE create(): the activity is alive and its intent is readable
    // from the moment android_main starts, and doing it here keeps the app's
    // create() free of any Android type at all.
    std::vector<uint8_t> bytes = read_intent_data_bytes(state);
    VCE_LOGI("ViewMage", "launched with %zu bytes", bytes.size());
    app.setSource(std::move(bytes));

    if (!app.create()) {
        VCE_LOGE("ViewMage", "create() failed; nothing to run");
        return;
    }
    app.run();
}
