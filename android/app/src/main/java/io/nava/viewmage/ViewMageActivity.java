package io.nava.viewmage;

import io.nava.appshell.AppShellActivity;

/**
 * ViewMage's entire Java surface.
 *
 * <p>Everything the application does is C++; this exists for one reason, and
 * it is the static block below. A {@link android.app.NativeActivity} brings
 * the native library up with {@code dlopen()} from its own native code, which
 * never tells the Java runtime about it — so the JVM will not resolve the
 * {@code native} methods declared on {@link AppShellActivity} against it, and
 * every one of them throws {@code UnsatisfiedLinkError} even though the
 * symbols are present in the mapped {@code .so}. For ViewMage that would mean
 * {@code readIntentData()} never reaching C++, and every launch showing
 * "No image to show".
 *
 * <p>The second load costs nothing: the runtime refcounts and simply registers
 * what is already there.
 *
 * <p>The name must match {@code add_library(viewmage ...)} in CMakeLists.txt
 * and the {@code android.app.lib_name} meta-data in the manifest.
 */
public class ViewMageActivity extends AppShellActivity {
    static { System.loadLibrary("viewmage"); }
}
