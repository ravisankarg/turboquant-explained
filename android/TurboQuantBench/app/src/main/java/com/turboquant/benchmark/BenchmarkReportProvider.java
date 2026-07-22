package com.turboquant.benchmark;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.database.Cursor;
import android.net.Uri;
import android.os.Binder;
import android.os.ParcelFileDescriptor;
import android.os.Process;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Shell-only read endpoint for benchmark JSON reports.
 *
 * Release APKs are intentionally non-debuggable, so adb run-as cannot read
 * filesDir. This provider keeps that release boundary and gives adb shell a
 * read-only pipe to the already-cached report without scraping the UI.
 */
public final class BenchmarkReportProvider extends ContentProvider {
    static final String AUTHORITY = "com.turboquant.benchmark.reports";

    @Override
    public boolean onCreate() {
        return true;
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        requireShellCaller();
        if (mode == null || !mode.startsWith("r")) {
            throw new FileNotFoundException("read-only report endpoint");
        }
        String benchmarkMode = uri == null ? null : uri.getQueryParameter("mode");
        if (benchmarkMode == null && uri != null) {
            benchmarkMode = uri.getLastPathSegment();
        }
        String report;
        report = BenchmarkService.cachedReport(getContext(), benchmarkMode);
        if (report == null) {
            throw new FileNotFoundException("no cached report for mode: " + benchmarkMode);
        }

        final ParcelFileDescriptor[] pipe;
        try {
            pipe = ParcelFileDescriptor.createPipe();
        } catch (IOException e) {
            FileNotFoundException failure = new FileNotFoundException("could not create report pipe");
            failure.initCause(e);
            throw failure;
        }
        final byte[] bytes = report.getBytes(StandardCharsets.UTF_8);
        Thread writer = new Thread(() -> {
            try (OutputStream out = new ParcelFileDescriptor.AutoCloseOutputStream(pipe[1])) {
                out.write(bytes);
            } catch (IOException ignored) {
                // The adb reader may close the pipe after extracting KPIs.
            }
        }, "benchmark-report-export");
        writer.start();
        return pipe[0];
    }

    private static void requireShellCaller() {
        int caller = Binder.getCallingUid();
        if (caller != Process.SHELL_UID && caller != Process.myUid()) {
            throw new SecurityException("benchmark provider is readable only by adb shell");
        }
    }

    @Override
    public String getType(Uri uri) {
        return "application/json";
    }

    @Override
    public Cursor query(Uri uri, String[] projection, String selection,
                        String[] selectionArgs, String sortOrder) {
        throw new UnsupportedOperationException("use content read with openFile");
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) {
        throw new UnsupportedOperationException("read-only provider");
    }

    @Override
    public int delete(Uri uri, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException("read-only provider");
    }

    @Override
    public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
        throw new UnsupportedOperationException("read-only provider");
    }
}
