package com.turboquant.benchmark;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class BenchmarkService extends Service {
    static final String ACTION_START = "com.turboquant.benchmark.action.START_BENCHMARK";
    static final String EXTRA_DATASETS_JSON = "datasets_json";
    static final String EXTRA_OUTPUT_DIR = "output_dir";
    static final String EXTRA_MODE = "mode";
    static final String MODE_QUERY_MAJOR = "query_major";
    static final String MODE_BATCHED = "batched_throughput";
    static final String MODE_HNSW_QUERY_MAJOR = "hnsw_query_major";
    static final String MODE_HNSW_BATCHED = "hnsw_batched_throughput";
    static final String FAMILY_FLAT = "FlatIndex";
    static final String FAMILY_HNSW = "HNSW";

    private static final String CHANNEL_ID = "turboquant_benchmarks";
    private static final int NOTIFICATION_ID = 4201;
    // Invalidate the previous reports when search semantics change: Flat B
    // now exact-reranks 4-bit candidates, and HNSW uses the recall-tuned
    // efSearch value reported by the native benchmark.
    private static final String FLAT_CACHE_VERSION = "v5";
    private static final String HNSW_CACHE_VERSION = "v4";
    private static final String LAST_MODE_PREFS = "benchmark_cache_state";
    private static final String LAST_MODE_KEY = "last_mode";
    private static final Object LOCK = new Object();
    private static State state = State.idle();

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private PowerManager.WakeLock wakeLock;

    @Override
    public void onCreate() {
        super.onCreate();
        createChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null || !ACTION_START.equals(intent.getAction())) {
            stopSelf(startId);
            return START_NOT_STICKY;
        }
        String datasetsJson = intent.getStringExtra(EXTRA_DATASETS_JSON);
        String outputDir = intent.getStringExtra(EXTRA_OUTPUT_DIR);
        String mode = normalizeMode(intent.getStringExtra(EXTRA_MODE));
        if (outputDir == null || mode == null) {
            stopSelf(startId);
            return START_NOT_STICKY;
        }
        synchronized (LOCK) {
            if (state.running) {
                return START_STICKY;
            }
            state = State.running("Starting " + modeLabel(mode), mode);
        }
        String cached = readCachedReport(this, mode);
        if (cached != null) {
            synchronized (LOCK) {
                state = State.done("Showing cached " + modeLabel(mode) + " result", cached, null, mode, true);
            }
            stopSelf(startId);
            return START_NOT_STICKY;
        }
        if (datasetsJson == null) {
            synchronized (LOCK) {
                state = State.done("Benchmark failed", null, "datasets are not downloaded", mode, false);
            }
            stopSelf(startId);
            return START_NOT_STICKY;
        }
        startForeground(NOTIFICATION_ID, notification("TurboQuant benchmark running", true));
        worker.execute(() -> runBenchmark(startId, datasetsJson, outputDir, mode));
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        worker.shutdownNow();
        releaseWakeLock();
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    static Snapshot snapshot(Context context) {
        synchronized (LOCK) {
            String result = state.resultJson;
            boolean done = state.done;
            String mode = state.mode;
            boolean cached = state.cached;
            if (result == null && !state.running && state.error == null) {
                mode = context.getSharedPreferences(LAST_MODE_PREFS, MODE_PRIVATE)
                        .getString(LAST_MODE_KEY, MODE_QUERY_MAJOR);
                result = readCachedReport(context, mode);
                done = result != null;
                cached = result != null;
            }
            String status = state.status;
            if (done && status == null) {
                status = cached ? "Showing cached benchmark result" : "Benchmark complete";
            }
            return new Snapshot(state.running, done, status, result, state.error, mode, cached);
        }
    }

    static boolean hasCached(Context context, String mode) {
        return readCachedReport(context, normalizeMode(mode)) != null;
    }

    static String cachedReport(Context context, String mode) {
        return readCachedReport(context, normalizeMode(mode));
    }

    static String modeLabel(String mode) {
        if (MODE_HNSW_QUERY_MAJOR.equals(mode)) {
            return "A) HNSW real traffic / query-major";
        }
        if (MODE_HNSW_BATCHED.equals(mode)) {
            return "B) HNSW batched throughput";
        }
        if (MODE_BATCHED.equals(mode)) {
            return "B) Batched throughput";
        }
        return "A) Real traffic / query-major";
    }

    static String familyForMode(String mode) {
        return isHnswMode(mode) ? FAMILY_HNSW : FAMILY_FLAT;
    }

    static boolean isHnswMode(String mode) {
        return MODE_HNSW_QUERY_MAJOR.equals(mode) || MODE_HNSW_BATCHED.equals(mode);
    }

    static String pairedMode(String mode) {
        if (MODE_QUERY_MAJOR.equals(mode)) {
            return MODE_BATCHED;
        }
        if (MODE_BATCHED.equals(mode)) {
            return MODE_QUERY_MAJOR;
        }
        if (MODE_HNSW_QUERY_MAJOR.equals(mode)) {
            return MODE_HNSW_BATCHED;
        }
        return MODE_HNSW_QUERY_MAJOR;
    }

    private static String normalizeMode(String mode) {
        if (MODE_BATCHED.equals(mode)) {
            return MODE_BATCHED;
        }
        if (MODE_HNSW_QUERY_MAJOR.equals(mode)) {
            return MODE_HNSW_QUERY_MAJOR;
        }
        if (MODE_HNSW_BATCHED.equals(mode)) {
            return MODE_HNSW_BATCHED;
        }
        if (MODE_QUERY_MAJOR.equals(mode) || mode == null) {
            return MODE_QUERY_MAJOR;
        }
        return null;
    }

    private static File cacheFile(Context context, String mode) {
        String version = isHnswMode(mode) ? HNSW_CACHE_VERSION : FLAT_CACHE_VERSION;
        return new File(context.getFilesDir(), "benchmark_" + mode + "_" + version + ".json");
    }

    private static String readCachedReport(Context context, String mode) {
        if (mode == null) {
            return null;
        }
        File report = cacheFile(context, mode);
        if (!report.isFile() || report.length() == 0L) {
            return null;
        }
        try {
            String result = new String(java.nio.file.Files.readAllBytes(report.toPath()), StandardCharsets.UTF_8);
            return result.startsWith("{") ? result : null;
        } catch (Exception ignored) {
            return null;
        }
    }

    private void runBenchmark(int startId, String datasetsJson, String outputDir, String mode) {
        acquireWakeLock();
        updateNotification("Running " + modeLabel(mode));
        try {
            String json;
            if (MODE_HNSW_QUERY_MAJOR.equals(mode)) {
                json = NativeBench.runHnswQueryMajor(datasetsJson, outputDir);
            } else if (MODE_HNSW_BATCHED.equals(mode)) {
                json = NativeBench.runHnswBatched(datasetsJson, outputDir);
            } else if (MODE_BATCHED.equals(mode)) {
                json = NativeBench.runBatched(datasetsJson, outputDir);
            } else {
                json = NativeBench.runQueryMajor(datasetsJson, outputDir);
            }
            if (json == null || !json.startsWith("{")) {
                throw new IllegalStateException(json == null ? "native benchmark returned no report" : json);
            }
            File report = cacheFile(this, mode);
            File temp = new File(report.getPath() + ".part");
            try (FileOutputStream out = new FileOutputStream(temp, false)) {
                out.write(json.getBytes(StandardCharsets.UTF_8));
            }
            if (!temp.renameTo(report)) {
                throw new IllegalStateException("could not commit cached benchmark report");
            }
            getSharedPreferences(LAST_MODE_PREFS, MODE_PRIVATE)
                    .edit()
                    .putString(LAST_MODE_KEY, mode)
                    .apply();
            synchronized (LOCK) {
                state = State.done("Benchmark complete", json, null, mode, false);
            }
            updateNotification(modeLabel(mode) + " complete");
        } catch (Throwable t) {
            synchronized (LOCK) {
                state = State.done("Benchmark failed", null, t.getMessage(), mode, false);
            }
            updateNotification(modeLabel(mode) + " failed");
        } finally {
            releaseWakeLock();
            stopSelf(startId);
        }
    }

    private void acquireWakeLock() {
        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "TurboQuantBench:Benchmark");
        wakeLock.setReferenceCounted(false);
        wakeLock.acquire();
    }

    private void releaseWakeLock() {
        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
        }
        wakeLock = null;
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT < 26) {
            return;
        }
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "TurboQuant benchmarks",
                NotificationManager.IMPORTANCE_LOW);
        channel.setDescription("Long-running vector benchmark jobs");
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(channel);
    }

    private Notification notification(String title, boolean ongoing) {
        Intent launch = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this, 0, launch, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        Notification.Builder builder = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        return builder
                .setSmallIcon(android.R.drawable.stat_notify_sync)
                .setContentTitle(title)
                .setContentText("Benchmark continues if the app is backgrounded or the screen is off")
                .setContentIntent(pendingIntent)
                .setOngoing(ongoing)
                .setOnlyAlertOnce(true)
                .setProgress(0, 0, ongoing)
                .build();
    }

    private void updateNotification(String title) {
        NotificationManager manager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        manager.notify(NOTIFICATION_ID, notification(title, state.running));
    }

    private static final class State {
        final boolean running;
        final boolean done;
        final String status;
        final String resultJson;
        final String error;

        final String mode;
        final boolean cached;

        State(boolean running, boolean done, String status, String resultJson, String error,
              String mode, boolean cached) {
            this.running = running;
            this.done = done;
            this.status = status;
            this.resultJson = resultJson;
            this.error = error;
            this.mode = mode;
            this.cached = cached;
        }

        static State idle() {
            return new State(false, false, null, null, null, null, false);
        }

        static State running(String status, String mode) {
            return new State(true, false, status, null, null, mode, false);
        }

        static State done(String status, String resultJson, String error, String mode, boolean cached) {
            return new State(false, true, status, resultJson, error, mode, cached);
        }
    }

    static final class Snapshot {
        final boolean running;
        final boolean done;
        final String status;
        final String resultJson;
        final String error;
        final String mode;
        final boolean cached;

        Snapshot(boolean running, boolean done, String status, String resultJson, String error,
                 String mode, boolean cached) {
            this.running = running;
            this.done = done;
            this.status = status;
            this.resultJson = resultJson;
            this.error = error;
            this.mode = mode;
            this.cached = cached;
        }
    }
}
