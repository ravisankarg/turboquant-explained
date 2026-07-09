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

    private static final String CHANNEL_ID = "turboquant_benchmarks";
    private static final int NOTIFICATION_ID = 4201;
    private static final String REPORT_FILE = "last_benchmark_report.json";
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
        if (datasetsJson == null || outputDir == null) {
            stopSelf(startId);
            return START_NOT_STICKY;
        }
        synchronized (LOCK) {
            if (state.running) {
                return START_STICKY;
            }
            state = State.running("Benchmark running in background");
        }
        startForeground(NOTIFICATION_ID, notification("TurboQuant benchmark running", true));
        worker.execute(() -> runBenchmark(startId, datasetsJson, outputDir));
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
            if (result == null && state.done) {
                File report = new File(context.getFilesDir(), REPORT_FILE);
                if (report.isFile()) {
                    try {
                        result = new String(java.nio.file.Files.readAllBytes(report.toPath()), StandardCharsets.UTF_8);
                    } catch (Exception ignored) {
                        result = null;
                    }
                }
            }
            return new Snapshot(state.running, state.done, state.status, result, state.error);
        }
    }

    private void runBenchmark(int startId, String datasetsJson, String outputDir) {
        acquireWakeLock();
        updateNotification("TurboQuant benchmark running");
        try {
            String json = NativeBench.runBenchmark(datasetsJson, outputDir);
            File report = new File(getFilesDir(), REPORT_FILE);
            try (FileOutputStream out = new FileOutputStream(report, false)) {
                out.write(json.getBytes(StandardCharsets.UTF_8));
            }
            synchronized (LOCK) {
                state = State.done("Benchmark complete", json, null);
            }
            updateNotification("TurboQuant benchmark complete");
        } catch (Throwable t) {
            synchronized (LOCK) {
                state = State.done("Benchmark failed", null, t.getMessage());
            }
            updateNotification("TurboQuant benchmark failed");
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

        State(boolean running, boolean done, String status, String resultJson, String error) {
            this.running = running;
            this.done = done;
            this.status = status;
            this.resultJson = resultJson;
            this.error = error;
        }

        static State idle() {
            return new State(false, false, null, null, null);
        }

        static State running(String status) {
            return new State(true, false, status, null, null);
        }

        static State done(String status, String resultJson, String error) {
            return new State(false, true, status, resultJson, error);
        }
    }

    static final class Snapshot {
        final boolean running;
        final boolean done;
        final String status;
        final String resultJson;
        final String error;

        Snapshot(boolean running, boolean done, String status, String resultJson, String error) {
            this.running = running;
            this.done = done;
            this.status = status;
            this.resultJson = resultJson;
            this.error = error;
        }
    }
}
