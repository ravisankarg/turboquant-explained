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

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class DownloadService extends Service {
    static final String ACTION_START = "com.turboquant.benchmark.action.START_DOWNLOAD";
    static final String EXTRA_ID = "id";
    static final String EXTRA_LABEL = "label";
    static final String EXTRA_FILE = "file";
    static final String EXTRA_BYTES = "bytes";

    private static final String VECTOR_URL =
            "https://huggingface.co/datasets/YoKONCy/Cohere-1M-wikipedia-768d/resolve/main/cohere_train.f32";
    private static final String CHANNEL_ID = "turboquant_downloads";
    private static final int NOTIFICATION_ID = 4101;
    private static final Object LOCK = new Object();
    private static State state = State.idle();

    private final ExecutorService worker = Executors.newSingleThreadExecutor();

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
        String id = intent.getStringExtra(EXTRA_ID);
        String label = intent.getStringExtra(EXTRA_LABEL);
        String fileName = intent.getStringExtra(EXTRA_FILE);
        long bytes = intent.getLongExtra(EXTRA_BYTES, 0L);
        if (id == null || label == null || fileName == null || bytes <= 0L) {
            stopSelf(startId);
            return START_NOT_STICKY;
        }
        synchronized (LOCK) {
            if (state.running) {
                return START_STICKY;
            }
            state = State.running(id, label, fileName, bytes);
        }
        startForeground(NOTIFICATION_ID, notification("Starting " + label, 0, bytes, true));
        worker.execute(() -> runDownload(startId, id, label, fileName, bytes));
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    static Snapshot snapshot(Context context) {
        synchronized (LOCK) {
            long existing = 0L;
            if (state.fileName != null) {
                File target = new File(context.getFilesDir(), state.fileName);
                File part = new File(context.getFilesDir(), state.fileName + ".part");
                if (target.isFile() && target.length() == state.totalBytes) {
                    existing = state.totalBytes;
                } else if (part.isFile()) {
                    existing = part.length();
                }
            }
            long downloaded = Math.max(state.downloadedBytes, existing);
            return new Snapshot(state.running, state.done, state.id, state.label, state.totalBytes,
                    downloaded, state.error);
        }
    }

    private void runDownload(int startId, String id, String label, String fileName, long totalBytes) {
        File target = new File(getFilesDir(), fileName);
        File tmp = new File(getFilesDir(), fileName + ".part");
        try {
            if (target.isFile() && target.length() == totalBytes) {
                markDone(id, label, totalBytes);
                updateNotification("Already downloaded " + label, totalBytes, totalBytes, false);
                stopSelf(startId);
                return;
            }
            if (tmp.isFile() && tmp.length() > totalBytes && !tmp.delete()) {
                throw new IllegalStateException("Could not reset oversized partial file");
            }
            long start = tmp.isFile() ? tmp.length() : 0L;
            updateState(start, null);

            HttpURLConnection conn = (HttpURLConnection) new URL(VECTOR_URL).openConnection();
            conn.setInstanceFollowRedirects(true);
            conn.setRequestProperty("Range", "bytes=" + start + "-" + (totalBytes - 1));
            conn.setConnectTimeout(30_000);
            conn.setReadTimeout(60_000);
            int code = conn.getResponseCode();
            boolean append = start > 0L && code == 206;
            if (start > 0L && code == 200) {
                if (!tmp.delete()) {
                    throw new IllegalStateException("Server ignored resume and old partial file could not be reset");
                }
                start = 0L;
                append = false;
            } else if (code != 206 && code != 200) {
                throw new IllegalStateException("HTTP " + code);
            }

            try (InputStream in = conn.getInputStream();
                 FileOutputStream out = new FileOutputStream(tmp, append)) {
                byte[] buf = new byte[1024 * 1024];
                long total = start;
                long lastNotify = 0L;
                int n;
                while ((n = in.read(buf)) != -1 && total < totalBytes) {
                    int keep = (int) Math.min(n, totalBytes - total);
                    out.write(buf, 0, keep);
                    total += keep;
                    updateState(total, null);
                    if (total - lastNotify >= 4L * 1024L * 1024L || total == totalBytes) {
                        lastNotify = total;
                        updateNotification("Downloading " + label, total, totalBytes, false);
                    }
                }
                if (total != totalBytes) {
                    throw new IllegalStateException("short download: " + total + " bytes");
                }
            } finally {
                conn.disconnect();
            }

            if (target.exists() && !target.delete()) {
                throw new IllegalStateException("Could not replace old vector file");
            }
            if (!tmp.renameTo(target)) {
                throw new IllegalStateException("Could not move downloaded vector file");
            }
            markDone(id, label, totalBytes);
            updateNotification("Download complete: " + label, totalBytes, totalBytes, false);
        } catch (Exception e) {
            markError(id, label, totalBytes, tmp.length(), e.getMessage());
            updateNotification("Download paused: " + label, tmp.length(), totalBytes, false);
        } finally {
            stopSelf(startId);
        }
    }

    private void updateState(long downloaded, String error) {
        synchronized (LOCK) {
            state.downloadedBytes = downloaded;
            state.error = error;
        }
    }

    private void markDone(String id, String label, long bytes) {
        synchronized (LOCK) {
            state = new State(false, true, id, label, null, bytes, bytes, null);
        }
    }

    private void markError(String id, String label, long bytes, long downloaded, String error) {
        synchronized (LOCK) {
            state = new State(false, false, id, label, null, bytes, downloaded, error);
        }
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT < 26) {
            return;
        }
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "TurboQuant downloads",
                NotificationManager.IMPORTANCE_LOW);
        channel.setDescription("Cohere vector dataset downloads");
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(channel);
    }

    private Notification notification(String title, long done, long total, boolean indeterminate) {
        Intent launch = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(
                this, 0, launch, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        Notification.Builder builder = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        int progress = total > 0L ? (int) Math.min(100, (done * 100L) / total) : 0;
        return builder
                .setSmallIcon(android.R.drawable.stat_sys_download)
                .setContentTitle(title)
                .setContentText(humanBytes(done) + " / " + humanBytes(total))
                .setContentIntent(pendingIntent)
                .setOngoing(done < total)
                .setOnlyAlertOnce(true)
                .setProgress(100, progress, indeterminate)
                .build();
    }

    private void updateNotification(String title, long done, long total, boolean indeterminate) {
        NotificationManager manager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        manager.notify(NOTIFICATION_ID, notification(title, done, total, indeterminate));
    }

    static String humanBytes(long bytes) {
        if (bytes >= 1024L * 1024L * 1024L) {
            return String.format(Locale.US, "%.2f GB", bytes / 1073741824.0);
        }
        if (bytes >= 1024L * 1024L) {
            return String.format(Locale.US, "%.1f MB", bytes / 1048576.0);
        }
        if (bytes >= 1024L) {
            return String.format(Locale.US, "%.1f KB", bytes / 1024.0);
        }
        return bytes + " B";
    }

    private static final class State {
        final boolean running;
        final boolean done;
        final String id;
        final String label;
        final String fileName;
        final long totalBytes;
        long downloadedBytes;
        String error;

        State(boolean running, boolean done, String id, String label, String fileName,
              long totalBytes, long downloadedBytes, String error) {
            this.running = running;
            this.done = done;
            this.id = id;
            this.label = label;
            this.fileName = fileName;
            this.totalBytes = totalBytes;
            this.downloadedBytes = downloadedBytes;
            this.error = error;
        }

        static State idle() {
            return new State(false, false, null, null, null, 0L, 0L, null);
        }

        static State running(String id, String label, String fileName, long bytes) {
            return new State(true, false, id, label, fileName, bytes, 0L, null);
        }
    }

    static final class Snapshot {
        final boolean running;
        final boolean done;
        final String id;
        final String label;
        final long totalBytes;
        final long downloadedBytes;
        final String error;

        Snapshot(boolean running, boolean done, String id, String label, long totalBytes,
                 long downloadedBytes, String error) {
            this.running = running;
            this.done = done;
            this.id = id;
            this.label = label;
            this.totalBytes = totalBytes;
            this.downloadedBytes = downloadedBytes;
            this.error = error;
        }
    }
}
