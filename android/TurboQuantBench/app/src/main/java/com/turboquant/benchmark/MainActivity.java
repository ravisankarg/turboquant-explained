package com.turboquant.benchmark;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final String VECTOR_URL =
            "https://huggingface.co/datasets/YoKONCy/Cohere-1M-wikipedia-768d/resolve/main/cohere_train.f32";
    private static final long VECTOR_BYTES = 50_000L * 768L * 4L;
    private static final String VECTOR_FILE = "cohere_50k_768.f32";

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private Button downloadButton;
    private Button benchmarkButton;
    private ProgressBar progress;
    private TextView status;
    private LinearLayout results;
    private File vectorPath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        vectorPath = new File(getFilesDir(), VECTOR_FILE);
        buildUi();
        updateInitialState();
    }

    @Override
    protected void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(18), dp(20), dp(24));
        root.setBackgroundColor(0xfff7f8fb);
        scroll.addView(root, new ScrollView.LayoutParams(
                ScrollView.LayoutParams.MATCH_PARENT,
                ScrollView.LayoutParams.WRAP_CONTENT));

        TextView title = new TextView(this);
        title.setText("TurboQuant Benchmark");
        title.setTextSize(24);
        title.setTextColor(0xff17202a);
        title.setGravity(Gravity.START);
        title.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        root.addView(title, matchWrap());

        TextView subtitle = new TextView(this);
        subtitle.setText("50K Cohere 768-d vectors, FP32 baseline, turbovec quantized indexes");
        subtitle.setTextSize(14);
        subtitle.setTextColor(0xff53606d);
        subtitle.setPadding(0, dp(4), 0, dp(18));
        root.addView(subtitle, matchWrap());

        downloadButton = new Button(this);
        downloadButton.setText("Download 50K Cohere vectors");
        downloadButton.setOnClickListener(v -> startDownload());
        root.addView(downloadButton, matchWrap());

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setMax(1000);
        progress.setProgress(0);
        LinearLayout.LayoutParams progressParams = matchWrap();
        progressParams.setMargins(0, dp(12), 0, dp(10));
        root.addView(progress, progressParams);

        benchmarkButton = new Button(this);
        benchmarkButton.setText("Benchmark");
        benchmarkButton.setOnClickListener(v -> startBenchmark());
        root.addView(benchmarkButton, matchWrap());

        status = new TextView(this);
        status.setTextSize(14);
        status.setTextColor(0xff2e3a46);
        status.setPadding(0, dp(14), 0, dp(8));
        root.addView(status, matchWrap());

        results = new LinearLayout(this);
        results.setOrientation(LinearLayout.VERTICAL);
        results.setPadding(0, dp(6), 0, 0);
        root.addView(results, matchWrap());

        setContentView(scroll);
    }

    private void updateInitialState() {
        boolean ready = vectorPath.isFile() && vectorPath.length() == VECTOR_BYTES;
        benchmarkButton.setEnabled(ready);
        progress.setProgress(ready ? progress.getMax() : 0);
        status.setText(ready
                ? "Dataset ready: " + humanBytes(vectorPath.length())
                : "Download stores only the first 50,000 vectors from the raw Cohere train file.");
    }

    private void startDownload() {
        downloadButton.setEnabled(false);
        benchmarkButton.setEnabled(false);
        results.removeAllViews();
        status.setText("Starting download...");
        worker.execute(() -> {
            File tmp = new File(getFilesDir(), VECTOR_FILE + ".part");
            try {
                downloadRange(tmp);
                if (vectorPath.exists() && !vectorPath.delete()) {
                    throw new IllegalStateException("Could not replace old vector file");
                }
                if (!tmp.renameTo(vectorPath)) {
                    throw new IllegalStateException("Could not move downloaded vector file");
                }
                main.post(() -> {
                    progress.setProgress(progress.getMax());
                    status.setText("Download complete: " + humanBytes(vectorPath.length()));
                    downloadButton.setEnabled(true);
                    benchmarkButton.setEnabled(true);
                });
            } catch (Exception e) {
                tmp.delete();
                main.post(() -> {
                    status.setText("Download failed: " + e.getMessage());
                    downloadButton.setEnabled(true);
                    benchmarkButton.setEnabled(vectorPath.isFile() && vectorPath.length() == VECTOR_BYTES);
                });
            }
        });
    }

    private void downloadRange(File tmp) throws Exception {
        HttpURLConnection conn = (HttpURLConnection) new URL(VECTOR_URL).openConnection();
        conn.setInstanceFollowRedirects(true);
        conn.setRequestProperty("Range", "bytes=0-" + (VECTOR_BYTES - 1));
        conn.setConnectTimeout(30_000);
        conn.setReadTimeout(60_000);
        int code = conn.getResponseCode();
        if (code != 206 && code != 200) {
            throw new IllegalStateException("HTTP " + code);
        }
        try (InputStream in = conn.getInputStream(); FileOutputStream out = new FileOutputStream(tmp)) {
            byte[] buf = new byte[1024 * 1024];
            long total = 0;
            int n;
            while ((n = in.read(buf)) != -1 && total < VECTOR_BYTES) {
                int keep = (int) Math.min(n, VECTOR_BYTES - total);
                out.write(buf, 0, keep);
                total += keep;
                long done = total;
                main.post(() -> {
                    progress.setProgress((int) ((done * progress.getMax()) / VECTOR_BYTES));
                    status.setText(String.format(Locale.US, "Downloading %s / %s",
                            humanBytes(done), humanBytes(VECTOR_BYTES)));
                });
            }
            if (total != VECTOR_BYTES) {
                throw new IllegalStateException("short download: " + total + " bytes");
            }
        } finally {
            conn.disconnect();
        }
    }

    private void startBenchmark() {
        downloadButton.setEnabled(false);
        benchmarkButton.setEnabled(false);
        results.removeAllViews();
        status.setText("Benchmark running. This performs 1000 self-query and 1000 random-query searches per index.");
        worker.execute(() -> {
            try {
                String json = NativeBench.runBenchmark(vectorPath.getAbsolutePath(), getFilesDir().getAbsolutePath());
                main.post(() -> {
                    status.setText("Benchmark complete");
                    renderReport(json);
                    downloadButton.setEnabled(true);
                    benchmarkButton.setEnabled(true);
                });
            } catch (Throwable t) {
                main.post(() -> {
                    status.setText("Benchmark failed: " + t.getMessage());
                    downloadButton.setEnabled(true);
                    benchmarkButton.setEnabled(true);
                });
            }
        });
    }

    private void renderReport(String json) {
        results.removeAllViews();
        try {
            JSONObject report = new JSONObject(json);
            addSectionTitle("Summary");
            LinearLayout summary = panel();
            addSummaryRow(summary, "Dataset", report.getString("dataset"));
            addSummaryRow(summary, "Dimension", String.valueOf(report.getInt("dim")));
            addSummaryRow(summary, "Base vectors", String.valueOf(report.getInt("base_vectors")));
            addSummaryRow(summary, "Self queries", String.valueOf(report.getInt("self_queries")));
            addSummaryRow(summary, "Random queries", String.valueOf(report.getInt("random_queries")));
            addSummaryRow(summary, "FP32 exact time", report.getString("fp32_exact_ms") + " ms");
            results.addView(summary, matchWrap());

            addSectionTitle("KPI Table");
            HorizontalScrollView hscroll = new HorizontalScrollView(this);
            hscroll.setHorizontalScrollBarEnabled(true);
            TableLayout table = new TableLayout(this);
            table.setStretchAllColumns(false);
            table.setShrinkAllColumns(false);
            table.addView(tableRow(new String[]{
                    "Index", "Bits", "Self R@1", "Self R@10", "Random R@1", "Random R@10",
                    "Index ms", "Prep ms", "Write ms", "Self ms", "Random ms", "us/query", "ROM", "RAM delta"
            }, true, false));

            JSONArray rows = report.getJSONArray("table");
            for (int i = 0; i < rows.length(); i++) {
                JSONObject r = rows.getJSONObject(i);
                table.addView(tableRow(new String[]{
                        r.getString("index"),
                        r.getString("bits"),
                        r.getString("self_r1"),
                        r.getString("self_r10"),
                        r.getString("random_r1"),
                        r.getString("random_r10"),
                        r.getString("index_ms"),
                        r.getString("prepare_ms"),
                        r.getString("write_ms"),
                        r.getString("self_search_ms"),
                        r.getString("random_search_ms"),
                        r.getString("us_per_query"),
                        r.getString("index_rom"),
                        r.getString("ram_delta")
                }, false, i % 2 == 1));
            }
            hscroll.addView(table);
            results.addView(hscroll, matchWrap());

            addSectionTitle("Latency Notes");
            LinearLayout notes = panel();
            addNote(notes, "Index ms = add + quantize + in-memory index store");
            addNote(notes, "Prep ms = blocked layout and search cache preparation");
            addNote(notes, "Write ms = persisted .tv file write");
            addNote(notes, "Self/random ms = 1000 searches each at k=10");
            JSONArray noteArray = report.getJSONArray("notes");
            for (int i = 0; i < noteArray.length(); i++) {
                addNote(notes, noteArray.getString(i));
            }
            results.addView(notes, matchWrap());
        } catch (Exception e) {
            TextView fallback = new TextView(this);
            fallback.setText(json);
            fallback.setTextIsSelectable(true);
            fallback.setTextColor(0xff9f2d20);
            fallback.setTextSize(13);
            fallback.setPadding(0, dp(8), 0, 0);
            results.addView(fallback, matchWrap());
        }
    }

    private LinearLayout panel() {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(12), dp(10), dp(12), dp(10));
        panel.setBackgroundColor(0xffffffff);
        return panel;
    }

    private void addSectionTitle(String text) {
        TextView title = new TextView(this);
        title.setText(text);
        title.setTextColor(0xff17202a);
        title.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        title.setTextSize(17);
        title.setPadding(0, dp(14), 0, dp(8));
        results.addView(title, matchWrap());
    }

    private void addSummaryRow(LinearLayout parent, String label, String value) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(0, dp(4), 0, dp(4));
        TextView left = textCell(label, false, false);
        left.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        TextView right = textCell(value, false, false);
        right.setGravity(Gravity.END);
        row.addView(left, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 0.9f));
        row.addView(right, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.3f));
        parent.addView(row, matchWrap());
    }

    private void addNote(LinearLayout parent, String text) {
        TextView note = new TextView(this);
        note.setText("- " + text);
        note.setTextColor(0xff344250);
        note.setTextSize(13);
        note.setPadding(0, dp(3), 0, dp(3));
        parent.addView(note, matchWrap());
    }

    private TableRow tableRow(String[] values, boolean header, boolean alternate) {
        TableRow row = new TableRow(this);
        row.setBackgroundColor(header ? 0xff233142 : (alternate ? 0xffeef2f6 : 0xffffffff));
        for (String value : values) {
            row.addView(textCell(value, header, true));
        }
        return row;
    }

    private TextView textCell(String value, boolean header, boolean table) {
        TextView cell = new TextView(this);
        cell.setText(value);
        cell.setTextSize(header ? 12 : 13);
        cell.setTextColor(header ? 0xffffffff : 0xff17202a);
        cell.setGravity(table ? Gravity.CENTER : Gravity.START);
        cell.setSingleLine(false);
        cell.setMinWidth(table ? dp(112) : 0);
        cell.setPadding(dp(8), dp(7), dp(8), dp(7));
        if (header) {
            cell.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        }
        return cell;
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static String humanBytes(long bytes) {
        if (bytes >= 1024L * 1024L) {
            return String.format(Locale.US, "%.1f MB", bytes / 1048576.0);
        }
        if (bytes >= 1024L) {
            return String.format(Locale.US, "%.1f KB", bytes / 1024.0);
        }
        return bytes + " B";
    }
}
