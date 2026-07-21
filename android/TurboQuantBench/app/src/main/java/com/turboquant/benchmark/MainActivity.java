package com.turboquant.benchmark;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final int DIM = 768;
    private static final Dataset DOWNLOAD_DATASET =
            new Dataset("cohere-100k", "Cohere 100K vectors", "cohere_100k_768.f32", 100_000);
    private static final BenchmarkSlice[] BENCHMARK_SLICES = new BenchmarkSlice[]{
            new BenchmarkSlice("cohere-50k", "Cohere 50K vectors", 50_000),
            new BenchmarkSlice("cohere-100k", "Cohere 100K vectors", 100_000)
    };

    private final Handler main = new Handler(Looper.getMainLooper());
    private final List<Button> downloadButtons = new ArrayList<>();
    private Button benchmarkButton;
    private ProgressBar progress;
    private TextView status;
    private LinearLayout results;
    private String renderedBenchmarkJson;
    private final Runnable downloadPoller = new Runnable() {
        @Override
        public void run() {
            refreshDownloadState();
            main.postDelayed(this, 1000L);
        }
    };
    private final Runnable benchmarkPoller = new Runnable() {
        @Override
        public void run() {
            refreshBenchmarkState();
            main.postDelayed(this, 1000L);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        updateInitialState();
    }

    @Override
    protected void onResume() {
        super.onResume();
        main.post(downloadPoller);
        main.post(benchmarkPoller);
    }

    @Override
    protected void onPause() {
        main.removeCallbacks(downloadPoller);
        main.removeCallbacks(benchmarkPoller);
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        main.removeCallbacks(downloadPoller);
        main.removeCallbacks(benchmarkPoller);
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
        title.setText("VectorDB 3.0");
        title.setTextSize(24);
        title.setTextColor(0xff17202a);
        title.setGravity(Gravity.START);
        title.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        root.addView(title, matchWrap());

        TextView subtitle = new TextView(this);
        subtitle.setText("Download one 100K Cohere 768-d vector slice, then benchmark separate 50K and 100K FlatIndex tables.");
        subtitle.setTextSize(14);
        subtitle.setTextColor(0xff53606d);
        subtitle.setPadding(0, dp(4), 0, dp(18));
        root.addView(subtitle, matchWrap());

        Button button = new Button(this);
        button.setText("Download " + DOWNLOAD_DATASET.label + " (" + humanBytes(DOWNLOAD_DATASET.bytes) + ")");
        button.setOnClickListener(v -> startDownload(DOWNLOAD_DATASET));
        LinearLayout.LayoutParams params = matchWrap();
        params.setMargins(0, 0, 0, dp(8));
        root.addView(button, params);
        downloadButtons.add(button);

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setMax(1000);
        progress.setProgress(0);
        LinearLayout.LayoutParams progressParams = matchWrap();
        progressParams.setMargins(0, dp(4), 0, dp(10));
        root.addView(progress, progressParams);

        benchmarkButton = new Button(this);
        benchmarkButton.setText("Benchmark available datasets");
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
        progress.setProgress(0);
        benchmarkButton.setEnabled(hasAnyDataset());
        status.setText(datasetStatus());
        refreshDownloadState();
    }

    private void startDownload(Dataset dataset) {
        if (DownloadService.snapshot(this).running) {
            status.setText("A download is already running.");
            return;
        }
        setDownloadRunningUi(true);
        results.removeAllViews();
        renderedBenchmarkJson = null;
        File tmp = new File(getFilesDir(), dataset.fileName + ".part");
        long resume = tmp.isFile() ? tmp.length() : 0L;
        status.setText(resume > 0L
                ? "Resuming " + dataset.label + " from " + humanBytes(resume)
                : "Starting background download: " + dataset.label);
        Intent intent = new Intent(this, DownloadService.class)
                .setAction(DownloadService.ACTION_START)
                .putExtra(DownloadService.EXTRA_ID, dataset.id)
                .putExtra(DownloadService.EXTRA_LABEL, dataset.label)
                .putExtra(DownloadService.EXTRA_FILE, dataset.fileName)
                .putExtra(DownloadService.EXTRA_BYTES, dataset.bytes);
        if (Build.VERSION.SDK_INT >= 26) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
        refreshDownloadState();
    }

    private void startBenchmark() {
        JSONArray datasets = availableDatasetsJson();
        if (datasets.length() == 0) {
            status.setText("Download at least one dataset first.");
            benchmarkButton.setEnabled(false);
            return;
        }
        setBusy(true);
        results.removeAllViews();
        renderedBenchmarkJson = null;
        status.setText("Benchmark starting in foreground service for " + datasets.length()
                + " dataset(s). It continues in background and with the screen off.");
        Intent intent = new Intent(this, BenchmarkService.class)
                .setAction(BenchmarkService.ACTION_START)
                .putExtra(BenchmarkService.EXTRA_DATASETS_JSON, datasets.toString())
                .putExtra(BenchmarkService.EXTRA_OUTPUT_DIR, getFilesDir().getAbsolutePath());
        if (Build.VERSION.SDK_INT >= 26) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
        refreshBenchmarkState();
    }

    private JSONArray availableDatasetsJson() {
        JSONArray arr = new JSONArray();
        File path = DOWNLOAD_DATASET.path(getFilesDir());
        if (path.isFile() && path.length() == DOWNLOAD_DATASET.bytes) {
            for (BenchmarkSlice slice : BENCHMARK_SLICES) {
                JSONObject obj = new JSONObject();
                try {
                    obj.put("id", slice.id);
                    obj.put("label", slice.label);
                    obj.put("path", path.getAbsolutePath());
                    obj.put("vectors", slice.vectors);
                    arr.put(obj);
                } catch (Exception ignored) {
                    // JSONObject with local strings cannot fail in practice.
                }
            }
        }
        return arr;
    }

    private void renderReport(String json) {
        results.removeAllViews();
        try {
            JSONObject report = new JSONObject(json);
            addSectionTitle("Summary");
            LinearLayout summary = panel();
            addSummaryRow(summary, "Datasets benchmarked", report.getString("datasets"));
            addSummaryRow(summary, "Dimension", String.valueOf(report.getInt("dim")));
            addSummaryRow(summary, "Self queries / dataset", String.valueOf(report.getInt("self_queries")));
            addSummaryRow(summary, "Random queries / dataset", String.valueOf(report.getInt("random_queries")));
            addSummaryRow(summary, "Steady-state vector RAM caps", report.getString("vector_ram_caps"));
            addSummaryRow(summary, "Raw FP32 chunk vectors", report.getString("raw_chunk_vectors"));
            addSummaryRow(summary, "Raw FP32 staging RAM", report.getString("raw_chunk_ram"));
            addSummaryRow(summary, "FP32 raw storage", report.getString("raw_fp32_storage"));
            addSummaryRow(summary, "Methods", report.getString("methods"));
            results.addView(summary, matchWrap());

            addSectionTitle("KPI Tables");
            JSONArray tables = report.getJSONArray("tables");
            for (int tableIndex = 0; tableIndex < tables.length(); tableIndex++) {
                JSONObject datasetTable = tables.getJSONObject(tableIndex);
                String vectorRamCap = datasetTable.optString("vector_ram_cap", "vector RAM cap unavailable");
                addSectionTitle(datasetTable.getString("dataset") + " - "
                        + datasetTable.getString("vectors") + " (" + vectorRamCap + ")");
                HorizontalScrollView hscroll = new HorizontalScrollView(this);
                hscroll.setHorizontalScrollBarEnabled(true);
                TableLayout table = new TableLayout(this);
                table.setStretchAllColumns(false);
                table.setShrinkAllColumns(false);
                table.addView(tableRow(new String[]{
                        "Method", "Bits", "Self R@1", "Self R@10", "Random R@10",
                        "Index ms", "Prep/load ms", "Write ms",
                        "ms/query", "ROM", "Data store", "Vector staging", "Search RAM"
                }, true, false));

                JSONArray rows = datasetTable.getJSONArray("rows");
                for (int i = 0; i < rows.length(); i++) {
                    JSONObject r = rows.getJSONObject(i);
                    table.addView(tableRow(new String[]{
                            r.getString("index"),
                            r.getString("bits"),
                            r.getString("self_r1"),
                            r.getString("self_r10"),
                            r.getString("random_r10"),
                            r.getString("index_ms"),
                            r.getString("prepare_ms"),
                            r.getString("write_ms"),
                            r.getString("ms_per_query"),
                            r.getString("index_rom"),
                            r.optString("data_store", "disk-backed"),
                            r.optString("vector_staging", "not reported"),
                            r.optString("vector_ram", vectorRamCap)
                    }, false, i % 2 == 1));
                }
                hscroll.addView(table);
                hscroll.setFocusable(false);
                hscroll.setFocusableInTouchMode(false);
                hscroll.scrollTo(0, 0);
                results.addView(hscroll, matchWrap());
            }

            addSectionTitle("Benchmark legend");
            LinearLayout notes = panel();
            addNote(notes, "Index ms = add + quantize + in-memory index store");
            addNote(notes, "Prep/load ms = persisted-index load plus cache preparation; FP16 reads its disk cache in bounded chunks during search.");
            addNote(notes, "Write ms = persisted index file write; FP16 writes IEEE FP16 and TurboQuant writes .tv.");
            addNote(notes, "Query workload: 1000 self queries and 1000 random queries per dataset at k=10.");
            addNote(notes, "Random R@1 is omitted. R@10 is the mean fraction of exact FP32 top-10 neighbors recovered by the approximate top-10.");
            addNote(notes, "All rows are FlatIndex scans: FP32, persisted FP16, TurboQuant 8-bit, and TurboQuant 4-bit; HNSW is disabled.");
            addNote(notes, "Search RAM is the accounted steady-state search working-set budget: query vectors plus one raw/decoded chunk or one compressed TurboQuant range, blocked SIMD copy, and search caches.");
            addNote(notes, "Vector staging is the method-specific vector payload held for the active disk chunk/range; Search RAM is the full accounted vector working-set cap, including query and search scratch.");
            addNote(notes, "All methods are disk-backed, including FP32. The full 50K/100K stores stay on disk; only bounded chunks/ranges enter RAM. The cap excludes app/UI memory, Rust/runtime/dependency code, allocator overhead, and OS file cache. It is 30 MiB for both 50K and 100K.");
            addNote(notes, "ms/query uses query-major latency probes (8 self + 8 random): one probe scans every bounded chunk or compressed range before the next probe starts, so chunks/ranges are not reused across unrelated requests. Recall still uses all 1000 self + 1000 random queries. TurboQuant range load and preparation are included in probe latency.");
            addNote(notes, "TurboQuant reads bounded .tv ranges, searches each range, merges top-10 results, and releases the range before loading the next one.");
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

    private boolean hasAnyDataset() {
        File path = DOWNLOAD_DATASET.path(getFilesDir());
        return path.isFile() && path.length() == DOWNLOAD_DATASET.bytes;
    }

    private void refreshDownloadState() {
        BenchmarkService.Snapshot benchmark = BenchmarkService.snapshot(this);
        if (benchmark.running || benchmark.done) {
            return;
        }
        DownloadService.Snapshot snap = DownloadService.snapshot(this);
        if (snap.running) {
            setDownloadRunningUi(true);
            if (snap.totalBytes > 0L) {
                progress.setProgress((int) ((snap.downloadedBytes * progress.getMax()) / snap.totalBytes));
            }
            status.setText(String.format(Locale.US, "Downloading in background: %s %s / %s",
                    snap.label, humanBytes(snap.downloadedBytes), humanBytes(snap.totalBytes)));
            return;
        }
        setDownloadRunningUi(false);
        if (snap.error != null && snap.label != null) {
            progress.setProgress(snap.totalBytes > 0L
                    ? (int) ((snap.downloadedBytes * progress.getMax()) / snap.totalBytes)
                    : 0);
            status.setText("Download paused for " + snap.label + " at "
                    + humanBytes(snap.downloadedBytes)
                    + ". Tap the same download button to resume. Last error: " + snap.error);
            return;
        }
        if (snap.done && snap.label != null) {
            progress.setProgress(progress.getMax());
            status.setText("Download complete: " + snap.label + " (" + humanBytes(snap.totalBytes) + ")");
        }
    }

    private void refreshBenchmarkState() {
        BenchmarkService.Snapshot snap = BenchmarkService.snapshot(this);
        if (snap.running) {
            setBusy(true);
            progress.setIndeterminate(true);
            status.setText(snap.status != null
                    ? snap.status
                    : "Benchmark running in background. It continues with the screen off.");
            return;
        }
        progress.setIndeterminate(false);
        if (snap.done) {
            setBusy(false);
            benchmarkButton.setEnabled(hasAnyDataset());
            if (snap.error != null) {
                status.setText("Benchmark failed: " + snap.error);
            } else if (snap.resultJson != null && !snap.resultJson.equals(renderedBenchmarkJson)) {
                status.setText("Benchmark complete");
                renderReport(snap.resultJson);
                renderedBenchmarkJson = snap.resultJson;
            }
        }
    }

    private String datasetStatus() {
        StringBuilder sb = new StringBuilder("Available datasets:");
        boolean any = false;
        File path = DOWNLOAD_DATASET.path(getFilesDir());
        if (path.isFile() && path.length() == DOWNLOAD_DATASET.bytes) {
            sb.append("\n").append(DOWNLOAD_DATASET.label).append(" ready (").append(humanBytes(path.length())).append(")");
            sb.append("\nBenchmark tables: Cohere 50K and Cohere 100K, each with FlatIndex FP32/FP16/8-bit/4-bit.");
            any = true;
        }
        if (!any) {
            sb.append("\nNone yet. Download the 100K raw Cohere slice; benchmark will produce separate 50K and 100K FlatIndex tables.");
        }
        return sb.toString();
    }

    private void setBusy(boolean busy) {
        for (Button button : downloadButtons) {
            button.setEnabled(!busy);
        }
        benchmarkButton.setEnabled(!busy && hasAnyDataset());
    }

    private void setDownloadRunningUi(boolean running) {
        for (Button button : downloadButtons) {
            button.setEnabled(!running);
        }
        benchmarkButton.setEnabled(!running && hasAnyDataset());
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
        cell.setMinWidth(table ? dp(108) : 0);
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

    private static final class Dataset {
        final String id;
        final String label;
        final String fileName;
        final int vectors;
        final long bytes;

        Dataset(String id, String label, String fileName, int vectors) {
            this.id = id;
            this.label = label;
            this.fileName = fileName;
            this.vectors = vectors;
            this.bytes = (long) vectors * DIM * 4L;
        }

        File path(File root) {
            return new File(root, fileName);
        }
    }

    private static final class BenchmarkSlice {
        final String id;
        final String label;
        final int vectors;

        BenchmarkSlice(String id, String label, int vectors) {
            this.id = id;
            this.label = label;
            this.vectors = vectors;
        }
    }
}
