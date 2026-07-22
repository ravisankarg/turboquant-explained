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
    private final List<Button> benchmarkButtons = new ArrayList<>();
    private Button flatIndexTab;
    private Button hnswTab;
    private Button queryMajorButton;
    private Button batchedButton;
    private ProgressBar progress;
    private TextView status;
    private LinearLayout results;
    private String renderedBenchmarkJson;
    private String displayedBenchmarkMode;
    private String selectedIndexFamily = BenchmarkService.FAMILY_FLAT;
    private String displayedIndexFamily;
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

        LinearLayout titleRow = new LinearLayout(this);
        titleRow.setOrientation(LinearLayout.HORIZONTAL);
        titleRow.setGravity(Gravity.CENTER_VERTICAL);
        root.addView(titleRow, matchWrap());

        TextView title = new TextView(this);
        title.setText("VectorDB 3.0");
        title.setTextSize(24);
        title.setTextColor(0xff17202a);
        title.setGravity(Gravity.START);
        title.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        titleRow.addView(title, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        TextView version = new TextView(this);
        version.setText("v" + BuildConfig.VERSION_NAME);
        version.setTextSize(12);
        version.setTextColor(0xff687582);
        version.setGravity(Gravity.BOTTOM);
        LinearLayout.LayoutParams versionParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        versionParams.setMargins(dp(8), dp(8), 0, 0);
        titleRow.addView(version, versionParams);

        TextView subtitle = new TextView(this);
        subtitle.setText("Download one 100K Cohere 768-d vector slice, then compare FlatIndex and HNSW with separate real-traffic and batched-throughput tests for 50K and 100K tables.");
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

        TextView familyTitle = new TextView(this);
        familyTitle.setText("Index family");
        familyTitle.setTextSize(15);
        familyTitle.setTextColor(0xff17202a);
        familyTitle.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        familyTitle.setPadding(0, dp(8), 0, dp(4));
        root.addView(familyTitle, matchWrap());

        LinearLayout familyTabs = new LinearLayout(this);
        familyTabs.setOrientation(LinearLayout.HORIZONTAL);
        flatIndexTab = new Button(this);
        hnswTab = new Button(this);
        flatIndexTab.setOnClickListener(v -> selectFamily(BenchmarkService.FAMILY_FLAT));
        hnswTab.setOnClickListener(v -> selectFamily(BenchmarkService.FAMILY_HNSW));
        familyTabs.addView(flatIndexTab, new LinearLayout.LayoutParams(0, dp(52), 1.0f));
        LinearLayout.LayoutParams hnswTabParams = new LinearLayout.LayoutParams(0, dp(52), 1.0f);
        hnswTabParams.setMargins(dp(6), 0, 0, 0);
        familyTabs.addView(hnswTab, hnswTabParams);
        root.addView(familyTabs, matchWrap());
        benchmarkButtons.add(flatIndexTab);
        benchmarkButtons.add(hnswTab);
        updateFamilyTabLabels();

        TextView modeHint = new TextView(this);
        modeHint.setText("A) Real traffic: one query completes across the index before the next.\nB) Batched throughput: the full query batch is optimized for aggregate throughput.\nFlatIndex scans bounded disk chunks; HNSW traverses a compact resident graph and reads candidate vectors on demand. The pure-vector search working set is 30 MiB for 50K and 50 MiB for 100K.");
        modeHint.setTextSize(13);
        modeHint.setTextColor(0xff53606d);
        modeHint.setPadding(0, dp(8), 0, dp(8));
        root.addView(modeHint, matchWrap());

        queryMajorButton = new Button(this);
        queryMajorButton.setText("A) Run real traffic / query-major");
        queryMajorButton.setOnClickListener(v -> startBenchmark(modeForFamily(false)));
        root.addView(queryMajorButton, matchWrap());
        benchmarkButtons.add(queryMajorButton);

        batchedButton = new Button(this);
        batchedButton.setText("B) Run batched throughput");
        batchedButton.setOnClickListener(v -> startBenchmark(modeForFamily(true)));
        LinearLayout.LayoutParams batchParams = matchWrap();
        batchParams.setMargins(0, dp(6), 0, 0);
        root.addView(batchedButton, batchParams);
        benchmarkButtons.add(batchedButton);

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
        setBenchmarkButtonsEnabled();
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

    private void startBenchmark(String mode) {
        selectedIndexFamily = BenchmarkService.familyForMode(mode);
        updateFamilyTabLabels();
        JSONArray datasets = availableDatasetsJson();
        boolean cached = BenchmarkService.hasCached(this, mode);
        if (datasets.length() == 0 && !cached) {
            status.setText("Download at least one dataset first.");
            setBenchmarkButtonsEnabled();
            return;
        }

        // A cached report is a local UI operation. Starting the foreground
        // service just to read it can be killed by Android before the service
        // reaches startForeground(), which made a second tap appear to close
        // the app.
        if (cached) {
            String cachedJson = BenchmarkService.cachedReport(this, mode);
            if (cachedJson != null) {
                setBusy(false);
                progress.setIndeterminate(false);
                renderedBenchmarkJson = cachedJson;
                status.setText("Showing cached " + BenchmarkService.modeLabel(mode) + " result");
                renderReport(cachedJson);
                return;
            }
        }

        setBusy(true);
        results.removeAllViews();
        renderedBenchmarkJson = null;
        displayedBenchmarkMode = mode;
        displayedIndexFamily = selectedIndexFamily;
        status.setText("Starting " + BenchmarkService.modeLabel(mode) + " for " + datasets.length()
                + " dataset(s). It continues in background and with the screen off.");
        Intent intent = new Intent(this, BenchmarkService.class)
                .setAction(BenchmarkService.ACTION_START)
                .putExtra(BenchmarkService.EXTRA_DATASETS_JSON, datasets.toString())
                .putExtra(BenchmarkService.EXTRA_OUTPUT_DIR, getFilesDir().getAbsolutePath())
                .putExtra(BenchmarkService.EXTRA_MODE, mode);
        if (Build.VERSION.SDK_INT >= 26) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
        refreshBenchmarkState();
    }

    private String modeForFamily(boolean batched) {
        if (BenchmarkService.FAMILY_HNSW.equals(selectedIndexFamily)) {
            return batched
                    ? BenchmarkService.MODE_HNSW_BATCHED
                    : BenchmarkService.MODE_HNSW_QUERY_MAJOR;
        }
        return batched ? BenchmarkService.MODE_BATCHED : BenchmarkService.MODE_QUERY_MAJOR;
    }

    private void selectFamily(String family) {
        if (family == null || family.equals(selectedIndexFamily)) {
            showCachedFamily(selectedIndexFamily);
            return;
        }
        selectedIndexFamily = family;
        displayedIndexFamily = family;
        displayedBenchmarkMode = null;
        renderedBenchmarkJson = null;
        updateFamilyTabLabels();
        setBenchmarkButtonsEnabled();
        showCachedFamily(family);
    }

    private void showCachedFamily(String family) {
        String queryMode = BenchmarkService.FAMILY_HNSW.equals(family)
                ? BenchmarkService.MODE_HNSW_QUERY_MAJOR
                : BenchmarkService.MODE_QUERY_MAJOR;
        String batchMode = BenchmarkService.FAMILY_HNSW.equals(family)
                ? BenchmarkService.MODE_HNSW_BATCHED
                : BenchmarkService.MODE_BATCHED;
        String json = BenchmarkService.cachedReport(this, queryMode);
        if (json == null) {
            json = BenchmarkService.cachedReport(this, batchMode);
        }
        results.removeAllViews();
        renderedBenchmarkJson = null;
        if (json == null) {
            status.setText(BenchmarkService.FAMILY_HNSW.equals(family)
                    ? "No HNSW benchmark cached yet. Tap A or B to build/search the graph."
                    : "No FlatIndex benchmark cached yet. Tap A or B to run it.");
            return;
        }
        renderedBenchmarkJson = json;
        status.setText("Showing cached " + family + " A/B results");
        renderReport(json);
    }

    private void updateFamilyTabLabels() {
        if (flatIndexTab != null) {
            flatIndexTab.setText(BenchmarkService.FAMILY_FLAT.equals(selectedIndexFamily)
                    ? "FlatIndex (selected)" : "FlatIndex");
        }
        if (hnswTab != null) {
            hnswTab.setText(BenchmarkService.FAMILY_HNSW.equals(selectedIndexFamily)
                    ? "HNSW (selected)" : "HNSW");
        }
        if (queryMajorButton != null) {
            queryMajorButton.setText(BenchmarkService.FAMILY_HNSW.equals(selectedIndexFamily)
                    ? "A) Run HNSW real traffic / query-major"
                    : "A) Run FlatIndex real traffic / query-major");
        }
        if (batchedButton != null) {
            batchedButton.setText(BenchmarkService.FAMILY_HNSW.equals(selectedIndexFamily)
                    ? "B) Run HNSW batched throughput"
                    : "B) Run FlatIndex batched throughput");
        }
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
        renderSingleReport(json);

        // Keep both completed measurement models visible together, but only
        // within the selected index-family tab.
        try {
            JSONObject primary = new JSONObject(json);
            String primaryMode = primary.optString("mode", BenchmarkService.MODE_QUERY_MAJOR);
            String family = primary.optString("index_family", BenchmarkService.familyForMode(primaryMode));
            selectedIndexFamily = family;
            displayedIndexFamily = family;
            displayedBenchmarkMode = primaryMode;
            updateFamilyTabLabels();
            String otherMode = BenchmarkService.pairedMode(primaryMode);
            String otherJson = BenchmarkService.cachedReport(this, otherMode);
            if (otherJson != null) {
                renderSingleReport(otherJson);
            }
        } catch (Exception ignored) {
            // renderSingleReport already exposes malformed primary JSON.
        }
    }

    private void renderSingleReport(String json) {
        try {
            JSONObject report = new JSONObject(json);
            addSectionTitle(report.optString("mode_label", "Benchmark result"));
            addSectionTitle("Summary");
            LinearLayout summary = panel();
            addSummaryRow(summary, "Datasets benchmarked", report.getString("datasets"));
            addSummaryRow(summary, "Index family", report.optString("index_family", "FlatIndex"));
            addSummaryRow(summary, "Execution mode", report.optString("mode_label", "unknown"));
            addSummaryRow(summary, "Timing model", report.optString("timing_model", "not reported"));
            addSummaryRow(summary, "Dimension", String.valueOf(report.getInt("dim")));
            addSummaryRow(summary, "Self queries / dataset", String.valueOf(report.getInt("self_queries")));
            addSummaryRow(summary, "Random queries / dataset", String.valueOf(report.getInt("random_queries")));
            addSummaryRow(summary, "Steady-state vector RAM caps", report.optString("vector_ram_caps", "50K: 30.0 MB; 100K: 50.0 MB"));
            addSummaryRow(summary, "Raw FP32 chunk vectors", report.optString("raw_chunk_vectors", "not reported"));
            addSummaryRow(summary, "Raw FP32 staging RAM", report.optString("raw_chunk_ram", "not reported"));
            addSummaryRow(summary, "FP32 raw storage", report.optString("raw_fp32_storage", "disk-backed"));
            addSummaryRow(summary, "Methods", report.optString("methods", "not reported"));
            if (report.has("graph_params")) {
                addSummaryRow(summary, "HNSW parameters", report.optString("graph_params", "not reported"));
                addSummaryRow(summary, "HNSW graph/search RAM", report.optString("graph_ram", "not reported"));
            }
            results.addView(summary, matchWrap());

            addSectionTitle("KPI Tables");
            String reportMode = report.optString("mode", BenchmarkService.MODE_QUERY_MAJOR);
            JSONArray tables = report.getJSONArray("tables");
            for (int tableIndex = 0; tableIndex < tables.length(); tableIndex++) {
                JSONObject datasetTable = tables.getJSONObject(tableIndex);
                String vectorRamCap = datasetTable.optString("vector_ram_cap", "vector RAM cap unavailable");
                JSONArray rows = datasetTable.getJSONArray("rows");
                addSectionTitle(datasetTable.getString("dataset") + " - "
                        + datasetTable.getString("vectors") + " (" + vectorRamCap + ")\n"
                        + threadSummaryForMode(reportMode) + "\n"
                        + "One-time index/build: " + oneTimeIndexSummary(rows)
                        + " (excluded from ms/query)");
                HorizontalScrollView hscroll = new HorizontalScrollView(this);
                hscroll.setHorizontalScrollBarEnabled(true);
                TableLayout table = new TableLayout(this);
                table.setStretchAllColumns(false);
                table.setShrinkAllColumns(false);
                table.addView(tableRow(new String[]{
                        "Method", "Bits", "Self R@1", "Self R@10", "Random R@10",
                        "Prep/load ms", "Write ms",
                        "ms/query", "First q ms", "Warm ms/query", "ROM",
                        "Data store", "Vector staging", "Search RAM"
                }, true, false));

                for (int i = 0; i < rows.length(); i++) {
                    JSONObject r = rows.getJSONObject(i);
                    table.addView(tableRow(new String[]{
                            r.getString("index"),
                            r.getString("bits"),
                            r.getString("self_r1"),
                            r.getString("self_r10"),
                            r.getString("random_r10"),
                            r.getString("prepare_ms"),
                            r.getString("write_ms"),
                            r.getString("ms_per_query"),
                            r.optString("first_query_ms", "n/a"),
                            r.optString("warm_ms_per_query", "n/a"),
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
            String modeKey = reportMode;
            boolean hnsw = BenchmarkService.isHnswMode(modeKey);
            boolean batched = BenchmarkService.MODE_BATCHED.equals(modeKey)
                    || BenchmarkService.MODE_HNSW_BATCHED.equals(modeKey);
            addNote(notes, "One-time index/build time is shown once in each dataset table title and is excluded from ms/query.");
            addNote(notes, hnsw
                    ? "Prep/load ms = persisted FP16/graph load and cache setup. HNSW candidate reads and traversal are included in query-major ms/query."
                    : (batched
                    ? "Prep/load ms = persisted-index load plus one-time range preparation; it is reported separately from warm batched ms/query."
                    : "Prep/load ms = persisted-index load plus cache preparation; TurboQuant range load/preparation is included in query-major ms/query."));
            addNote(notes, hnsw
                    ? "Write ms = persisted FP16 navigation store plus compact HNSW graph write; 8/4-bit rows also include their persisted TurboQuant payload sidecars."
                    : "Write ms = persisted index file write; FP16 writes IEEE FP16 and TurboQuant writes .tv.");
            addNote(notes, "Query workload: 1000 self queries and 1000 random queries per dataset at k=10.");
            addNote(notes, "Random R@1 is omitted. R@10 is the mean fraction of exact FP32 top-10 neighbors recovered by the approximate top-10.");
            addNote(notes, hnsw
                    ? "HNSW rows share one compact graph built from disk-backed FP16 navigation vectors. FP32/FP16 score graph candidates from disk; 8/4-bit rows score compressed candidates and exact-rerank them with disk-backed FP32 reads."
                    : "All rows are FlatIndex scans: FP32, persisted FP16, TurboQuant 8-bit, and TurboQuant 4-bit.");
            addNote(notes, hnsw
                    ? "HNSW graph parameters are " + report.optString("graph_params", "not reported") + ". These parameters trade recall against candidate expansion and disk reads; efSearch is the main latency/recall knob."
                    : "Search RAM is the accounted steady-state search working-set budget: query vectors plus one raw f32/f16 chunk or one compressed TurboQuant range, blocked SIMD copy, and search caches.");
            addNote(notes, hnsw
                    ? "HNSW graph/search RAM is " + report.optString("graph_ram", "the per-dataset cap") + ". It excludes app/UI memory, Rust/runtime/dependency code, allocator overhead, and OS file cache."
                    : "Vector staging is the method-specific vector payload held for the active disk chunk/range. Query-major FP16 keeps raw half bits and converts during dot; Search RAM is the full accounted vector-working-set cap, including query and search scratch.");
            if (!hnsw) {
                addNote(notes, "FlatIndex low-bit recall uses up to 256 approximate candidates per bounded range followed by exact disk-backed FP32 reranking. This applies to FlatIndex A 8/4-bit and FlatIndex B 4-bit; the full raw database is still never resident.");
            }
            addNote(notes, "All methods are disk-backed, including FP32. The full 50K/100K stores stay on disk; only bounded chunks/ranges or HNSW candidates enter RAM. The cap excludes app/UI memory, Rust/runtime/dependency code, allocator overhead, and OS file cache. It is 30 MiB for 50K and 50 MiB for 100K.");
            addNote(notes, batched
                    ? (hnsw
                    ? "ms/query is parallel HNSW batch throughput: all 2000 queries search the resident graph in parallel and total elapsed time is divided by 2000."
                    : "ms/query is warm batched throughput: each bounded chunk or compressed range is loaded once, searched against all 1000 self + 1000 random queries, and the total is divided by 2000.")
                    : (hnsw
                    ? "ms/query is the average of 16 HNSW query-major latency probes (8 self + 8 random). First q ms is the first probe; Warm ms/query is the mean of the remaining probes."
                    : "ms/query is the average of 16 query-major latency probes (8 self + 8 random). First q ms is the first probe; Warm ms/query is the mean of the remaining probes. One probe scans every bounded chunk or compressed range before the next probe starts. TurboQuant range load and preparation are included."));
            addNote(notes, hnsw
                    ? "The HNSW graph is reused by HNSW A and HNSW B after the first build; graph build/index time is not charged to each query."
                    : "TurboQuant reads bounded .tv ranges, searches each range, merges top-10 results, and releases the range before loading the next one.");
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

    private String oneTimeIndexSummary(JSONArray rows) {
        StringBuilder summary = new StringBuilder();
        for (int i = 0; i < rows.length(); i++) {
            try {
                JSONObject row = rows.getJSONObject(i);
                if (summary.length() > 0) {
                    summary.append("; ");
                }
                summary.append(methodLabel(row.optString("bits", "n/a")))
                        .append(" ")
                        .append(row.optString("index_ms", "n/a"))
                        .append(" ms");
            } catch (Exception ignored) {
                // A malformed row should not prevent the rest of the report rendering.
            }
        }
        return summary.length() == 0 ? "not reported" : summary.toString();
    }

    private String methodLabel(String bits) {
        if ("32".equals(bits)) {
            return "FP32";
        }
        if ("16".equals(bits)) {
            return "FP16";
        }
        return bits + "-bit";
    }

    private String threadSummaryForMode(String mode) {
        int rayonWorkers = Math.max(1, Runtime.getRuntime().availableProcessors());
        boolean batched = BenchmarkService.MODE_BATCHED.equals(mode)
                || BenchmarkService.MODE_HNSW_BATCHED.equals(mode);
        return batched
                ? "Threads: " + rayonWorkers + " Rayon workers for timed batch"
                : "Threads: 1 timed query worker; " + rayonWorkers
                + " Rayon workers used by parallel recall/prep";
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
        boolean modeMatchesDisplayed = displayedBenchmarkMode == null
                || snap.mode == null
                || snap.mode.equals(displayedBenchmarkMode);
        boolean familyMatchesDisplayed = displayedIndexFamily == null
                || snap.mode == null
                || BenchmarkService.familyForMode(snap.mode).equals(displayedIndexFamily);
        if (snap.done && modeMatchesDisplayed && familyMatchesDisplayed) {
            setBusy(false);
            setBenchmarkButtonsEnabled();
            if (snap.error != null) {
                status.setText("Benchmark failed: " + snap.error);
            } else if (snap.resultJson != null && !snap.resultJson.equals(renderedBenchmarkJson)) {
                status.setText(snap.cached
                        ? "Showing cached " + BenchmarkService.modeLabel(snap.mode) + " result"
                        : BenchmarkService.modeLabel(snap.mode) + " complete");
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
            sb.append("\nFlatIndex cached modes: ")
                    .append(BenchmarkService.hasCached(this, BenchmarkService.MODE_QUERY_MAJOR) ? "A ready" : "A not run")
                    .append(", ")
                    .append(BenchmarkService.hasCached(this, BenchmarkService.MODE_BATCHED) ? "B ready" : "B not run");
            sb.append("\nHNSW cached modes: ")
                    .append(BenchmarkService.hasCached(this, BenchmarkService.MODE_HNSW_QUERY_MAJOR) ? "A ready" : "A not run")
                    .append(", ")
                    .append(BenchmarkService.hasCached(this, BenchmarkService.MODE_HNSW_BATCHED) ? "B ready" : "B not run");
            any = true;
        }
        if (!any) {
            boolean cached = BenchmarkService.hasCached(this, BenchmarkService.MODE_QUERY_MAJOR)
                    || BenchmarkService.hasCached(this, BenchmarkService.MODE_BATCHED)
                    || BenchmarkService.hasCached(this, BenchmarkService.MODE_HNSW_QUERY_MAJOR)
                    || BenchmarkService.hasCached(this, BenchmarkService.MODE_HNSW_BATCHED);
            sb.append(cached
                    ? "\nA cached benchmark result is available; download the raw slice to run a new mode."
                    : "\nNone yet. Download the 100K raw Cohere slice; benchmark will produce separate 50K and 100K tables for FlatIndex and HNSW.");
        }
        return sb.toString();
    }

    private void setBusy(boolean busy) {
        for (Button button : downloadButtons) {
            button.setEnabled(!busy);
        }
        if (busy) {
            for (Button button : benchmarkButtons) {
                button.setEnabled(false);
            }
        } else {
            setBenchmarkButtonsEnabled();
        }
    }

    private void setDownloadRunningUi(boolean running) {
        for (Button button : downloadButtons) {
            button.setEnabled(!running);
        }
        if (running) {
            for (Button button : benchmarkButtons) {
                button.setEnabled(false);
            }
        } else {
            setBenchmarkButtonsEnabled();
        }
    }

    private void setBenchmarkButtonsEnabled() {
        updateFamilyTabLabels();
        String queryMode = modeForFamily(false);
        String batchMode = modeForFamily(true);
        queryMajorButton.setEnabled(hasAnyDataset() || BenchmarkService.hasCached(this, queryMode));
        batchedButton.setEnabled(hasAnyDataset() || BenchmarkService.hasCached(this, batchMode));
        flatIndexTab.setEnabled(hasAnyDataset()
                || BenchmarkService.hasCached(this, BenchmarkService.MODE_QUERY_MAJOR)
                || BenchmarkService.hasCached(this, BenchmarkService.MODE_BATCHED));
        hnswTab.setEnabled(hasAnyDataset()
                || BenchmarkService.hasCached(this, BenchmarkService.MODE_HNSW_QUERY_MAJOR)
                || BenchmarkService.hasCached(this, BenchmarkService.MODE_HNSW_BATCHED));
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
