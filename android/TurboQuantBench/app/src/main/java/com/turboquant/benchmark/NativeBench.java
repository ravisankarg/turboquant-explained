package com.turboquant.benchmark;

final class NativeBench {
    static {
        System.loadLibrary("tqbench");
    }

    private NativeBench() {}

    static native String runBenchmark(String datasetsJson, String outputDir);

    static native String runQueryMajor(String datasetsJson, String outputDir);

    static native String runBatched(String datasetsJson, String outputDir);

    static native String runHnswQueryMajor(String datasetsJson, String outputDir);

    static native String runHnswBatched(String datasetsJson, String outputDir);
}
