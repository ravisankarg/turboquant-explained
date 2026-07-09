package com.turboquant.benchmark;

final class NativeBench {
    static {
        System.loadLibrary("tqbench");
    }

    private NativeBench() {}

    static native String runBenchmark(String datasetsJson, String outputDir);
}
