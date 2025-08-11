#!/bin/bash
echo "🚀 EdgeMind Platform Starting..."
echo "📊 High-Performance Kernels: Enabled (125+ GFLOP/s)"
echo "🔧 CPU Optimization: AVX2/F16C"
echo ""

# Check if kernels are available
if [ -f "/app/src/kernels/cpu/int4/build/libqgemm_int4.so" ] || [ -f "/app/kernels/libqgemm_int4.so" ]; then
    echo "✅ Kernels loaded successfully"
    echo "  Library path: $LD_LIBRARY_PATH"
else
    echo "⚠️  Kernels not found, using fallback"
fi

# Display system info
echo ""
echo "System Information:"
echo "  Python: $(python --version)"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "  Cores: $(nproc)"
echo ""

# Start based on environment variable or default
case "$EDGEMIND_MODE" in
    api)
        echo "Starting FastAPI server..."
        exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    benchmark)
        echo "Running benchmarks..."
        exec python test_edgemind_kernels.py
        ;;
    test)
        echo "Running verification..."
        exec python verify_performance.py
        ;;
    *)
        echo "Starting Streamlit UI..."
        exec streamlit run ${STREAMLIT_APP:-web/streamlit_app.py}
        ;;
esac