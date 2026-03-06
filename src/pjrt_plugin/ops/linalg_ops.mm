// Linear algebra operations: cholesky, triangular_solve
// Uses native MPS kernels (MPSMatrixDecompositionCholesky, MPSMatrixSolveTriangular)
// via the NativeOpRegistry.

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "pjrt_plugin/mps_buffer.h"
#import "pjrt_plugin/ops/registry.h"

namespace jax_mps {

// ---------------------------------------------------------------------------
// Helpers for MPS row-byte alignment.
// MPS requires 16-byte-aligned row strides; rowBytesFromColumns returns the
// recommended value (e.g. 16 for a 2-column float32 matrix instead of 8).
// When the data stride differs we blit rows into an aligned staging buffer
// before calling the MPS kernel and blit back afterwards.
// ---------------------------------------------------------------------------

/// Blit rows between buffers with different row strides on the command buffer.
static void BlitRows(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src, NSUInteger srcRowBytes,
                     id<MTLBuffer> dst, NSUInteger dstRowBytes, int64_t rows,
                     NSUInteger copyBytes) {
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    for (int64_t r = 0; r < rows; r++) {
        [blit copyFromBuffer:src
                 sourceOffset:(NSUInteger)r * srcRowBytes
                     toBuffer:dst
            destinationOffset:(NSUInteger)r * dstRowBytes
                         size:copyBytes];
    }
    [blit endEncoding];
}

/// Copy contiguous rows from src to pre-allocated dst with different row strides.
/// If no padding is needed (strides match), just copies the data directly.
static void PadToBuffer(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src, id<MTLBuffer> dst,
                        int64_t rows, NSUInteger dataRowBytes, NSUInteger mpsRowBytes) {
    if (mpsRowBytes == dataRowBytes) {
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:src.length];
        [blit endEncoding];
    } else {
        BlitRows(cmdBuf, src, dataRowBytes, dst, mpsRowBytes, rows, dataRowBytes);
    }
}

/// Copy padded rows from src to pre-allocated contiguous dst.
/// If no padding was used (strides match), just copies the data directly.
static void UnpadToBuffer(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src, id<MTLBuffer> dst,
                          int64_t rows, NSUInteger dataRowBytes, NSUInteger mpsRowBytes) {
    if (mpsRowBytes == dataRowBytes) {
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:dst.length];
        [blit endEncoding];
    } else {
        BlitRows(cmdBuf, src, mpsRowBytes, dst, dataRowBytes, rows, dataRowBytes);
    }
}

// ---------------------------------------------------------------------------
// stablehlo.cholesky – native MPSMatrixDecompositionCholesky
// Supports batched inputs of shape [batch..., n, n] by looping over batch dims.
//
// NOTE: Unlike MPSMatrixMultiplication which has native batch support via
// batchStart/batchSize, MPSMatrixDecompositionCholesky only supports single
// matrix operations. The loop-based approach is necessary. This matches how
// other frameworks (e.g., mlx) handle batched Cholesky on MPS.
// ---------------------------------------------------------------------------

/// Fill a buffer with zeros using blit command encoder.
static void FillBufferWithZeros(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> buffer, size_t size) {
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    [blit fillBuffer:buffer range:NSMakeRange(0, size) value:0];
    [blit endEncoding];
}

static NativeResult NativeHandle_cholesky(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                          mlir::Operation* op,
                                          const std::vector<id<MTLBuffer>>& inputs) {
    auto choleskyOp = mlir::dyn_cast<mlir::stablehlo::CholeskyOp>(op);
    if (!choleskyOp) {
        return NativeResult::Error("cholesky: expected CholeskyOp");
    }

    bool lower = true;
    if (choleskyOp.getLowerAttr()) {
        lower = choleskyOp.getLower();
    }

    auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto shape = resultType.getShape();
    if (shape.size() < 2) {
        return NativeResult::Error("cholesky: expected at least rank 2 (got rank " +
                                   std::to_string(shape.size()) + ")");
    }
    int64_t n = shape[shape.size() - 1];
    int64_t m = shape[shape.size() - 2];
    if (n != m) {
        return NativeResult::Error("cholesky: expected square matrix (got " + std::to_string(m) +
                                   " x " + std::to_string(n) + ")");
    }

    // Compute batch size (product of all dimensions except last two).
    int64_t batchSize = 1;
    for (size_t i = 0; i < shape.size() - 2; i++) {
        batchSize *= shape[i];
    }

    if (!resultType.getElementType().isF32()) {
        return NativeResult::Error("cholesky: only float32 is supported");
    }

    MPSDataType mps_dtype = MlirTypeToMps(resultType.getElementType());
    int pjrt_dtype = MlirTypeToPjrtDtype(resultType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);
    NSUInteger dataRowBytes = (NSUInteger)(n * (int64_t)elem_size);
    NSUInteger mpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n
                                                             dataType:mps_dtype];
    size_t matrixDataSize = (size_t)(n * n) * elem_size;  // Size of one matrix in input.
    size_t matrixMpsSize = (size_t)n * mpsRowBytes;       // Size of one matrix with MPS alignment.

    // Allocate output buffer for all batches.
    size_t totalOutSize = (size_t)batchSize * matrixDataSize;
    id<MTLBuffer> outBuf = [device newBufferWithLength:totalOutSize
                                               options:MTLResourceStorageModeShared];

    // Compile the verification shader once.
    static id<MTLComputePipelineState> verifyPipeline = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
      NSString* source = @"#include <metal_stdlib>\n"
                          "using namespace metal;\n"
                          "kernel void cholesky_verify(\n"
                          "    device float *L [[buffer(0)]],\n"
                          "    constant uint &n [[buffer(1)]],\n"
                          "    constant uint &stride [[buffer(2)]],\n"
                          "    uint tid [[thread_position_in_grid]]\n"
                          ") {\n"
                          "    if (tid != 0) return;\n"
                          "    for (uint j = 0; j < n; j++) {\n"
                          "        if (L[j * stride + j] <= 0.0f) {\n"
                          "            for (uint r = 0; r < n; r++)\n"
                          "                for (uint c = 0; c < n; c++)\n"
                          "                    L[r * stride + c] = NAN;\n"
                          "            return;\n"
                          "        }\n"
                          "    }\n"
                          "}\n";
      NSError* error = nil;
      id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
      if (lib) {
          id<MTLFunction> func = [lib newFunctionWithName:@"cholesky_verify"];
          verifyPipeline = [device newComputePipelineStateWithFunction:func error:&error];
      }
      if (!verifyPipeline) {
          MPS_LOG_ERROR("cholesky: failed to compile verify shader: %s\n",
                        error.localizedDescription.UTF8String);
      }
    });

    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                      columns:(NSUInteger)n
                                                                     rowBytes:mpsRowBytes
                                                                     dataType:mps_dtype];

    MPSMatrixDecompositionCholesky* cholesky =
        [[MPSMatrixDecompositionCholesky alloc] initWithDevice:device
                                                         lower:lower
                                                         order:(NSUInteger)n];

    // Pre-allocate reusable buffers for batch processing.
    // srcSlice: holds one matrix slice copied from input
    // srcBuf: padded source for MPS (same as srcSlice if no padding needed)
    // resultBuf: MPS output with alignment
    // unpaddedBuf: contiguous result (same as resultBuf if no padding needed)
    //
    // NOTE: Using MTLResourceStorageModeShared for simplicity. If profiling shows
    // memory bandwidth is a bottleneck, consider MTLResourceStorageModePrivate for
    // GPU-only intermediate buffers (srcBuf, resultBuf when padding is needed).
    bool needsPadding = (mpsRowBytes != dataRowBytes);
    id<MTLBuffer> srcSlice = [device newBufferWithLength:matrixDataSize
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> srcBuf = needsPadding ? [device newBufferWithLength:matrixMpsSize
                                                              options:MTLResourceStorageModeShared]
                                        : srcSlice;
    id<MTLBuffer> resultBuf = [device newBufferWithLength:matrixMpsSize
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> unpaddedBuf = needsPadding
                                    ? [device newBufferWithLength:matrixDataSize
                                                          options:MTLResourceStorageModeShared]
                                    : resultBuf;

    MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:srcBuf descriptor:desc];
    MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:resultBuf descriptor:desc];

    // Verification kernel constants.
    uint32_t n32 = (uint32_t)n;
    uint32_t lStride = (uint32_t)(mpsRowBytes / elem_size);

    // Process each matrix in the batch.
    for (int64_t b = 0; b < batchSize; b++) {
        // Blit this matrix slice from input buffer.
        size_t srcOffset = (size_t)b * matrixDataSize;
        id<MTLBlitCommandEncoder> blitIn = [cmdBuf blitCommandEncoder];
        [blitIn copyFromBuffer:inputs[0]
                  sourceOffset:srcOffset
                      toBuffer:srcSlice
             destinationOffset:0
                          size:matrixDataSize];
        [blitIn endEncoding];

        // Pad source rows to MPS-recommended alignment if needed.
        if (needsPadding) {
            memset(srcBuf.contents, 0, matrixMpsSize);
            PadToBuffer(cmdBuf, srcSlice, srcBuf, n, dataRowBytes, mpsRowBytes);
        }

        // Zero-fill result buffer (unused triangle must be clean).
        FillBufferWithZeros(cmdBuf, resultBuf, matrixMpsSize);

        // The status buffer is unreliable on Apple Silicon — it always writes 0
        // (success) regardless of whether the input is positive definite.
        // See https://developer.apple.com/forums/thread/736787
        [cholesky encodeToCommandBuffer:cmdBuf
                           sourceMatrix:sourceMatrix
                           resultMatrix:resultMatrix
                                 status:nil];

        // Verification kernel to check diagonal and fill with NaN if non-positive.
        if (verifyPipeline) {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:verifyPipeline];
            [enc setBuffer:resultBuf offset:0 atIndex:0];
            [enc setBytes:&n32 length:sizeof(n32) atIndex:1];
            [enc setBytes:&lStride length:sizeof(lStride) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        }

        // Unpad result to contiguous layout if needed.
        if (needsPadding) {
            UnpadToBuffer(cmdBuf, resultBuf, unpaddedBuf, n, dataRowBytes, mpsRowBytes);
        }

        // Blit the result to the output buffer at the correct offset.
        size_t dstOffset = (size_t)b * matrixDataSize;
        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
        [blit copyFromBuffer:unpaddedBuf
                 sourceOffset:0
                     toBuffer:outBuf
            destinationOffset:dstOffset
                         size:matrixDataSize];
        [blit endEncoding];
    }

    return NativeResult::Buffer(outBuf);
}

REGISTER_NATIVE_MPS_OP("stablehlo.cholesky", NativeHandle_cholesky);

// ---------------------------------------------------------------------------
// stablehlo.triangular_solve – native MPSMatrixSolveTriangular (float32)
//                               Metal compute shader (complex64)
// Supports batched inputs of shape [batch..., n, n] by looping over batch dims.
//
// NOTE: MPSMatrixSolveTriangular only supports single matrix operations.
// The loop-based approach is necessary (same limitation as Cholesky above).
// For complex types, we use a custom Metal compute shader that implements
// forward/back substitution directly, parallelized across RHS columns.
// ---------------------------------------------------------------------------

// Metal compute shader for complex triangular solve.
// Uses float2 to represent complex numbers (real, imag).
// Each thread handles one column of the RHS.
static id<MTLComputePipelineState> GetComplexTriSolvePipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
      NSString* source =
          @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "\n"
           "// Complex multiply: (a.x+a.y*i)(b.x+b.y*i)\n"
           "static inline float2 cmul(float2 a, float2 b) {\n"
           "    return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);\n"
           "}\n"
           "\n"
           "// Complex divide: a / b\n"
           "static inline float2 cdiv(float2 a, float2 b) {\n"
           "    float d = b.x*b.x + b.y*b.y;\n"
           "    return float2((a.x*b.x + a.y*b.y) / d,\n"
           "                  (a.y*b.x - a.x*b.y) / d);\n"
           "}\n"
           "\n"
           "kernel void complex_tri_solve(\n"
           "    device const float2 *A [[buffer(0)]],\n"
           "    device const float2 *B [[buffer(1)]],\n"
           "    device float2 *X [[buffer(2)]],\n"
           "    constant uint &n [[buffer(3)]],\n"
           "    constant uint &nrhs [[buffer(4)]],\n"
           "    constant uint &flags [[buffer(5)]],\n"
           "    uint tid [[thread_position_in_grid]]\n"
           ") {\n"
           "    if (tid >= nrhs) return;\n"
           "    uint k = tid;\n"
           "\n"
           "    bool lower = (flags & 1u) != 0;\n"
           "    bool unit_diag = (flags & 2u) != 0;\n"
           "    bool do_trans = (flags & 4u) != 0;\n"
           "    bool do_adjoint = (flags & 8u) != 0;\n"
           "\n"
           "    // Copy B column to X\n"
           "    for (uint i = 0; i < n; i++) {\n"
           "        X[i * nrhs + k] = B[i * nrhs + k];\n"
           "    }\n"
           "\n"
           "    // Determine iteration direction:\n"
           "    // lower + no trans/adj → forward\n"
           "    // upper + no trans/adj → backward\n"
           "    // lower + trans/adj → backward\n"
           "    // upper + trans/adj → forward\n"
           "    bool forward = lower != (do_trans || do_adjoint);\n"
           "\n"
           "    for (uint step = 0; step < n; step++) {\n"
           "        uint i = forward ? step : (n - 1 - step);\n"
           "        float2 sum = X[i * nrhs + k];\n"
           "\n"
           "        for (uint prev = 0; prev < step; prev++) {\n"
           "            uint j = forward ? prev : (n - 1 - prev);\n"
           "            float2 a = (do_trans || do_adjoint) ? A[j * n + i] : A[i * n + j];\n"
           "            if (do_adjoint) a.y = -a.y;\n"
           "            sum -= cmul(a, X[j * nrhs + k]);\n"
           "        }\n"
           "\n"
           "        if (!unit_diag) {\n"
           "            float2 diag = A[i * n + i];\n"
           "            if (do_adjoint) diag.y = -diag.y;\n"
           "            sum = cdiv(sum, diag);\n"
           "        }\n"
           "\n"
           "        X[i * nrhs + k] = sum;\n"
           "    }\n"
           "}\n";
      NSError* error = nil;
      id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
      if (lib) {
          id<MTLFunction> func = [lib newFunctionWithName:@"complex_tri_solve"];
          pipeline = [device newComputePipelineStateWithFunction:func error:&error];
      }
      if (!pipeline) {
          MPS_LOG_ERROR("complex_tri_solve: failed to compile shader: %s\n",
                        error.localizedDescription.UTF8String);
      }
    });
    return pipeline;
}

static NativeResult NativeHandle_triangular_solve(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                                  mlir::Operation* op,
                                                  const std::vector<id<MTLBuffer>>& inputs) {
    auto triSolveOp = mlir::dyn_cast<mlir::stablehlo::TriangularSolveOp>(op);
    if (!triSolveOp) {
        return NativeResult::Error("triangular_solve: expected TriangularSolveOp");
    }

    bool leftSide = triSolveOp.getLeftSide();
    bool lower = triSolveOp.getLower();
    bool unitDiagonal = triSolveOp.getUnitDiagonal();
    auto transposeA = triSolveOp.getTransposeA();
    bool isTranspose = (transposeA == mlir::stablehlo::Transpose::TRANSPOSE);
    bool isAdjoint = (transposeA == mlir::stablehlo::Transpose::ADJOINT);
    bool transpose = isTranspose || isAdjoint;

    auto aType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto bType = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();

    if (aShape.size() < 2 || bShape.size() < 2) {
        return NativeResult::Error("triangular_solve: expected at least rank 2 (got ranks " +
                                   std::to_string(aShape.size()) + ", " +
                                   std::to_string(bShape.size()) + ")");
    }

    // A must be square.
    int64_t aRows = aShape[aShape.size() - 2];
    int64_t aCols = aShape[aShape.size() - 1];
    if (aRows != aCols) {
        return NativeResult::Error("triangular_solve: matrix A must be square (got " +
                                   std::to_string(aRows) + " x " + std::to_string(aCols) + ")");
    }

    // A and B must have matching rank and batch dimensions.
    if (aShape.size() != bShape.size()) {
        return NativeResult::Error("triangular_solve: A and B must have same rank (got " +
                                   std::to_string(aShape.size()) + " vs " +
                                   std::to_string(bShape.size()) + ")");
    }
    for (size_t i = 0; i < aShape.size() - 2; i++) {
        if (aShape[i] != bShape[i]) {
            return NativeResult::Error(
                "triangular_solve: A and B batch dimensions must match (dim " + std::to_string(i) +
                ": " + std::to_string(aShape[i]) + " vs " + std::to_string(bShape[i]) + ")");
        }
    }

    // Compute batch size (product of all dimensions except last two).
    int64_t batchSize = 1;
    size_t batchRank = aShape.size() - 2;
    for (size_t i = 0; i < batchRank; i++) {
        batchSize *= aShape[i];
    }

    int64_t n = aShape[aShape.size() - 1];
    int64_t bRows = bShape[bShape.size() - 2];
    int64_t bCols = bShape[bShape.size() - 1];

    // Detect complex element type.
    mlir::Type elemType = bType.getElementType();
    bool isComplex = mlir::isa<mlir::ComplexType>(elemType);

    if (!isComplex && !elemType.isF32()) {
        return NativeResult::Error("triangular_solve: only float32 and complex64 are supported");
    }

    if (isComplex && !leftSide) {
        return NativeResult::Error("triangular_solve: right-side complex solve not yet supported");
    }

    int pjrt_dtype = MlirTypeToPjrtDtype(elemType);
    size_t elem_size = DtypeByteSize(pjrt_dtype);
    size_t aMatrixDataSize = (size_t)(n * n) * elem_size;
    size_t bMatrixDataSize = (size_t)(bRows * bCols) * elem_size;
    size_t totalOutSize = (size_t)batchSize * bMatrixDataSize;

    id<MTLBuffer> outBuf = [device newBufferWithLength:totalOutSize
                                               options:MTLResourceStorageModeShared];

    if (isComplex) {
        // --- Complex path: Metal compute shader ---
        id<MTLComputePipelineState> pipeline = GetComplexTriSolvePipeline(device);
        if (!pipeline) {
            return NativeResult::Error("triangular_solve: complex shader compilation failed");
        }

        // Flags: bit 0 = lower, bit 1 = unit_diag, bit 2 = transpose, bit 3 = adjoint
        uint32_t flags = (lower ? 1U : 0U) | (unitDiagonal ? 2U : 0U) | (isTranspose ? 4U : 0U) |
                         (isAdjoint ? 8U : 0U);
        uint32_t n32 = (uint32_t)n;
        uint32_t nrhs32 = (uint32_t)bCols;

        // Per-batch buffers for the shader (A slice, B slice, X output slice).
        id<MTLBuffer> aSlice = [device newBufferWithLength:aMatrixDataSize
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bSlice = [device newBufferWithLength:bMatrixDataSize
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> xSlice = [device newBufferWithLength:bMatrixDataSize
                                                   options:MTLResourceStorageModeShared];

        for (int64_t b = 0; b < batchSize; b++) {
            size_t aOffset = (size_t)b * aMatrixDataSize;
            size_t bOffset = (size_t)b * bMatrixDataSize;

            // Copy A and B slices from input buffers.
            id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
            [blit copyFromBuffer:inputs[0]
                     sourceOffset:aOffset
                         toBuffer:aSlice
                destinationOffset:0
                             size:aMatrixDataSize];
            [blit copyFromBuffer:inputs[1]
                     sourceOffset:bOffset
                         toBuffer:bSlice
                destinationOffset:0
                             size:bMatrixDataSize];
            [blit endEncoding];

            // Dispatch the complex triangular solve shader.
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:aSlice offset:0 atIndex:0];
            [enc setBuffer:bSlice offset:0 atIndex:1];
            [enc setBuffer:xSlice offset:0 atIndex:2];
            [enc setBytes:&n32 length:sizeof(n32) atIndex:3];
            [enc setBytes:&nrhs32 length:sizeof(nrhs32) atIndex:4];
            [enc setBytes:&flags length:sizeof(flags) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(nrhs32, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(
                                          MIN(nrhs32,
                                              (uint32_t)pipeline.maxTotalThreadsPerThreadgroup),
                                          1, 1)];
            [enc endEncoding];

            // Copy result to output buffer.
            id<MTLBlitCommandEncoder> blitOut = [cmdBuf blitCommandEncoder];
            [blitOut copyFromBuffer:xSlice
                       sourceOffset:0
                           toBuffer:outBuf
                  destinationOffset:bOffset
                               size:bMatrixDataSize];
            [blitOut endEncoding];
        }
    } else {
        // --- Real float32 path: MPSMatrixSolveTriangular ---
        MPSDataType mps_dtype = MPSDataTypeFloat32;

        NSUInteger nrhs = leftSide ? (NSUInteger)bCols : (NSUInteger)bRows;

        NSUInteger aDataRowBytes = (NSUInteger)(n * (int64_t)elem_size);
        NSUInteger aMpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n
                                                                  dataType:mps_dtype];
        NSUInteger bDataRowBytes = (NSUInteger)(bCols * (int64_t)elem_size);
        NSUInteger bMpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)bCols
                                                                  dataType:mps_dtype];

        bool aNeedsPadding = (aMpsRowBytes != aDataRowBytes);
        bool bNeedsPadding = (bMpsRowBytes != bDataRowBytes);
        size_t aMpsSize = (size_t)n * aMpsRowBytes;
        size_t bMpsSize = (size_t)bRows * bMpsRowBytes;

        MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                           columns:(NSUInteger)n
                                                                          rowBytes:aMpsRowBytes
                                                                          dataType:mps_dtype];

        MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)bRows
                                                                           columns:(NSUInteger)bCols
                                                                          rowBytes:bMpsRowBytes
                                                                          dataType:mps_dtype];

        MPSMatrixSolveTriangular* solver =
            [[MPSMatrixSolveTriangular alloc] initWithDevice:device
                                                       right:!leftSide
                                                       upper:!lower
                                                   transpose:transpose
                                                        unit:unitDiagonal
                                                       order:(NSUInteger)n
                                      numberOfRightHandSides:nrhs
                                                       alpha:1.0];

        id<MTLBuffer> aSlice = [device newBufferWithLength:aMatrixDataSize
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> aBuf = aNeedsPadding
                                 ? [device newBufferWithLength:aMpsSize
                                                       options:MTLResourceStorageModeShared]
                                 : aSlice;
        id<MTLBuffer> bSlice = [device newBufferWithLength:bMatrixDataSize
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = bNeedsPadding
                                 ? [device newBufferWithLength:bMpsSize
                                                       options:MTLResourceStorageModeShared]
                                 : bSlice;
        id<MTLBuffer> solBuf = [device newBufferWithLength:bMpsSize
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> unpaddedBuf = bNeedsPadding
                                        ? [device newBufferWithLength:bMatrixDataSize
                                                              options:MTLResourceStorageModeShared]
                                        : solBuf;

        MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
        MPSMatrix* rhsMatrix = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
        MPSMatrix* solutionMatrix = [[MPSMatrix alloc] initWithBuffer:solBuf descriptor:bDesc];

        for (int64_t b = 0; b < batchSize; b++) {
            size_t aOffset = (size_t)b * aMatrixDataSize;
            size_t bOffset = (size_t)b * bMatrixDataSize;

            id<MTLBlitCommandEncoder> blitA = [cmdBuf blitCommandEncoder];
            [blitA copyFromBuffer:inputs[0]
                     sourceOffset:aOffset
                         toBuffer:aSlice
                destinationOffset:0
                             size:aMatrixDataSize];
            [blitA endEncoding];

            id<MTLBlitCommandEncoder> blitB = [cmdBuf blitCommandEncoder];
            [blitB copyFromBuffer:inputs[1]
                     sourceOffset:bOffset
                         toBuffer:bSlice
                destinationOffset:0
                             size:bMatrixDataSize];
            [blitB endEncoding];

            if (aNeedsPadding) {
                memset(aBuf.contents, 0, aMpsSize);
                PadToBuffer(cmdBuf, aSlice, aBuf, n, aDataRowBytes, aMpsRowBytes);
            }
            if (bNeedsPadding) {
                memset(bBuf.contents, 0, bMpsSize);
                PadToBuffer(cmdBuf, bSlice, bBuf, bRows, bDataRowBytes, bMpsRowBytes);
            }

            [solver encodeToCommandBuffer:cmdBuf
                             sourceMatrix:sourceMatrix
                      rightHandSideMatrix:rhsMatrix
                           solutionMatrix:solutionMatrix];

            if (bNeedsPadding) {
                UnpadToBuffer(cmdBuf, solBuf, unpaddedBuf, bRows, bDataRowBytes, bMpsRowBytes);
            }

            size_t dstOffset = (size_t)b * bMatrixDataSize;
            id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
            [blit copyFromBuffer:unpaddedBuf
                     sourceOffset:0
                         toBuffer:outBuf
                destinationOffset:dstOffset
                             size:bMatrixDataSize];
            [blit endEncoding];
        }
    }

    return NativeResult::Buffer(outBuf);
}

REGISTER_NATIVE_MPS_OP("stablehlo.triangular_solve", NativeHandle_triangular_solve);

}  // namespace jax_mps
