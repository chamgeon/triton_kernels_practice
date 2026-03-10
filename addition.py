import torch
import triton
import triton.language as tl

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
DEVICE = torch.device("cuda:0")


# addition kernel
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid*BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask = mask, other = None)
    y = tl.load(y_ptr + offsets, mask = mask, other = None)
    
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


# addition wrapper
def add(x: torch.Tensor, y:torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE,\
        f'DEVICE: {DEVICE}, x.device: {x.device}, y.device: {y.device}, output.device: {output.device}'
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE = 1024)
    return output


# kernel testing
def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)

    out_tri = add(x,y)
    out_ref = x+y

    torch.testing.assert_close(out_tri, out_ref, atol = atol, rtol = rtol)
    print(f"size = {size} test passed.")


# benchmarking, decorator configure benchmarking / plot style
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # arg names for x axis
        x_vals=[2**i for i in range(12, 28, 1)], # values of x args to benchmark
        x_log = True, # makes x-axis logarithmic
        line_arg='provider', # arg name for line
        line_vals=['triton', 'torch'], # values of line_arg to benchmark
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device = DEVICE, dtype = torch.float32)
    y = torch.rand(size, device = DEVICE, dtype = torch.float32)
    quantiles = [0.5, 0.05, 0.95]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x+y, quantiles = quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x,y), quantiles = quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=98432)

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)