# IPython log file


image = np.random.random((2048, 2048))
from skan.vendored import thresholding
size = 31
size = 31; kernel1 = np.zeros(size+1); kernel1[[0, -1]] = [-1, 1]
kernel2 = np.outer(kernel1, kernel1)
_ = thresholding.correlate_sparse(image, kernel2) # warm up JIT
get_ipython().magic('timeit thresholding.correlate_sparse(image, kernel2)')
get_ipython().magic('timeit ndi.correlate(image, kernel2)')
size = 51; kernel1 = np.zeros(size+1); kernel1[[0, -1]] = [-1, 1]
kernel2 = np.outer(kernel1, kernel1)
get_ipython().magic('timeit ndi.correlate(image, kernel2)')
kernel2.shape
size = 301; kernel1 = np.zeros(size+1); kernel1[[0, -1]] = [-1, 1]
kernel2 = np.outer(kernel1, kernel1)
get_ipython().magic('timeit ndi.correlate(image, kernel2)')
get_ipython().magic('timeit thresholding.correlate_sparse(image, kernel2)')
get_ipython().set_next_input('%lprun -f thresholding._correlate_sparse_offsets');get_ipython().magic('pinfo thresholding._correlate_sparse_offsets')
get_ipython().magic('pinfo thresholding._correlate_sparse_offsets')
get_ipython().magic('lprun -f thresholding._correlate_sparse_offsets thresholding.correlate_sparse(image, kernel2)')
get_ipython().magic('lprun -f thresholding.correlate_sparse thresholding.correlate_sparse(image, kernel2)')
31065.0 + 5
num_loops = 2048 * 2048 * 4
time = 31070.0
time_ns = 31070.0 * 1e3
ns_per_loop = time_ns / num_loops
print('nanoseconds per loop:', ns_per_loop)
