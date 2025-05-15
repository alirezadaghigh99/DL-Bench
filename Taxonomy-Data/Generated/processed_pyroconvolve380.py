import torch

def convolve(signal, kernel, mode="full"):
    """
    Computes the 1-d convolution of signal by kernel using FFTs.

    :param torch.Tensor signal: A signal to convolve.
    :param torch.Tensor kernel: A convolution kernel.
    :param str mode: One of: 'full', 'valid', 'same'.
    :return: A tensor with broadcasted shape. Letting ``m = signal.size(-1)``
        and ``n = kernel.size(-1)``, the rightmost size of the result will be:
        ``m + n - 1`` if mode is 'full';
        ``max(m, n) - min(m, n) + 1`` if mode is 'valid'; or
        ``max(m, n)`` if mode is 'same'.
    :rtype torch.Tensor:
    """
    # Ensure the signal and kernel are 1D
    signal = signal.flatten()
    kernel = kernel.flatten()

    m = signal.size(-1)
    n = kernel.size(-1)

    # Determine the size of the FFT
    if mode == 'full':
        fft_size = m + n - 1
    elif mode == 'same':
        fft_size = max(m, n)
    elif mode == 'valid':
        fft_size = max(m, n) - min(m, n) + 1
    else:
        raise ValueError("Mode must be 'full', 'valid', or 'same'.")

    # Compute the FFT of both the signal and the kernel
    signal_fft = torch.fft.fft(signal, n=fft_size)
    kernel_fft = torch.fft.fft(kernel, n=fft_size)

    # Element-wise multiplication in the frequency domain
    result_fft = signal_fft * kernel_fft

    # Inverse FFT to get the convolution result
    result = torch.fft.ifft(result_fft).real

    # Adjust the result based on the mode
    if mode == 'full':
        return result
    elif mode == 'same':
        start = (fft_size - m) // 2
        return result[start:start + m]
    elif mode == 'valid':
        start = n - 1
        end = fft_size - (n - 1)
        return result[start:end]

