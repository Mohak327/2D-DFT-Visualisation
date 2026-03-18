# DFT-Based Steganography Extension

As an extension to the existing 2D DFT visualization project, I will explore using frequency-domain representations of images to embed and recover hidden messages (image steganography via Fourier transforms). The idea is to slightly perturb selected mid-frequency DFT coefficients to encode bits of a message while keeping the reconstructed spatial image visually indistinguishable from the original.

Concretely, the extended scope will include:

- Implementing an encoder that takes an input image, a binary message, and a key specifying which frequency bands to modify, then produces a stego-image via inverse DFT.
- Implementing a corresponding decoder that, given the same key, recovers the embedded bits from the modified coefficients.
- Evaluating trade-offs between capacity, imperceptibility (e.g., PSNR/SSIM), and robustness to basic image operations (compression, mild noise, or blurring).

This extension ties directly back to the original goal of "breaking images into constituent frequencies and reconstructing them," now using that understanding not just for filtering but also for controlled information hiding in the frequency domain.