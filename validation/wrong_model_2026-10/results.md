# Results

- **Illustration model on a photo:** posterizes the butterfly wing into flat painted shapes; texture and soft gradients are lost.
- **Photo models on an illustration:** all methods handle simple flat art well. The illustration model gives the cleanest edges; SeedVR2 adds faint blotches to flat color.
- **Clean-trained model on JPEG:** SwinIR classical enlarges the 8x8 compression blocks. Real-world SwinIR and Real-ESRGAN remove them, but the wing's eyespots, destroyed by compression, are not recovered.
- **6 px text:** Real-ESRGAN produces sharp but wrong characters: digits in "$1,284.50" and "0123456789" change. LDSR garbles a few characters. SeedVR2 reads it almost perfectly. Bicubic is blurry but never wrong.
