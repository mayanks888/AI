import lzma

lzc = lzma.LZMACompressor()
dat1 = lzc.compress(spatial_features_np.tobytes())
dat2=lzc.flush()
result = b"".join([dat1, dat2])
with lzma.open("/ssd-disk2/mfe_op/file.xz", "w") as f:
    f.write(result)