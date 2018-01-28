### note

This is the speech demo from mxnet, but for some [reason](https://github.com/apache/incubator-mxnet/pull/9060), they removed this demo. I had contributed timit recipe for this demo, and make it work quite well.

I will update this demo for the latest kaldi and mxnet. currently, we use a old kaldi. we will upgraded to latest in a few day.

thanks for the original author @pluskid and @yzhang87.

### Build Kaldi

Build Kaldi as **shared libraties** if you have not already done so.

```bash
cd kaldi/src
./configure --shared # and other options that you need
make depend
make
```

### Build Python Wrapper

1. Copy or link the attached `python_wrap` folder to `kaldi/src`.
2. Compile python_wrap/

```
cd kaldi/src/python_wrap/
make
```


### usage (simple version:timit)

1. run timit recipe in Kaldi. 
2  modify egs/timit/s5/global.cfg
3. `./run.sh`

every thing should be ok.
