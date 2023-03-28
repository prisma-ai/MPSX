for f in $(find $1 -name "*.onnx"); do
    onnxsim $f $f
    python ONNX2MPSX.py --half --input $f --output $f
done;
