# Super-Resolution Upscaler

<img width="2880" height="1704" alt="image" src="https://github.com/user-attachments/assets/aa92fe45-fa30-4276-be3e-7b724eea5510" />

Upscale image in single or batch of files and having features to choose supported onnx upscale models from dropdown and having feature to preview Before and After upscale. The models are downloaded when not available.

# Build on Windows
1. Download the required dll from `https://github.com/microsoft/onnxruntime/releases` and extract `onnxruntime.dll` and put in the root folder of the project
2. Run below command replace `<Path to DLL>` to the directory of the dll
```
    set ORT_DYLIB_PATH=<Path to DLL>\onnxruntime.dll
    cargo build --release
```
3. The program has 3 files `DirectML.dll`, `upscale_npu.exe`, `onnxruntime.dll`, upon run of upscale_npu.exe it will download the models from `https://huggingface.co/Xenova` repositories and use the model to upscale the input file

# Uses Dynamic Loading of ORT
onnxruntime : `https://github.com/microsoft/onnxruntime/releases`
