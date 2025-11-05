import openvino_genai as ov_genai
import openvino as ov


def test_add_extension():
    print(f"== test_add_extension")
    print(ov_genai.get_version())
    tokenizer_path = "/mnt/xiping/mygithub/ov_self_build_model_example/python/custom_op/1_register_kernel/cpu/build/libopenvino_custom_add_extension.so"
    try:
        ov_genai.add_extension(tokenizer_path)
    except:
        assert(False)

if __name__ == "__main__":
    print("OV Version:", ov.get_version())
    test_add_extension()
